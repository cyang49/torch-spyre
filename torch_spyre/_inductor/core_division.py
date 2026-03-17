# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    MultiOutput,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    ExternKernelSchedulerNode,
    SchedulerNode,
    NopKernelSchedulerNode,
)


from .stickify import is_sparse
from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("core_division")


aten = torch.ops.aten
spyreop = torch.ops.spyre


def get_host_dim_size(layout: FixedTiledLayout, host_dim_idx: int) -> int:
    """
    Get the parallelizable size of a host dimension.

    For non-stick dimensions this is simply the dimension size. For the stick
    dimension (the last host dimension), the elements are packed into sticks, so
    the parallelizable unit is the number of sticks rather than the number of
    elements.

    This function properly consults the dim_map to find which device dimension
    corresponds to the requested host dimension, handling tiling and sparse tensors.

    Args:
        layout: The tensor's FixedTiledLayout
        host_dim_idx: The host dimension index (negative indices are supported)

    Returns:
        The number of parallelizable units along this dimension
    """
    if host_dim_idx < 0:
        host_dim_idx = len(layout.size) + host_dim_idx

    assert host_dim_idx < len(layout.size)

    dl = layout.device_layout

    # Use dim_map to find the device dimension that corresponds to this host dimension
    # For tiled dimensions (appearing multiple times in dim_map), we use the first occurrence
    # which corresponds to the outermost device dimension for that host dimension
    try:
        device_dim_idx = dl.dim_map.index(host_dim_idx)
    except ValueError:
        raise RuntimeError(
            f"Host dimension {host_dim_idx} not found in dim_map {dl.dim_map}"
        )

    return dl.device_size[device_dim_idx]


def core_split(size: int, max_cores: int) -> int:
    """
    Find the largest divisor of size that doesn't exceed max_cores.

    Args:
        size: The dimension size to split
        max_cores: Maximum number of cores to use for this dimension

    Returns:
        Number of cores to use (always divides size evenly)
    """
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1


def multi_dim_core_split(
    sizes: list[int], max_cores: int, priorities: list[int] | None = None
) -> list[int]:
    """
    Distribute max_cores across multiple dimensions optimally.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a greedy approach that
    prioritizes dimensions based on:
    1. User-specified priorities (if provided)
    2. Dimension size (larger dimensions get priority)
    3. Divisibility (dimensions that divide evenly get priority)

    Dimensions with negative priorities are excluded from splitting and will
    always have a split value of 1.

    Args:
        sizes: List of dimension sizes that can be parallelized
        max_cores: Total number of cores available
        priorities: Optional list of priority values (higher = more important)
                   If None, uses dimension sizes as priorities.
                   Use negative values to exclude dimensions from splitting.

    Returns:
        List of core splits for each dimension (same length as sizes)
        The product of all splits will be <= max_cores

    Example:
        >>> multi_dim_core_split([128, 64, 32], max_cores=8)
        [4, 2, 1]  # 4*2*1 = 8 cores total

        >>> multi_dim_core_split([100, 50], max_cores=10)
        [5, 2]  # 5*2 = 10 cores total

        >>> multi_dim_core_split([128, 64, 32], max_cores=8, priorities=[3, -1, 2])
        [4, 1, 2]  # Middle dimension excluded from splitting (priority=-1)
    """
    if not sizes:
        return []

    n_dims = len(sizes)
    splits = [1] * n_dims

    # Use provided priorities or default to the sizes of dimensions
    if priorities is None:
        priorities = sizes.copy()

    # Create list of (dimension_index, size, priority) tuples
    # Filter out dimensions with negative priorities (they should not be split)
    dim_info = [
        (i, sizes[i], priorities[i]) for i in range(n_dims) if priorities[i] >= 0
    ]

    # Sort by priority (descending), then by size (descending)
    dim_info.sort(key=lambda x: (x[2], x[1]), reverse=True)

    n_cores_to_split = max_cores

    # Greedy allocation: try to split highest priority dimensions first
    for dim_idx, size, _ in dim_info:
        if n_cores_to_split <= 1:
            break

        # Find the best split for this dimension given n_cores_to_split
        best_split = core_split(size, n_cores_to_split)

        if best_split > 1:
            splits[dim_idx] = best_split
            n_cores_to_split = n_cores_to_split // best_split

    return splits


def if_broadcast_get_safe_dims(
    output: FixedTiledLayout, args: list[SchedNodeArg]
) -> tuple[bool, None | list[bool]]:
    """
    Check if inputs require broadcasting and determine safe dimensions for parallelization.

    Returns (False, None) for non-broadcast shape mismatches (e.g., squeeze/view operations).
    Returns (True, safe_dims) for actual broadcasts, where safe_dims indicates which
    output dimensions can be safely parallelized.

    Broadcasting aligns dimensions from the right (trailing dimensions).
    """
    ndim = len(output.size)
    safe_dims = [True] * ndim
    is_broadcast_dim = [False] * ndim

    for arg in args:
        arg_ndim = len(arg.layout.size)

        if arg_ndim > ndim:
            return False, None

        dim_offset = ndim - arg_ndim

        for out_dim_idx in range(ndim):
            arg_dim_idx = out_dim_idx - dim_offset

            # Implicit broadcast dimension (input has fewer dims than output)
            # Mark as unsafe because codegen doesn't correctly handle address calculation
            # when work is divided across cores for these dimensions
            if arg_dim_idx < 0:
                is_broadcast_dim[out_dim_idx] = True
                safe_dims[out_dim_idx] = False
                continue

            output_size = output.size[out_dim_idx]
            arg_size = arg.layout.size[arg_dim_idx]

            if arg_size == 1 and output_size > 1:
                is_broadcast_dim[out_dim_idx] = True

            if arg_size != output_size and arg_size != 1:
                return False, None

    has_broadcast = any(is_broadcast_dim)
    return has_broadcast, safe_dims if has_broadcast else None


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    output: FixedTiledLayout = n.node.get_layout()
    ndim = len(output.size)
    n.n_cores_used = 1

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"divide_pointwise_op {n.node.get_name()}: output.size={output.size}"
        )
        for i, a in enumerate(args):
            logger.debug(f"  arg[{i}].layout.size={a.layout.size}")

    if max_cores == 1:
        return

    if len(n.node.get_outputs()) > 2:
        # Core division currently only implemented for 1 or 2 tensors
        return

    has_shape_mismatch = any(a.layout.size != output.size for a in args)

    if has_shape_mismatch:
        has_broadcast, safe_dims = if_broadcast_get_safe_dims(output, args)

        if not has_broadcast:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"pointwise {n.node.get_name()}: disabling parallelization due to "
                    f"non-broadcast shape mismatch (output.size={output.size}, "
                    f"arg sizes={[a.layout.size for a in args]})"
                )
            return

        if is_sparse(output.device_layout) or any(
            is_sparse(a.layout.device_layout) for a in args
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"pointwise {n.node.get_name()}: disabling parallelization due to "
                    f"sparse broadcast (not yet supported)"
                )
            return

    # Collect parallelizable sizes for all host dimensions
    sizes = [get_host_dim_size(output, i) for i in range(ndim)]

    # Use dimension index as base priority (earlier dimensions get higher priority)
    priorities = list(range(ndim, 0, -1))

    if has_shape_mismatch:
        # We know it's a broadcast at this point - use safe_dims for priority adjustment
        # Set negative priority for unsafe dimensions to exclude them from splitting
        for i in range(ndim):
            assert safe_dims is not None
            if not safe_dims[i]:
                priorities[i] = -1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise broadcast for {n.node.get_name()}: safe_dims={safe_dims}"
            )

    splits = multi_dim_core_split(sizes, max_cores, priorities)
    n.n_cores_used = math.prod(splits)

    if n.n_cores_used > 1:
        n.op_dim_splits = splits
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise work_division {n.node.get_name()}: cores={n.n_cores_used}, "
                f"sizes={sizes}, priorities={priorities}, op_dim_splits={n.op_dim_splits}"
            )


def divide_reduction_op(
    n: SchedulerNode, args: list[SchedNodeArg], max_cores, enable_splitk=True
):
    red: Reduction = n.node.data
    n.n_cores_used = 1

    if max_cores == 1:
        return

    if red.reduction_type == MATMUL_REDUCTION_OP:
        assert len(args) == 2, "matmul has exactly 2 input args"

        # Operation dimensions: [M, K] @ [K, N] --> [M, N]
        # dim_labels in codegen: ["mb", "in", "out"] = [M, K, N]

        # Get operation dimension sizes from host layouts.
        M = get_host_dim_size(args[0].layout, 0)
        K = get_host_dim_size(args[0].layout, 1)
        N = get_host_dim_size(args[1].layout, 1)

        # Parallelizable operation dimensions: M, K, and N
        # K has lowest priority (1) - only split when M and N are exhausted
        # Use negative priority to exclude K from splitting when splitk is disabled
        sizes = [M, K, N]
        priorities = [3, 1 if enable_splitk else -1, 2]
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Store op_dim_splits directly matching dim_labels = ["mb", "in", "out"]
        n.op_dim_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"matmul work_division: M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                f"splits=[M={splits[0]}, K={splits[1]}, N={splits[2]}]"
            )

    if red.reduction_type == BATCH_MATMUL_OP:
        assert len(args) == 2, "bmm has exactly 2 input args"

        # Determine if this is 3D or 4D BMM based on the number of dimensions
        num_dims = len(args[0].layout.size)

        if num_dims == 3:
            # 3D BMM: [B, M, K] @ [B, K, N] --> [B, M, N]
            #     or  [B, M, K] @ [K, N] --> [B, M, N]
            # dim_labels in codegen: ["x", "mb", "in", "out"] = [B, M, K, N]

            # Get operation dimension sizes from host layouts
            B = get_host_dim_size(args[0].layout, 0)
            M = get_host_dim_size(args[0].layout, 1)
            K = get_host_dim_size(args[0].layout, 2)
            N = get_host_dim_size(args[1].layout, -1)

            # Parallelizable operation dimensions: B, M, K, and N
            # K has lowest priority (1) - only split when B, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            sizes = [B, M, K, N]
            priorities = [4, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"bmm_3d work_division: B={B}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B={splits[0]}, M={splits[1]}, K={splits[2]}, N={splits[3]}]"
                )

        elif num_dims == 4:
            # 4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] --> [B1, B2, M, N]
            # dim_labels in codegen: ["x", "y", "mb", "in", "out"] = [B1, B2, M, K, N]

            # Get operation dimension sizes from host layouts
            B1 = get_host_dim_size(args[0].layout, 0)
            B2 = get_host_dim_size(args[0].layout, 1)
            M = get_host_dim_size(args[0].layout, 2)
            K = get_host_dim_size(args[0].layout, 3)
            N = get_host_dim_size(args[1].layout, -1)

            # Parallelizable operation dimensions: B1, B2, M, K, and N
            # K has lowest priority (1) - only split when B1, B2, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            # NOTE: split priority can affect numerical error in unit tests
            sizes = [B1, B2, M, K, N]
            priorities = [4, 5, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "y", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"bmm_4d work_division: B1={B1}, B2={B2}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B1={splits[0]}, B2={splits[1]}, M={splits[2]}, K={splits[3]}, N={splits[4]}]"
                )

        else:
            raise RuntimeError(f"Unsupported BMM dimension count: {num_dims}")


def core_division_planning(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    # Nodes are in topological order (guaranteed by caller).
    max_cores = int(os.getenv("SENCORES", "32"))
    if max_cores > 32 or max_cores < 1:
        raise Unsupported(f"invalid SENCORES value {max_cores}")

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            if isinstance(n.node.data, Pointwise):
                divide_pointwise_op(n, get_mem_deps(n), max_cores)
            elif isinstance(n.node.data, Reduction):
                divide_reduction_op(n, get_mem_deps(n), max_cores)
            else:
                # Core division not supported on other IRNode types
                pass
        elif isinstance(n, ExternKernelSchedulerNode):
            if isinstance(n.node, FallbackKernel):
                n = next(it, None)
                if not (
                    isinstance(n, ExternKernelSchedulerNode)
                    and isinstance(n.node, MultiOutput)
                ):
                    raise RuntimeError("FallbackKernel must be followed by MultiOutput")

                # Core division not supported on fallback kernels
                pass
            else:
                logger.warning(f"unhandled node type {type(n.node)}")
        elif isinstance(n, NopKernelSchedulerNode):
            pass
        else:
            logger.warning(f"unhandled scheduler node type {type(n)}")

    return nodes
