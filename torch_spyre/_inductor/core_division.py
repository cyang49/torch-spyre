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

from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("core_division")


aten = torch.ops.aten
spyreop = torch.ops.spyre

_CORE_OVERFLOW_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB hardware limit per core


def _tensor_divisor(op_dim_splits: list[int], tensor_op_dims: list[int]) -> int:
    """Compute the core divisor for a tensor from its participating op dimensions.

    Each tensor dimension maps to one op dimension; the per-core tensor size is
    total_bytes divided by the product of the splits for those op dimensions.

    Args:
        op_dim_splits: Split factor per op dimension (e.g. [M_split, K_split, N_split])
        tensor_op_dims: Indices into op_dim_splits for each tensor dimension

    Returns:
        Product of the relevant split factors
    """
    return math.prod(op_dim_splits[i] for i in tensor_op_dims)


def _warn_if_per_core_overflow(
    tensors: list[tuple[FixedTiledLayout, int]],
    op_name: str,
) -> None:
    """Warn if any tensor's per-core slice exceeds the hardware memory limit."""
    for layout, divisor in tensors:
        total_bytes = math.prod(layout.size) * layout.dtype.itemsize
        per_core_bytes = total_bytes // divisor
        if per_core_bytes > _CORE_OVERFLOW_LIMIT_BYTES:
            logger.critical(
                f"{op_name}: per-core tensor size {(per_core_bytes / (1024 * 1024)):.2f} MB "
                f"(shape={list(layout.size)}, dtype={layout.dtype}, "
                f"total={(total_bytes / (1024 * 1024)):.2f} MB, divisor={divisor}) "
                f"exceeds hardware limit of {(_CORE_OVERFLOW_LIMIT_BYTES / (1024 * 1024)):.2f} MB"
            )


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


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    output: FixedTiledLayout = n.node.get_layout()
    ndim = len(output.size)
    n.n_cores_used = 1

    can_split = (
        max_cores > 1
        and len(n.node.get_outputs()) <= 2
        and all(a.layout.size == output.size for a in args)
    )

    if can_split:
        # Collect parallelizable sizes for all host dimensions
        # For stick dimension: this returns the number of sticks
        # For non-stick dimensions: this returns the dimension size
        sizes = [get_host_dim_size(output, i) for i in range(ndim)]

        # Use sizes as priorities (larger dimensions get higher priority)
        priorities = sizes.copy()

        # Use multi-dimensional core splitting
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        if n.n_cores_used > 1:
            n.op_dim_splits = splits

            # Consolidated DEBUG log for pointwise work division
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"pointwise work_division {n.node.get_name()}: cores={n.n_cores_used}, "
                    f"sizes={sizes}, priorities={priorities}, op_dim_splits={n.op_dim_splits}"
                )

    # All tensors share the same splits, so divisor is n_cores_used for each.
    tensors = [(output, n.n_cores_used)] + [(a.layout, n.n_cores_used) for a in args]
    _warn_if_per_core_overflow(tensors, n.node.get_name())

    if n.n_cores_used < max_cores:
        logger.debug(
            f"pointwise work_division {n.node.get_name()}: under-utilization "
            f"cores_used={n.n_cores_used}, max_cores={max_cores}"
        )


def divide_reduction_op(
    n: SchedulerNode, args: list[SchedNodeArg], max_cores, enable_splitk=True
):
    red: Reduction = n.node.data
    n.n_cores_used = 1
    can_split = max_cores > 1

    if red.reduction_type == MATMUL_REDUCTION_OP:
        assert len(args) == 2, "matmul has exactly 2 input args"

        # Operation dimensions: [M, K] @ [K, N] --> [M, N]
        # dim_labels in codegen: ["mb", "in", "out"] = [M, K, N]

        # Get operation dimension sizes from host layouts.
        M = get_host_dim_size(args[0].layout, -2)
        K = get_host_dim_size(args[0].layout, -1)
        N = get_host_dim_size(args[1].layout, -1)

        if can_split:
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
        else:
            splits = [1, 1, 1]

        # op_dim_splits = [M_split, K_split, N_split]
        # A [M, K] participates in op dims 0 (M) and 1 (K)
        # B [K, N] participates in op dims 1 (K) and 2 (N)
        # Output [M, N] participates in op dims 0 (M) and 2 (N)
        _warn_if_per_core_overflow(
            [
                (args[0].layout, _tensor_divisor(splits, [0, 1])),
                (args[1].layout, _tensor_divisor(splits, [1, 2])),
                (n.node.get_layout(), _tensor_divisor(splits, [0, 2])),
            ],
            n.node.get_name(),
        )

    elif red.reduction_type == BATCH_MATMUL_OP:
        assert len(args) == 2, "bmm has exactly 2 input args"

        # Determine if this is 3D or 4D BMM based on the number of dimensions
        num_dims = len(args[0].layout.size)

        if num_dims == 3:
            # 3D BMM: [B, M, K] @ [B, K, N] --> [B, M, N]
            #     or  [B, M, K] @ [K, N] --> [B, M, N]
            # dim_labels in codegen: ["x", "mb", "in", "out"] = [B, M, K, N]

            # Get operation dimension sizes from host layouts
            B = get_host_dim_size(args[0].layout, 0)
            M = get_host_dim_size(args[0].layout, -2)
            K = get_host_dim_size(args[0].layout, -1)
            N = get_host_dim_size(args[1].layout, -1)

            if can_split:
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
            else:
                splits = [1, 1, 1, 1]

            # op_dim_splits = [B_split, M_split, K_split, N_split]
            # A [B, M, K] participates in op dims 0 (B), 1 (M), 2 (K)
            # B participates in op dims 2 (K) and 3 (N); additionally op dim 0 (B)
            # when B is not broadcast (i.e. has 3 dimensions instead of 2)
            # Output [B, M, N] participates in op dims 0 (B), 1 (M), 3 (N)
            args1_op_dims = [2, 3] if len(args[1].layout.size) == 2 else [0, 2, 3]
            _warn_if_per_core_overflow(
                [
                    (args[0].layout, _tensor_divisor(splits, [0, 1, 2])),
                    (args[1].layout, _tensor_divisor(splits, args1_op_dims)),
                    (n.node.get_layout(), _tensor_divisor(splits, [0, 1, 3])),
                ],
                n.node.get_name(),
            )

        elif num_dims == 4:
            # 4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] --> [B1, B2, M, N]
            # dim_labels in codegen: ["x", "y", "mb", "in", "out"] = [B1, B2, M, K, N]

            # Get operation dimension sizes from host layouts
            B1 = get_host_dim_size(args[0].layout, 0)
            B2 = get_host_dim_size(args[0].layout, 1)
            M = get_host_dim_size(args[0].layout, -2)
            K = get_host_dim_size(args[0].layout, -1)
            N = get_host_dim_size(args[1].layout, -1)

            if can_split:
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
                splits = [1, 1, 1, 1, 1]

            # op_dim_splits = [B1_split, B2_split, M_split, K_split, N_split]
            # A [B1, B2, M, K] participates in op dims 0 (B1), 1 (B2), 2 (M), 3 (K)
            # B [B1, B2, K, N] participates in op dims 0 (B1), 1 (B2), 3 (K), 4 (N)
            # Output [B1, B2, M, N] participates in op dims 0 (B1), 1 (B2), 2 (M), 4 (N)
            _warn_if_per_core_overflow(
                [
                    (args[0].layout, _tensor_divisor(splits, [0, 1, 2, 3])),
                    (args[1].layout, _tensor_divisor(splits, [0, 1, 3, 4])),
                    (n.node.get_layout(), _tensor_divisor(splits, [0, 1, 2, 4])),
                ],
                n.node.get_name(),
            )

        else:
            raise RuntimeError(f"Unsupported BMM dimension count: {num_dims}")
    else:
        # For other reductions (sum, mean, etc.), check overflow without work division
        assert len(args) == 1  # assuming only 1 input tensor
        _warn_if_per_core_overflow(
            [
                (args[0].layout, n.n_cores_used),
                (n.node.get_layout(), n.n_cores_used),
            ],
            n.node.get_name(),
        )

    if logger.isEnabledFor(logging.DEBUG) and n.n_cores_used < max_cores:
        logger.debug(
            f"reduction work_division {n.node.get_name()}: under-utilization "
            f"cores_used={n.n_cores_used}, max_cores={max_cores}"
        )


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
