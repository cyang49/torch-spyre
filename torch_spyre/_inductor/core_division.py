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
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("core_division")


aten = torch.ops.aten
spyreop = torch.ops.spyre


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


def divide_pointwise_op(n: SchedulerNode, max_cores):
    n.n_cores_used = 1

    if max_cores == 1:
        return

    # Check if access_info is available
    if not hasattr(n, "access_info") or n.access_info is None:
        logger.warning(
            f"No access_info for {n.node.get_name()}, skipping core division"
        )
        return

    if len(n.node.get_outputs()) > 2:
        # Core division currently only implemented for 1 or 2 tensors
        return

    access_info = n.access_info
    output = n.node.get_layout()

    # Check for broadcasts (core division not supported if there are broadcasts)
    for inp_info in access_info.input_info:
        if inp_info.layout.size != output.size:
            return

    # Use pre-computed operation dimension sizes from access_info
    sizes = access_info.op_dim_sizes

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


def divide_reduction_op(n: SchedulerNode, max_cores, enable_splitk=True):
    red: Reduction = n.node.data
    n.n_cores_used = 1

    if max_cores == 1:
        return

    # Check if access_info is available
    if not hasattr(n, "access_info") or n.access_info is None:
        logger.warning(
            f"No access_info for {n.node.get_name()}, skipping core division"
        )
        return

    access_info = n.access_info

    if red.reduction_type == MATMUL_REDUCTION_OP:
        # Operation dimensions: [M, K] @ [K, N] --> [M, N]
        # dim_labels in codegen: ["mb", "in", "out"] = [M, K, N]

        # Use pre-computed op_dim_sizes: [M, K, N]
        sizes = access_info.op_dim_sizes
        assert len(sizes) == 3, f"Expected 3 dims for matmul, got {len(sizes)}"

        # Parallelizable operation dimensions: M, K, and N
        # K has lowest priority (1) - only split when M and N are exhausted
        # Use negative priority to exclude K from splitting when splitk is disabled
        priorities = [3, 1 if enable_splitk else -1, 2]
        splits = multi_dim_core_split(sizes, max_cores, priorities)
        n.n_cores_used = math.prod(splits)

        # Store op_dim_splits directly matching dim_labels = ["mb", "in", "out"]
        n.op_dim_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            M, K, N = sizes
            logger.debug(
                f"matmul work_division: M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                f"splits=[M={splits[0]}, K={splits[1]}, N={splits[2]}]"
            )

    elif red.reduction_type == BATCH_MATMUL_OP:
        # Use pre-computed op_dim_sizes from access_info
        sizes = access_info.op_dim_sizes

        if len(sizes) == 4:
            # 3D BMM: [B, M, K] @ [B, K, N] --> [B, M, N]
            #     or  [B, M, K] @ [K, N] --> [B, M, N]
            # dim_labels in codegen: ["x", "mb", "in", "out"] = [B, M, K, N]

            # Parallelizable operation dimensions: B, M, K, and N
            # K has lowest priority (1) - only split when B, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            priorities = [4, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                B, M, K, N = sizes
                logger.debug(
                    f"bmm_3d work_division: B={B}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B={splits[0]}, M={splits[1]}, K={splits[2]}, N={splits[3]}]"
                )

        elif len(sizes) == 5:
            # 4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] --> [B1, B2, M, N]
            # dim_labels in codegen: ["x", "y", "mb", "in", "out"] = [B1, B2, M, K, N]

            # Parallelizable operation dimensions: B1, B2, M, K, and N
            # K has lowest priority (1) - only split when B1, B2, M, and N are exhausted
            # Use negative priority to exclude K from splitting when splitk is disabled
            # NOTE: split priority can affect numerical error in unit tests
            priorities = [4, 5, 2, 1 if enable_splitk else -1, 3]
            splits = multi_dim_core_split(sizes, max_cores, priorities)
            n.n_cores_used = math.prod(splits)

            # Store op_dim_splits directly matching dim_labels = ["x", "y", "mb", "in", "out"]
            n.op_dim_splits = splits

            if logger.isEnabledFor(logging.DEBUG):
                B1, B2, M, K, N = sizes
                logger.debug(
                    f"bmm_4d work_division: B1={B1}, B2={B2}, M={M}, K={K}, N={N}, cores={n.n_cores_used}, "
                    f"splits=[B1={splits[0]}, B2={splits[1]}, M={splits[2]}, K={splits[3]}, N={splits[4]}]"
                )

        else:
            raise RuntimeError(f"Unsupported BMM dimension count: {len(sizes)}")


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
                divide_pointwise_op(n, max_cores)
            elif isinstance(n.node.data, Reduction):
                divide_reduction_op(n, max_cores)
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
