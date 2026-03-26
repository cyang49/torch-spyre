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


import dataclasses
import math
import os
from sympy import Expr, Symbol

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

from torch._inductor.dependencies import MemoryDep

from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps, device_coordinates, iteration_space
from .logging_utils import get_inductor_logger
import logging

logger = get_inductor_logger("core_division")

# Maximum memory access span per core: 256MB hardware limit
MAX_SPAN_BYTES = 256 * 1024 * 1024
MAX_SPAN_STICKS = MAX_SPAN_BYTES // 128

aten = torch.ops.aten
spyreop = torch.ops.spyre


@dataclasses.dataclass
class TensorDep:
    """Bundles a MemoryDep with its FixedTiledLayout and pre-computes device coordinates."""

    dep: MemoryDep
    layout: FixedTiledLayout
    device_coords: list[Expr] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_coords = device_coordinates(self.layout, self.dep)


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


def multi_dim_iteration_space_split(
    iteration_space: dict[Symbol, Expr],
    max_cores: int,
    priorities: list[Symbol],
    min_splits: dict[Symbol, int] | None = None,
) -> dict[Symbol, int]:
    """
    Distribute max_cores across multiple dimensions of an iteration space.

    This function tries to split cores across multiple dimensions to maximize
    parallelism while ensuring even division. It uses a two-pass approach:
    1. First pass: satisfy minimum split requirements (hardware constraints)
    2. Second pass: distribute remaining cores by priority

    Args:
        iteration_space: The iteration space to be parallelized
        max_cores: Total number of cores available
        priorities: Order in which to consider the dimensions
        min_splits: Minimum splits required for each dimension (optional)

    Returns:
        The core splits for the iteration_space
        The product of all splits will be <= max_cores
    """
    splits = {v: 1 for v in iteration_space.keys()}
    n_cores_remaining = max_cores

    # First pass: satisfy minimum split requirements
    if min_splits:
        for var, min_split in min_splits.items():
            # Check if we have enough cores for this minimum split
            if n_cores_remaining // min_split <= 0:
                logger.critical(
                    f"Cannot satisfy minimum split requirement for {var}: "
                    f"need {min_split} splits but only {n_cores_remaining} cores remaining. "
                    f"Skipping this constraint - hardware span limit may be violated."
                )
                continue  # Skip this variable, leave splits[var] = 1

            # Safe to apply the minimum split
            splits[var] = min_split
            n_cores_remaining = n_cores_remaining // min_split

    # Second pass: distribute remaining cores by priority
    for v in priorities:
        if n_cores_remaining <= 1:
            break
        if min_splits and v in min_splits:
            continue  # Already handled in first pass

        best_split = core_split(iteration_space[v], n_cores_remaining)
        if best_split > 1:
            splits[v] = best_split
            n_cores_remaining = n_cores_remaining // best_split

    return splits


def adjust_it_space_for_sticks(
    it_space: dict[Symbol, Expr],
    tensor_deps: list[TensorDep],
) -> None:
    """Adjust iteration space sizes to count sticks rather than elements.

    For each tensor, find the variable that indexes its stick dimension and
    convert its size in it_space from elements to sticks. This ensures core
    division treats sticks as atomic units. Adjusts each variable at most once.
    """
    adjusted: set[Symbol] = set()
    for td in tensor_deps:
        stick_expr = td.device_coords[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        if stick_var in adjusted or stick_var not in it_space:
            continue
        elems_per_stick = td.layout.device_layout.elems_per_stick()
        it_space[stick_var] = (
            it_space[stick_var] + elems_per_stick - 1
        ) // elems_per_stick
        adjusted.add(stick_var)


def must_split_vars(
    tensor_deps: list[TensorDep] | None,
) -> dict[Symbol, int]:
    """
    Return iteration variables that must be split to bring violating tensors'
    memory span within MAX_SPAN_BYTES, along with the minimum number of splits
    required.

    For each tensor whose total device memory span exceeds the limit, find the
    first non-size-1 outer dimension that can be split. Since device layout is
    always row-major, splitting outer dimensions reduces contiguous memory span.

    Span is measured in sticks (128 bytes each). The minimum split is rounded
    up to the nearest divisor of the dimension size so each core gets an equal
    integer-sized slice.

    Returns:
        dict mapping Symbol -> minimum number of splits required to satisfy
        the hardware span constraint, guaranteed to evenly divide the dimension.
    """
    if tensor_deps is None:
        return {}
    result: dict[Symbol, int] = {}
    for td in tensor_deps:
        dl = td.layout.device_layout
        # device_size[-1] is elements per stick (fixed by dtype); all other
        # dims count sticks, so the total stick count excludes the last dim
        total_sticks = math.prod(dl.device_size[:-1])
        if total_sticks <= MAX_SPAN_STICKS:
            continue

        # Find the first splittable outer dimension (non-size-1).
        # Device layout is row-major, so outer dimensions have larger strides
        # and splitting them effectively reduces contiguous memory span.
        for i, coord in enumerate(td.device_coords[:-1]):  # exclude stick dim
            dim_size = dl.device_size[i]
            if dim_size > 1:  # splittable dimension
                assert coord.free_symbols, (
                    f"Device dimension {i} has size {dim_size} > 1 but its "
                    f"coordinate expression has no free symbols: {coord!r}. "
                    f"Cannot determine which iteration variable to split."
                )
                # Minimum splits so that sticks-per-core <= MAX_SPAN_STICKS.
                # Round up to the nearest divisor of dim_size so each core
                # gets an equal, integer-sized slice.
                min_split_raw = math.ceil(total_sticks / MAX_SPAN_STICKS)
                min_split = next(
                    (
                        d
                        for d in range(min_split_raw, dim_size + 1)
                        if dim_size % d == 0
                    ),
                    dim_size,  # fallback: split fully; best effort if dim is prime
                )
                if min_split == dim_size and dim_size < min_split_raw:
                    logger.warning(
                        f"Cannot fully satisfy span limit for dimension {i} "
                        f"(size={dim_size}, need {min_split_raw} splits): "
                        f"using full split of {dim_size}."
                    )
                for var in coord.free_symbols:
                    # If multiple tensors require splits on same var, use max
                    result[var] = max(result.get(var, 1), min_split)
                break

    return result


def prioritize_dimensions(
    output: TensorDep,
    it_space: dict[Symbol, Expr],
    inputs: list[TensorDep] | None = None,
) -> tuple[list[Symbol], dict[Symbol, int]]:
    """
    Return iteration variables in priority order for core division, along with
    minimum split requirements.

    Priority tiers:
      1. Must-split vars: outermost dims of tensors that violate MAX_SPAN_BYTES.
         Splitting these is required to bring memory span within hardware limits.
      2. Remaining output dims (present in output coords), by decreasing size.
      3. Reduction dims (absent from output coords), by decreasing size.

    Returns:
        tuple of (priority list, min_splits dict)
    """
    # Collect free symbols from all output device coords except the stick dim.
    # The stick dim is always the innermost device dimension and shares its host
    # dimension with an outer coord, so its free symbol is already captured here.
    coord_vars = {v for e in output.device_coords[:-1] for v in e.free_symbols}

    all_deps = (inputs + [output]) if inputs is not None else [output]
    min_splits = must_split_vars(all_deps)
    priority = list(min_splits.keys())

    remaining_output = []
    reduction_dims = []
    for s, e in it_space.items():
        if s in min_splits:
            continue
        if s in coord_vars:
            remaining_output.append((s, e))
        else:
            reduction_dims.append((s, e))

    remaining_output.sort(key=lambda t: t[1], reverse=True)
    reduction_dims.sort(key=lambda t: t[1], reverse=True)
    priority += [t[0] for t in remaining_output]
    priority += [t[0] for t in reduction_dims]

    return priority, min_splits


def divide_pointwise_op_new(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    if max_cores == 1:
        return

    it_space = iteration_space(n)
    output_td = TensorDep(next(iter(n.read_writes.writes)), n.node.get_layout())

    adjust_it_space_for_sticks(it_space, [output_td])

    priorities, min_splits = prioritize_dimensions(output_td, it_space)
    splits = multi_dim_iteration_space_split(
        it_space, max_cores, priorities, min_splits
    )

    cores_used = math.prod(splits.values())

    if cores_used > 1:
        n.op_it_space_splits = splits

        # Consolidated DEBUG log for pointwise work division
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"pointwise work_division {n.node.get_name()}: cores={n.n_cores_used}, "
                f"iteration_space={it_space}, priorities={priorities}, "
                f"min_splits={min_splits}, op_it_space_splits={n.op_it_space_splits}"
            )


def divide_reduction_op_new(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    if max_cores == 1:
        return

    red: Reduction = n.node.data
    if red.reduction_type not in (MATMUL_REDUCTION_OP, BATCH_MATMUL_OP):
        return

    it_space = iteration_space(n)
    input_tds = [TensorDep(a.dep, a.layout) for a in args]
    output_td = TensorDep(next(iter(n.read_writes.writes)), n.node.get_layout())

    # Adjust all stick dimension variables (inputs and output) to count sticks
    adjust_it_space_for_sticks(it_space, input_tds + [output_td])

    priorities, min_splits = prioritize_dimensions(output_td, it_space, input_tds)
    splits = multi_dim_iteration_space_split(
        it_space, max_cores, priorities, min_splits
    )

    cores_used = math.prod(splits.values())
    if cores_used > 1:
        n.n_cores_used = cores_used
        n.op_it_space_splits = splits

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"reduction work_division {n.node.get_name()}: cores={cores_used}, "
                f"iteration_space={it_space}, priorities={priorities}, "
                f"min_splits={min_splits}, op_it_space_splits={n.op_it_space_splits}"
            )


def divide_pointwise_op(n: SchedulerNode, args: list[SchedNodeArg], max_cores):
    output: FixedTiledLayout = n.node.get_layout()
    ndim = len(output.size)
    n.n_cores_used = 1

    if max_cores == 1:
        return

    if len(n.node.get_outputs()) > 2:
        # Core division currently only implemented for 1 or 2 tensors
        return

    for a in args:
        if a.layout.size != output.size:
            # Core division not supported if there are broadcasts
            return

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
                divide_pointwise_op_new(n, get_mem_deps(n), max_cores)
            elif isinstance(n.node.data, Reduction):
                divide_reduction_op(n, get_mem_deps(n), max_cores)
                divide_reduction_op_new(n, get_mem_deps(n), max_cores)
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
