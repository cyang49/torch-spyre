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

"""
Compute memory access information for scheduler nodes.

This pass analyzes each operation and computes metadata about how it accesses
memory, including:
- Operation dimension mappings (it_dim_map for each tensor)
- Device coordinate expressions
- Host coordinate expressions
- Memory access patterns

This metadata is used by downstream passes like core_division_planning to make
informed decisions about work distribution.
"""

import logging
from typing import Optional

from torch._inductor.ir import (
    ComputedBuffer,
    Pointwise,
    Reduction,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
)

from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps, host_coordinates, device_coordinates
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("access_info")


class TensorAccessInfo:
    """
    Information about how a tensor is accessed in an operation.
    """

    def __init__(
        self,
        layout: FixedTiledLayout,
        it_dim_map: list[int],
        host_coords: list,
        device_coords: list,
    ):
        self.layout = layout
        self.it_dim_map = it_dim_map  # Maps tensor dims to operation dims
        self.host_coords = host_coords  # Host coordinate expressions
        self.device_coords = device_coords  # Device coordinate expressions


class OpAccessInfo:
    """
    Information about memory access patterns for an entire operation.
    """

    def __init__(
        self,
        op_type: str,
        op_dim_sizes: list[int],
        input_info: list[TensorAccessInfo],
        output_info: TensorAccessInfo,
    ):
        self.op_type = op_type
        self.op_dim_sizes = op_dim_sizes
        self.input_info = input_info
        self.output_info = output_info


def compute_pointwise_access_info(
    n: SchedulerNode, args: list[SchedNodeArg]
) -> OpAccessInfo:
    """
    Compute access info for pointwise operations.

    For pointwise ops, operation dimensions = host dimensions (identity mapping).
    """
    output: FixedTiledLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    ndim = len(output.size)

    # For pointwise, operation dimensions are the same as host dimensions
    op_dim_sizes = [int(output.size[i]) for i in range(ndim)]

    # Compute info for each input
    input_info = []
    for arg in args:
        it_dim_map = list(range(len(arg.layout.size)))  # Identity mapping for pointwise
        host_coords = host_coordinates(arg.layout, arg.dep)
        device_coords = device_coordinates(arg.layout, arg.dep)

        input_info.append(
            TensorAccessInfo(
                layout=arg.layout,
                it_dim_map=it_dim_map,
                host_coords=host_coords,
                device_coords=device_coords,
            )
        )

    # Compute info for output
    output_host_coords = host_coordinates(output, output_dep)
    output_device_coords = device_coordinates(output, output_dep)
    output_info = TensorAccessInfo(
        layout=output,
        it_dim_map=list(range(ndim)),
        host_coords=output_host_coords,
        device_coords=output_device_coords,
    )

    return OpAccessInfo(
        op_type="pointwise",
        op_dim_sizes=op_dim_sizes,
        input_info=input_info,
        output_info=output_info,
    )


def compute_matmul_access_info(
    n: SchedulerNode, args: list[SchedNodeArg]
) -> OpAccessInfo:
    """
    Compute access info for matrix multiplication.

    Operation dimensions: [M, K, N]
    - Input A [M, K]: it_dim_map = [0, 1]
    - Input B [K, N]: it_dim_map = [1, 2]
    - Output [M, N]: it_dim_map = [0, 2]
    """
    assert len(args) == 2, "matmul has exactly 2 inputs"

    # Get operation dimension sizes from host layouts
    from .core_division import get_host_dim_size

    M = get_host_dim_size(args[0].layout, 0)
    K = get_host_dim_size(args[0].layout, 1)
    N = get_host_dim_size(args[1].layout, 1)
    op_dim_sizes = [M, K, N]

    # Input A: [M, K]
    input_a_info = TensorAccessInfo(
        layout=args[0].layout,
        it_dim_map=[0, 1],  # tensor_dim[0]→M, tensor_dim[1]→K
        host_coords=host_coordinates(args[0].layout, args[0].dep),
        device_coords=device_coordinates(args[0].layout, args[0].dep),
    )

    # Input B: [K, N]
    input_b_info = TensorAccessInfo(
        layout=args[1].layout,
        it_dim_map=[1, 2],  # tensor_dim[0]→K, tensor_dim[1]→N
        host_coords=host_coordinates(args[1].layout, args[1].dep),
        device_coords=device_coordinates(args[1].layout, args[1].dep),
    )

    # Output: [M, N]
    output: FixedTiledLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    output_info = TensorAccessInfo(
        layout=output,
        it_dim_map=[0, 2],  # tensor_dim[0]→M, tensor_dim[1]→N
        host_coords=host_coordinates(output, output_dep),
        device_coords=device_coordinates(output, output_dep),
    )

    return OpAccessInfo(
        op_type="matmul",
        op_dim_sizes=op_dim_sizes,
        input_info=[input_a_info, input_b_info],
        output_info=output_info,
    )


def compute_bmm_access_info(n: SchedulerNode, args: list[SchedNodeArg]) -> OpAccessInfo:
    """
    Compute access info for batch matrix multiplication.

    3D BMM: [B, M, K] @ [B, K, N] or [B, M, K] @ [K, N] → [B, M, N]
    Operation dimensions: [B, M, K, N]
    - Input A [B, M, K]: it_dim_map = [0, 1, 2]
    - Input B [B, K, N] or [K, N]: it_dim_map = [0, 2, 3] or [2, 3]
    - Output [B, M, N]: it_dim_map = [0, 1, 3]

    4D BMM: [B1, B2, M, K] @ [B1, B2, K, N] → [B1, B2, M, N]
    Operation dimensions: [B1, B2, M, K, N]
    """
    assert len(args) == 2, "bmm has exactly 2 inputs"

    output: FixedTiledLayout = n.node.get_layout()
    output_dep = next(iter(n.read_writes.writes))
    num_dims = len(args[0].layout.size)

    from .core_division import get_host_dim_size

    if num_dims == 3:
        # 3D BMM
        B = get_host_dim_size(args[0].layout, 0)
        M = get_host_dim_size(args[0].layout, 1)
        K = get_host_dim_size(args[0].layout, 2)
        N = get_host_dim_size(args[1].layout, -1)
        op_dim_sizes = [B, M, K, N]

        # Input A: [B, M, K]
        input_a_info = TensorAccessInfo(
            layout=args[0].layout,
            it_dim_map=[0, 1, 2],
            host_coords=host_coordinates(args[0].layout, args[0].dep),
            device_coords=device_coordinates(args[0].layout, args[0].dep),
        )

        # Input B: [B, K, N] or [K, N]
        if len(args[1].layout.size) == 3:
            it_dim_map_b = [0, 2, 3]
        else:
            it_dim_map_b = [2, 3]

        input_b_info = TensorAccessInfo(
            layout=args[1].layout,
            it_dim_map=it_dim_map_b,
            host_coords=host_coordinates(args[1].layout, args[1].dep),
            device_coords=device_coordinates(args[1].layout, args[1].dep),
        )

        # Output: [B, M, N]
        output_info = TensorAccessInfo(
            layout=output,
            it_dim_map=[0, 1, 3],
            host_coords=host_coordinates(output, output_dep),
            device_coords=device_coordinates(output, output_dep),
        )

    elif num_dims == 4:
        # 4D BMM
        B1 = get_host_dim_size(args[0].layout, 0)
        B2 = get_host_dim_size(args[0].layout, 1)
        M = get_host_dim_size(args[0].layout, 2)
        K = get_host_dim_size(args[0].layout, 3)
        N = get_host_dim_size(args[1].layout, -1)
        op_dim_sizes = [B1, B2, M, K, N]

        # Input A: [B1, B2, M, K]
        input_a_info = TensorAccessInfo(
            layout=args[0].layout,
            it_dim_map=[0, 1, 2, 3],
            host_coords=host_coordinates(args[0].layout, args[0].dep),
            device_coords=device_coordinates(args[0].layout, args[0].dep),
        )

        # Input B: [B1, B2, K, N]
        input_b_info = TensorAccessInfo(
            layout=args[1].layout,
            it_dim_map=[0, 1, 3, 4],
            host_coords=host_coordinates(args[1].layout, args[1].dep),
            device_coords=device_coordinates(args[1].layout, args[1].dep),
        )

        # Output: [B1, B2, M, N]
        output_info = TensorAccessInfo(
            layout=output,
            it_dim_map=[0, 1, 2, 4],
            host_coords=host_coordinates(output, output_dep),
            device_coords=device_coordinates(output, output_dep),
        )

    else:
        raise RuntimeError(f"Unsupported BMM dimension count: {num_dims}")

    return OpAccessInfo(
        op_type="bmm",
        op_dim_sizes=op_dim_sizes,
        input_info=[input_a_info, input_b_info],
        output_info=output_info,
    )


def compute_access_info(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    Compute memory access information for each operation.

    This pass analyzes each scheduler node and attaches metadata about:
    - How each tensor is accessed (it_dim_map, host/device coordinates)
    - Operation dimension structure
    - Memory access patterns

    This metadata is used by downstream passes for optimization decisions.

    Must run after propagate_spyre_tensor_layouts so that FixedTiledLayout
    is already determined for all nodes.
    """
    for n in nodes:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            args = get_mem_deps(n)

            access_info: Optional[OpAccessInfo] = None

            if isinstance(n.node.data, Pointwise):
                access_info = compute_pointwise_access_info(n, args)

            elif isinstance(n.node.data, Reduction):
                red = n.node.data
                if red.reduction_type == MATMUL_REDUCTION_OP:
                    access_info = compute_matmul_access_info(n, args)
                elif red.reduction_type == BATCH_MATMUL_OP:
                    access_info = compute_bmm_access_info(n, args)

            if access_info is not None:
                n.access_info = access_info

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Computed access info for {n.node.get_name()}: "
                        f"op_type={access_info.op_type}, "
                        f"op_sizes={access_info.op_dim_sizes}"
                    )

    return nodes
