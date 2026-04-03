#!/usr/bin/env python3
# Spyre tensor layout utilities.
#
# Reproduces the logic from torch_spyre/csrc/spyre_tensor_impl.cpp without
# requiring a Spyre build. Useful for offline debugging of layout issues.
#
# The public API uses device_size and stride_map — not dim_map, which is an
# internal C++ implementation detail being deprecated from Python-layer reasoning.
#
# Usage:
#   python3 layout_utils.py --shape 2880 90 44 64 --dtype fp16
#   python3 layout_utils.py --shape 512 256 --dtype fp16 --dim_order 1 0

import argparse
import math
from dataclasses import dataclass


ELEMS_PER_STICK = {"fp16": 64, "bf16": 64, "fp32": 32, "int32": 32, "int8": 128}
BYTES_PER_STICK = 128


# ---------------------------------------------------------------------------
# Internal helpers (mirror spyre_tensor_impl.cpp; dim_map is internal only)
# ---------------------------------------------------------------------------


def _get_generic_stick_layout(dim_order: list[int]) -> list[int]:
    """Internal: reproduce get_generic_stick_layout from spyre_tensor_impl.cpp:46-80.

    Returns the internal dim_map used to compute device_size and stride_map.
    Not part of the public API — use make_layout() instead.
    """
    rank = len(dim_order)
    if rank == 0:
        return [-1, -1]
    if rank == 1:
        return [dim_order[0], dim_order[0]]
    sparse = dim_order[-1] == -1
    if sparse:
        inner = dim_order[:-1]
        return inner[1:] + [inner[0]] + [-1]
    return dim_order[1:] + [dim_order[0]] + [dim_order[-1]]


def _compute_device_size(
    host_size: list[int], dim_map: list[int], elems_per_stick: int
) -> list[int]:
    """Internal: reproduce device_size computation from spyre_tensor_impl.cpp:169-191."""
    stick_dim = dim_map[-1]
    device_size = [0] * len(dim_map)
    device_size[-1] = elems_per_stick
    sparse = stick_dim == -1
    for i in range(len(dim_map) - 1):
        d = dim_map[i]
        if d == -1:
            device_size[i] = 1
        elif d == stick_dim:
            device_size[i] = 1 if sparse else math.ceil(host_size[d] / elems_per_stick)
        else:
            device_size[i] = host_size[d]
    return device_size


def _compute_host_stride(host_size: list[int]) -> list[int]:
    n = len(host_size)
    stride = [1] * n
    for i in range(n - 2, -1, -1):
        stride[i] = stride[i + 1] * host_size[i + 1]
    return stride


def _compute_stride_map(
    dim_map: list[int],
    host_size: list[int],
    host_stride: list[int],
    device_size: list[int],
) -> list[int]:
    """Internal: reproduce dim_map_to_stride_map from spyre_tensor_impl.cpp:111-131.

    stride_map[j] = the host stride for device dimension j.
    -1 means this device dim is unused (synthetic or size-1 host dim).
    """
    n = len(dim_map)
    stride_map = [-1] * n
    last_stride: dict[int, int] = {}
    for j in range(n - 1, -1, -1):
        d = dim_map[j]
        if d == -1 or host_size[d] == 1:
            stride_map[j] = -1
        elif d not in last_stride:
            stride_map[j] = host_stride[d]
        else:
            stride_map[j] = last_stride[d]
        if d != -1 and host_size[d] != 1:
            last_stride[d] = stride_map[j] * device_size[j]
    return stride_map


def _compute_device_stride(device_size: list[int]) -> list[int]:
    """Row-major implicit strides from device_size."""
    n = len(device_size)
    stride = [1] * n
    for i in range(n - 2, -1, -1):
        stride[i] = stride[i + 1] * device_size[i + 1]
    return stride


# ---------------------------------------------------------------------------
# Public layout descriptor
# ---------------------------------------------------------------------------


@dataclass
class SpyreLayout:
    """Offline representation of a SpyreTensorLayout.

    Fields mirror what Python code uses: device_size, stride_map, device_stride.
    dim_map is intentionally not exposed — it is deprecated for Python-layer reasoning.
    """

    host_size: list[int]
    host_stride: list[int]
    device_size: list[int]
    stride_map: list[int]
    device_stride: list[int]
    elems_per_stick: int
    dtype: str

    @property
    def total_bytes(self) -> int:
        return math.prod(self.device_size[:-1]) * BYTES_PER_STICK

    def print_summary(self) -> None:
        print(f"Host shape:     {self.host_size}")
        print(f"Host stride:    {self.host_stride}")
        print(f"dtype:          {self.dtype}  (elems_per_stick={self.elems_per_stick})")
        print()
        print(f"device_size:    {self.device_size}")
        print(f"device_stride:  {self.device_stride}")
        print(f"stride_map:     {self.stride_map}")
        print(
            "  → stride_map[i] = host stride per unit advance in device dim i (-1 = unused)"
        )
        for i, (sm, ds) in enumerate(zip(self.stride_map, self.device_size)):
            label = "(stick)" if i == len(self.device_size) - 1 else ""
            active = f"host stride {sm}" if sm >= 0 else "unused"
            print(f"    dev[{i}] size={ds}, {active}  {label}")
        print()
        print(f"Total device bytes: {self.total_bytes:,}")
        word_bytes = BYTES_PER_STICK // self.elems_per_stick
        host_bytes = math.prod(self.host_size) * word_bytes
        print(f"Total host bytes:   {host_bytes:,}")
        if self.total_bytes == host_bytes:
            print("  ✓ sizes match")
        else:
            print(f"  ✗ MISMATCH (device={self.total_bytes} vs host={host_bytes})")


def make_layout(
    host_size: list[int],
    dtype: str = "fp16",
    dim_order: list[int] | None = None,
    host_stride: list[int] | None = None,
) -> SpyreLayout:
    """Compute the SpyreLayout (device_size, stride_map) for a host tensor.

    dim_order: custom host dim ordering (default: [0, 1, ..., N-1]).
    host_stride: custom host strides (default: row-major).
    """
    eps = ELEMS_PER_STICK[dtype]
    if dim_order is None:
        dim_order = list(range(len(host_size)))
    if host_stride is None:
        host_stride = _compute_host_stride(host_size)

    # dim_map is internal — used only to compute device_size and stride_map
    dim_map = _get_generic_stick_layout(dim_order)
    device_size = _compute_device_size(host_size, dim_map, eps)
    stride_map = _compute_stride_map(dim_map, host_size, host_stride, device_size)
    device_stride = _compute_device_stride(device_size)

    return SpyreLayout(
        host_size=host_size,
        host_stride=host_stride,
        device_size=device_size,
        stride_map=stride_map,
        device_stride=device_stride,
        elems_per_stick=eps,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Address computation helpers
# ---------------------------------------------------------------------------


def host_flat_index_to_device_offset(layout: SpyreLayout, host_flat_index: int) -> int:
    """Return the device byte offset for a given host flat index.

    Uses stride_map to map from host index to device coordinates, matching
    the semantics of compute_coordinates / device_coordinates in the inductor pipeline.
    """
    device_elem_offset = 0
    for sm, ds, dstride in zip(
        layout.stride_map, layout.device_size, layout.device_stride
    ):
        if sm <= 0:
            continue
        coord = (host_flat_index // sm) % ds
        device_elem_offset += coord * dstride
    word_bytes = BYTES_PER_STICK // layout.elems_per_stick
    return device_elem_offset * word_bytes


def per_core_span_sticks(
    layout: SpyreLayout,
    per_core_device_sizes: list[int],
) -> int:
    """Compute the per-core memory span in sticks.

    per_core_device_sizes: per-core size for each device dim (excluding stick dim).
    The span is determined by the outermost device dim with per-core size > 1.

    Matches the span formula from docs/must_split_vars_rework.md.
    """
    device_size = layout.device_size[:-1]  # exclude stick dim
    for i, (pc_size, _) in enumerate(zip(per_core_device_sizes, device_size)):
        if pc_size > 1:
            stride_in_sticks = math.prod(device_size[i + 1 :])
            return pc_size * stride_in_sticks
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Spyre device layout for a host tensor shape."
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        required=True,
        help="Host tensor shape, e.g. --shape 2880 90 44 64",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=list(ELEMS_PER_STICK.keys()),
        help="Data type (default: fp16)",
    )
    parser.add_argument(
        "--dim_order",
        nargs="+",
        type=int,
        default=None,
        help="Custom host dim ordering (default: 0 1 ... N-1)",
    )
    parser.add_argument(
        "--stride",
        nargs="+",
        type=int,
        default=None,
        help="Custom host strides (default: row-major)",
    )
    args = parser.parse_args()

    layout = make_layout(args.shape, args.dtype, args.dim_order, args.stride)
    layout.print_summary()
