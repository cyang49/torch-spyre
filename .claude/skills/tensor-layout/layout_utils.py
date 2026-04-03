#!/usr/bin/env python3
# Spyre tensor layout utilities.
#
# Reproduces the logic from torch_spyre/csrc/spyre_tensor_impl.cpp without
# requiring a Spyre build. Useful for offline debugging of layout issues.
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
# Core layout computation (mirrors spyre_tensor_impl.cpp)
# ---------------------------------------------------------------------------


def get_generic_stick_layout(dim_order: list[int]) -> list[int]:
    """Reproduce get_generic_stick_layout from spyre_tensor_impl.cpp:46-80.

    Given a host dim_order (e.g. [0,1,2,3] for default), returns dim_map.
    Pattern for rank N: [dim1, dim2, ..., dimN, dim0, dimN]
    """
    rank = len(dim_order)
    if rank == 0:
        return [-1, -1]
    if rank == 1:
        return [dim_order[0], dim_order[0]]
    # General pattern: [dim_order[1], ..., dim_order[N-1], dim_order[0], dim_order[N-1]]
    # Exception: if dim_order ends in -1 (sparse), keep -1 at the end as-is.
    sparse = dim_order[-1] == -1
    if sparse:
        inner = dim_order[:-1]
        result = inner[1:] + [inner[0]] + [-1]
    else:
        result = dim_order[1:] + [dim_order[0]] + [dim_order[-1]]
    return result


def compute_device_size(
    host_size: list[int], dim_map: list[int], elems_per_stick: int
) -> list[int]:
    """Reproduce device_size computation from spyre_tensor_impl.cpp:169-191."""
    stick_dim = dim_map[-1]
    device_size = [0] * len(dim_map)
    device_size[-1] = elems_per_stick
    sparse = stick_dim == -1
    elems_in_stick = 1 if sparse else elems_per_stick
    for i in range(len(dim_map) - 1):
        d = dim_map[i]
        if d == -1:
            device_size[i] = 1
        elif d == stick_dim:
            if sparse:
                device_size[i] = 1
            else:
                device_size[i] = math.ceil(host_size[stick_dim] / elems_in_stick)
        else:
            device_size[i] = host_size[d]
    return device_size


def compute_host_stride(host_size: list[int]) -> list[int]:
    n = len(host_size)
    stride = [1] * n
    for i in range(n - 2, -1, -1):
        stride[i] = stride[i + 1] * host_size[i + 1]
    return stride


def compute_stride_map(
    dim_map: list[int],
    host_size: list[int],
    host_stride: list[int],
    device_size: list[int],
) -> list[int]:
    """Reproduce dim_map_to_stride_map from spyre_tensor_impl.cpp:111-131."""
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


def compute_device_stride(device_size: list[int]) -> list[int]:
    """Row-major strides from device_size (standard implicit stride formula)."""
    n = len(device_size)
    stride = [1] * n
    for i in range(n - 2, -1, -1):
        stride[i] = stride[i + 1] * device_size[i + 1]
    return stride


# ---------------------------------------------------------------------------
# High-level layout descriptor
# ---------------------------------------------------------------------------


@dataclass
class SpyreLayout:
    host_size: list[int]
    host_stride: list[int]
    dim_map: list[int]
    device_size: list[int]
    stride_map: list[int]
    device_stride: list[int]
    elems_per_stick: int
    dtype: str

    @property
    def stick_dim(self) -> int:
        return self.dim_map[-1]

    @property
    def total_bytes(self) -> int:
        return math.prod(self.device_size[:-1]) * BYTES_PER_STICK

    def print_summary(self) -> None:
        print(f"Host shape:     {self.host_size}")
        print(f"Host stride:    {self.host_stride}")
        print(f"dtype:          {self.dtype}  (elems_per_stick={self.elems_per_stick})")
        print()
        print(f"dim_map:        {self.dim_map}")
        print("  → device dim → host dim mapping")
        for i, d in enumerate(self.dim_map):
            label = "(stick)" if i == len(self.dim_map) - 1 else ""
            host_label = f"host[{d}]={self.host_size[d]}" if d >= 0 else "synthetic(-1)"
            print(f"    dev[{i}]: {host_label}  {label}")
        print()
        print(f"device_size:    {self.device_size}")
        print(f"device_stride:  {self.device_stride}")
        print(f"stride_map:     {self.stride_map}")
        print("  → host stride each device dim indexes with")
        for i, (sm, dm) in enumerate(zip(self.stride_map, self.device_size)):
            label = "(stick)" if i == len(self.dim_map) - 1 else ""
            print(f"    dev[{i}] size={dm}, stride_map={sm}  {label}")
        print()
        print(f"Total device bytes: {self.total_bytes:,}")
        host_bytes = math.prod(self.host_size) * (
            BYTES_PER_STICK // self.elems_per_stick
        )
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
    """Compute the full SpyreLayout for a host tensor.

    dim_order: custom host dim ordering (default: [0, 1, ..., N-1]).
    host_stride: custom host strides (default: row-major).
    """
    eps = ELEMS_PER_STICK[dtype]
    if dim_order is None:
        dim_order = list(range(len(host_size)))
    if host_stride is None:
        host_stride = compute_host_stride(host_size)

    dim_map = get_generic_stick_layout(dim_order)
    device_size = compute_device_size(host_size, dim_map, eps)
    stride_map = compute_stride_map(dim_map, host_size, host_stride, device_size)
    device_stride = compute_device_stride(device_size)

    return SpyreLayout(
        host_size=host_size,
        host_stride=host_stride,
        dim_map=dim_map,
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

    Useful for verifying that two tensors with different layouts point to the
    same physical data.
    """
    byte_offset = 0
    for j, (d, sm, ds, dstride) in enumerate(
        zip(layout.dim_map, layout.stride_map, layout.device_size, layout.device_stride)
    ):
        if sm <= 0:
            continue
        coord = (host_flat_index // sm) % ds
        byte_offset += coord * dstride
    word_bytes = BYTES_PER_STICK // layout.elems_per_stick
    return byte_offset * word_bytes


def address_step_for_dim(
    layout: SpyreLayout, sdsc_dim_name: str, sdsc_dims: list[str]
) -> int | None:
    """Return the byte step in device memory when the slice index for an SDSC
    dimension increases by 1, given the per-core slice sizes in ss_.

    sdsc_dim_name: the SDSC dimension label (e.g. 'in', 'out', 'mb')
    sdsc_dims: ordered list of SDSC dim names matching layout.dim_map order
               (excluding the stick device dim)

    Returns None if the dimension is not found or not split.
    """
    # This is mostly documentation — the actual check is in validator.py.
    # The step formula (from debugging session):
    #   stick dim:     (ss_[dim] // elems_per_stick) * device_stride[i] * word_bytes
    #   non-stick dim: ss_[dim] * device_stride[i] * word_bytes
    # where i is the device dim index for sdsc_dim_name.
    return None  # placeholder; see validator.py for the live formula


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
