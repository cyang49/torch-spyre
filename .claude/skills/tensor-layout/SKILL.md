---
name: tensor-layout
description: "Reference for Spyre device tensor layout: SpyreTensorLayout fields (device_size, stride_map), stick tiling rules, stride_map computation, compute_coordinates, device_coords, and how to compute/verify addresses for any host tensor shape. Use when debugging tensor layout issues, wrong addresses, incorrect view propagation, or unexpected stick dimension placement."
---

# Spyre Tensor Layout Reference

Source of truth: `torch_spyre/csrc/spyre_tensor_impl.cpp`, `torch_spyre/_inductor/stickify.py`,
and `torch_spyre/_inductor/pass_utils.py`.
Docs (`docs/source/user_guide/tensors_and_layouts.md`) may lag the code.

---

## Core Data Structure: `SpyreTensorLayout`

```cpp
struct SpyreTensorLayout {
    vector<int64_t> device_size;   // on-device shape (outermost→innermost, stick last)
    vector<int64_t> stride_map;    // device_dim → host stride (-1 = unused/size-1)
    DataFormats     device_dtype;
    // dim_map: internal C++ only, deprecated for Python-layer reasoning
};
```

`device_size` and `stride_map` have the same length = device rank.

**`dim_map` is deprecated** — it exists in the C++ struct but is not exposed to Python
reasoning. All layout analysis in Python uses `device_size`, `stride_map`, and
`device_coords` (symbolic coordinate expressions from `compute_coordinates`).

---

## Stick Rules

- **Stick** = 128 bytes = `elems_per_stick` elements (64 for FP16, 32 for INT32)
- **Stick dimension** = always the **last** device dimension
- `device_size[-1]` is always exactly `elems_per_stick`
- The stick dimension is the **innermost** (fastest-varying) dimension in device memory
- All other device dimensions are in row-major order

---

## Default Layout: `dim_order` and `device_size`

When constructing a layout from a host tensor, a `dim_order` (permutation of host dims)
determines which host dim maps to each device dim. The constructor
`SpyreTensorLayout(host_size, host_stride, dtype, dim_order)` is the primary entry point.

**Default `dim_order`** for an N-dim tensor: `[0, 1, ..., N-1]`

The internal `get_generic_stick_layout(dim_order)` produces the device layout following
the pattern `[dim1, dim2, ..., dimN, dim0, dimN]` — the last host dim becomes the stick,
and the first host dim is placed second-to-last.

**`device_size` computation** (source: `spyre_tensor_impl.cpp:169-191`):

```
stick_host_dim = last entry of dim_order   (the host dim that becomes the stick)
device_size[-1] = elems_per_stick

for each non-stick device position i:
    h = the host dim for that position
    if h == stick_host_dim:
        device_size[i] = ceil(host_size[h] / elems_per_stick)   # stick tiling count
    else:
        device_size[i] = host_size[h]
```

### Concrete example: shape `[2880, 90, 44, 64]`, FP16

`dim_order = [0,1,2,3]`, stick dim = host dim 3 (size 64 = elems_per_stick exactly)

```
device position:  0     1     2          3       4(stick)
host dim mapped:  1     2     3          0       3
device_size:     [90,   44,   64/64=1,   2880,   64]
```

`device_size = [90, 44, 1, 2880, 64]`

---

## `stride_map` Computation

Source: `spyre_tensor_impl.cpp:111-131`. Iterates device dims **right to left**,
tracking accumulated strides per host dim:

```python
stride_map = [-1] * device_rank
last_stride = {}   # host_dim → accumulated stride

for j from device_rank-1 down to 0:
    h = dim_map[j]                       # host dim for device pos j
    if h == -1 or host_size[h] == 1:
        stride_map[j] = -1               # unused: synthetic dim or size-1
    elif h not in last_stride:
        stride_map[j] = host_stride[h]   # first occurrence: actual host stride
    else:
        stride_map[j] = last_stride[h]   # subsequent: accumulated
    if h != -1 and host_size[h] != 1:
        last_stride[h] = stride_map[j] * device_size[j]
```

**What `stride_map[j]` means**: to advance by 1 in device dimension `j`, you advance
by `stride_map[j]` elements in the original host tensor's flat buffer. A value of -1
means this device dim does not correspond to any active host data movement.

### Example: shape `[2880, 90, 44, 64]`, FP16

```
host_stride = [253440, 2816, 64, 1]
device_size = [90, 44, 1, 2880, 64]    (device positions 0..4)

j=4 (stick, h=3): first occurrence → stride_map[4]=1,  last[3]=1*64=64
j=3 (h=0):        first occurrence → stride_map[3]=253440, last[0]=253440*2880
j=2 (h=3, size=1 device, but host_size[3]=64≠1): second occurrence → stride_map[2]=64, last[3]=64*1=64
j=1 (h=2):        first occurrence → stride_map[1]=64,  last[2]=64*44=2816
j=0 (h=1):        first occurrence → stride_map[0]=2816, last[1]=2816*90=253440

stride_map = [2816, 64, 64, 253440, 1]
```

---

## Device Memory Layout and Address Computation

All device dimensions use **row-major implicit strides**:

```
device_stride[i] = prod(device_size[i+1:])
```

For `device_size = [90, 44, 1, 2880, 64]`:
```
device_stride = [8110080, 184320, 184320, 64, 1]
```

**Host flat index → device byte offset** (using `stride_map` to find coordinates):

```python
device_elem_offset = 0
for j in range(device_rank):
    if stride_map[j] <= 0:
        continue
    coord = (host_flat_index // stride_map[j]) % device_size[j]
    device_elem_offset += coord * device_stride[j]
device_byte_offset = device_elem_offset * word_bytes
```

**Total device bytes** = `prod(device_size[:-1]) * 128`

---

## Symbolic Coordinates: `device_coords`

In the inductor pipeline, layouts are reasoned about through **symbolic coordinate
expressions**, not through `stride_map` or `dim_map` directly.

**`compute_coordinates(size, stride, var_ranges, index)`** (`views.py:22-79`):
- `size`, `stride`: the tensor's dimensions and strides (host or device)
- `var_ranges`: the loop variable → range mapping (the iteration space)
- `index`: the flat symbolic index expression into the tensor
- Returns: one symbolic expression per dimension, giving the coordinate in that dim

**`device_coordinates(layout, dep)`** (`pass_utils.py`):
Calls `compute_coordinates` with `device_size` and `stride_map`, returning
`device_coords: list[Expr]` — one per device dim, with `device_coords[-1]` being
the stick coordinate expression.

**`host_coordinates(layout, dep)`** (`pass_utils.py`):
Same but uses host `size` and `stride`.

### Key properties of `device_coords`

- `device_coords[-1]` is the stick coordinate — its free symbol identifies the stick dim
- Each coordinate expression is a **sum of independent single-variable terms**
  (guaranteed by `compute_coordinates` structure)
- A coordinate expression of `0` means this device dim is not varied (broadcast/unused)

### Using `matching_dim` (`views.py:105-120`)

```python
matching_dim(coords, expr)
```

Given a coordinate list and a symbolic expression, returns the unique index `d` where
`coords[d]`'s value range is a superset of `expr`'s value range. Used in `stickify.py`
to find which host dim corresponds to the device's stick expression.

---

## Layout Propagation in `stickify.py`

### Same-access propagation

When input and output tensors have the same host coordinates and index expression,
and same `elems_per_stick`, the existing `SpyreTensorLayout` is reused directly
(copying `device_size`, `dim_map`, `stride_map`, with new `device_dtype`).

### New layout computation

When access patterns differ, a new `dim_order` is built from `host_coordinates`:

```python
dim_order = [d for d in range(N) if d != out_stick_dim and out_coords[d] != 0]  # active dims
dim_order += [d for d in range(N) if d != out_stick_dim and out_coords[d] == 0] # zero dims
dim_order += [out_stick_dim]                                                      # stick last
```

`out_stick_dim` is found via `matching_dim(out_coords, in_device_coords[-1])`.

### Matmul output layout (`stickify.py:269-275`)

For mm/bmm, the output `dim_order` orders the last two dims so the stick dim
(matching `y_stick_expr`) comes last.

---

## Sparse Layout

When `dim_order` ends with `-1` (e.g. for `exx2` reduction outputs):
- `device_size[-2] = 1`, `device_size[-1] = elems_per_stick`
- `stride_map[-1] = -1` (synthetic stick — no host data movement)
- The stick dimension does not correspond to any host dimension

---

## Per-Core Span (for `must_split_vars`)

The memory span of a tensor on a single core is determined by the **outermost device
dimension with per-core size > 1**:

```
span_in_sticks = per_core_size[d_outer] * prod(device_size[d_outer+1:-1])
```

Per-core size of device dim `d` given split assignment `{v → s_v}`:

```python
per_core_max = sum(
    int(coord.subs({u: 0 for u in coord.free_symbols - {v}}).subs(v, it_space[v]//s_v - 1))
    for v in device_coords[d].free_symbols
)
per_core_size = per_core_max + 1
```

(Valid because `compute_coordinates` produces sums of independent single-variable terms,
so zeroing out all but one variable correctly isolates each term's contribution.)

---

## Python Utilities (`layout_utils.py` in this skill)

```bash
python3 .claude/skills/tensor-layout/layout_utils.py --shape 2880 90 44 64 --dtype fp16
```

Computes `device_size`, `stride_map`, and device strides. Does not expose `dim_map`.

---

## Quick Checklist for Layout Debugging

1. **What is the stick dimension?** → `device_coords[-1]` free symbol identifies it
2. **Is `device_size[-1] == elems_per_stick`?** → always must be true
3. **Is `stride_map[-1]`?** → should be 1 for a real stick dim; -1 for sparse
4. **Correct total size?** → `prod(device_size[:-1]) * 128 bytes == host tensor bytes`
5. **Address step when advancing 1 in a device dim `i`?** → `device_stride[i] * word_bytes`
6. **Is the layout propagated or recomputed?** → check `stickify.py` same-access condition
7. **Which host dim is the stick?** → `matching_dim(host_coords, device_coords[-1])`
