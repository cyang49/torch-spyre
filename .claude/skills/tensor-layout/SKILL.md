---
name: tensor-layout
description: "Reference for Spyre device tensor layout: SpyreTensorLayout fields (device_size, dim_map, stride_map), stick tiling rules, get_generic_stick_layout, stride_map computation, compute_coordinates, and how to compute/verify addresses for any host tensor shape. Use when debugging tensor layout issues, wrong addresses, incorrect view propagation, or unexpected stick dimension placement."
---

# Spyre Tensor Layout Reference

Source of truth: `torch_spyre/csrc/spyre_tensor_impl.cpp` and `torch_spyre/_inductor/stickify.py`.
Docs (`docs/source/user_guide/tensors_and_layouts.md`) may lag the code.

---

## Core Data Structure: `SpyreTensorLayout`

```cpp
struct SpyreTensorLayout {
    vector<int64_t> device_size;   // on-device shape
    vector<int32_t> dim_map;       // device_dim → host_dim (-1 = synthetic)
    vector<int64_t> stride_map;    // device_dim → host stride (-1 = unused)
    DataFormats     device_dtype;
};
```

All three vectors have the same length = device rank.

---

## Stick Rules

- **Stick** = 128 bytes = `elems_per_stick` elements (64 for FP16, 32 for INT32, etc.)
- **Stick dimension** = `dim_map.back()` = always the last device dimension
- `device_size[-1]` is always exactly `elems_per_stick`
- The stick dimension is the **innermost** (fastest-varying) dimension in device memory

---

## `get_generic_stick_layout` — Default dim_map

Source: `spyre_tensor_impl.cpp:46-80`. Given `host_dim_order = [0,1,...,N-1]`:

| Host rank | `dim_map` (in order) |
|-----------|---------------------|
| 1 | `[0, 0]` |
| 2 | `[1, 0, 1]` |
| 3 | `[1, 2, 0, 2]` |
| 4 | `[1, 2, 3, 0, 3]` |
| 5 | `[1, 2, 3, 4, 0, 4]` |

Pattern: `[dim1, dim2, ..., dimN, dim0, dimN]`

- `dim0` (the first host dimension — the "batch" or outermost) appears second-to-last
- `dimN` (the last host dimension — the "feature" / stick candidate) appears **twice**: once near the front (as a tiling count) and once last (as the actual stick)
- The number of sticks: `device_size[position of first dimN appearance] = ceil(host_size[N] / elems_per_stick)`
- `device_size[-1] = elems_per_stick` always

### Concrete example: shape `[2880, 90, 44, 64]`, FP16

`host_dim_order = [0,1,2,3]` → `dim_map = [1, 2, 3, 0, 3]`

```
dim_map:   [1,    2,   3,       0,     3  ]
               host dim →  1    2    3        0     3(stick)
device_size:  [90,  44,  64/64,  2880,  64 ]
           =  [90,  44,  1,      2880,  64 ]
```

Since `host_size[3] = 64 = elems_per_stick`, the tiling count for dim3 is exactly 1.
So `device_size = [90, 44, 1, 2880, 64]`.

---

## `device_size` Computation

Source: `spyre_tensor_impl.cpp:169-191`

```
stick_dim = dim_map[-1]
device_size[-1] = elems_per_stick

for i in range(len(dim_map) - 1):
    d = dim_map[i]
    if d == stick_dim:
        device_size[i] = ceil(host_size[stick_dim] / elems_per_stick)
    else:
        device_size[i] = host_size[d]
```

For sparse layouts (`dim_order` ends in `-1`): the tiling count for stick is 1, not ceil.

---

## `stride_map` Computation

Source: `spyre_tensor_impl.cpp:111-131`. Iterates device dims **right to left**:

```python
stride_map = [-1] * len(dim_map)
last_stride = {}  # host_dim → accumulated stride so far

for j in range(len(dim_map) - 1, -1, -1):
    d = dim_map[j]
    if d == -1 or host_size[d] == 1:
        stride_map[j] = -1
    elif d not in last_stride:
        stride_map[j] = host_stride[d]          # first occurrence: use actual host stride
    else:
        stride_map[j] = last_stride[d]           # subsequent occurrence: use accumulated
    if d != -1 and host_size[d] != 1:
        last_stride[d] = stride_map[j] * device_size[j]
```

**Key insight**: when `dim_map[i] == dim_map[-1] == stick_dim` (the tiling count position),
`stride_map[i]` gets the accumulated stride = `host_stride[stick_dim] * device_size[-1]`
= `host_stride[stick_dim] * elems_per_stick`.

### Example: shape `[2880, 90, 44, 64]`, FP16

```
host_stride = [90*44*64, 44*64, 64, 1] = [253440, 2816, 64, 1]
dim_map     = [1, 2, 3, 0, 3]
device_size = [90, 44, 1, 2880, 64]

Iterate j = 4 (stick, d=3): first occurrence of 3
    stride_map[4] = host_stride[3] = 1
    last_stride[3] = 1 * 64 = 64

Iterate j = 3 (d=0): first occurrence of 0
    stride_map[3] = host_stride[0] = 253440
    last_stride[0] = 253440 * 2880 = (doesn't matter here)

Iterate j = 2 (d=3): second occurrence of 3, device_size[2]=1
    host_size[3]=64, not 1, so:
    stride_map[2] = last_stride[3] = 64
    last_stride[3] = 64 * 1 = 64

Iterate j = 1 (d=2): first occurrence of 2
    stride_map[1] = host_stride[2] = 64
    last_stride[2] = 64 * 44 = 2816

Iterate j = 0 (d=1): first occurrence of 1
    stride_map[0] = host_stride[1] = 2816
    last_stride[1] = 2816 * 90 = 253440

Result: stride_map = [2816, 64, 64, 253440, 1]
```

---

## Device Memory Layout and Strides

All device dimensions are in **row-major** order. The implicit stride of device dimension `i` is:

```
device_stride[i] = prod(device_size[i+1:])
```

So for `device_size = [90, 44, 1, 2880, 64]`:
```
device_stride = [44*1*2880*64, 1*2880*64, 2880*64, 64, 1]
              = [8,110,080,    184,320,    184,320, 64, 1]
```

Total device bytes = `device_stride[0] * device_size[0] * 2` (for FP16)
                   = `8,110,080 * 90 * 2` = 1,459,814,400 bytes ✓ (= 2880×90×44×64×2)

---

## Host ↔ Device Address Mapping

A host element at flat host index `h` maps to device byte offset:

```
device_byte_offset = sum(
    (h // host_stride[dim_map[j]] % host_size[dim_map[j]])   # host coordinate for device dim j
    * device_stride[j]
    for j where stride_map[j] > 0
) * wordLength
```

Equivalently, using `stride_map` (the host stride per device dim):

```python
device_elem_offset = 0
for j in range(len(dim_map)):
    if stride_map[j] <= 0:
        continue
    # flat host index → coordinate along dim_map[j]
    # using stride_map[j] = host_stride[dim_map[j]] or its accumulation
    coord = (h // stride_map[j]) % device_size[j]
    device_elem_offset += coord * device_stride[j]
```

---

## Layout Propagation in Stickify (`stickify.py`)

### Same-access propagation

If input and output tensors have identical host coordinates and index expressions, and
same `elems_per_stick`, the `device_layout` is directly copied (device_size, dim_map,
stride_map all reused with new `device_dtype`).

### New layout computation

When access patterns differ, a new `dim_order` is built:
1. Non-zero-accessed non-stick dims (active data dims), in host dimension order
2. Zero-accessed non-stick dims (broadcast/unused dims)
3. The stick dim last

Then `SpyreTensorLayout(size, stride, dtype, dim_order)` is called, which re-runs
`get_generic_stick_layout(dim_order)` and recomputes `device_size` and `stride_map`.

### Matmul output layout (`stickify.py:269-275`)

For mm/bmm: the output `dim_order` is `[0, ..., N-3, second_to_last, last]` where the
last two dimensions are ordered so the stick dimension (`y_stick_expr`) comes last.

---

## Sparse Layout

When `dim_order` ends with `-1` (e.g. for `exx2` reduction outputs):
- `dim_map` will end with `-1`
- `device_size[-2] = 1` (one sparse stick tile), `device_size[-1] = elems_per_stick`
- The stick dimension is synthetic — no host dimension corresponds to it

---

## Python Utilities (`layout_utils.py` in this skill)

```bash
python3 .claude/skills/tensor-layout/layout_utils.py --shape 2880 90 44 64 --dtype fp16
```

Computes and prints `dim_map`, `device_size`, `stride_map`, and `device_stride` for any
host tensor shape, matching the logic in `spyre_tensor_impl.cpp`.

---

## Quick Checklist for Layout Debugging

1. **What is the stick dimension?** → `dim_map[-1]` = last host dim in default layout
2. **How many sticks along the stick dim?** → `device_size[position where dim_map[i] == stick_dim and i != last]`
3. **Is `device_size[-1] == elems_per_stick`?** → always must be true
4. **Do `stride_map` values match host strides?** → use `stride_map` computation above
5. **Correct total size?** → `prod(device_size[:-1]) * 128 bytes` must equal host tensor bytes
6. **Is the layout propagated or recomputed?** → check `stickify.py` same-access condition
