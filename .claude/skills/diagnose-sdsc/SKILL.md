---
name: diagnose-sdsc
description: "Validate SDSC JSON files for internal consistency. Checks N_/ss_/numWkSlicesPerDim_ ratios, coreIdToWkSlice_ mapping, startAddressCoreCorelet_ byte offsets, and coordinate fold factor products. Run via validator.py or adapt checks inline."
---

# SDSC Diagnostic Skill

Use this skill when asked to debug or validate a generated SDSC JSON file.

## Quick Start

```bash
python3 .claude/skills/diagnose-sdsc/validator.py <path/to/sdsc.json>
```

The script prints a per-tensor, per-check PASS/FAIL report with details on failures.

---

## Layout Semantics

### `layoutDimOrder_` = [d0, d1, ..., d_n]

Listed **innermost → outermost** in memory. The first entry (d0) is the fastest-changing
dimension (stride = 1 at the element/stick level); the last entry (d_n) is slowest.

### Stick dimension

`stickDimOrder_` names the dimension that is packed into 128-byte aligned sticks.
`stickSize_[0]` = elements per stick (64 for FP16).

The stick dimension is always the **innermost** (first) entry of `layoutDimOrder_`.

### Row-major stride formula (from `_calculate_device_stride`, `compute_ops.py:167`)

Given `layoutDimOrder_ = [d0, d1, ..., d_n]` and per-core sizes `ss_ = {d0:s0, d1:s1, ...}`:

```
stride(d0) = 1              (innermost; this is the stick or elem dimension)
stride(d1) = s0
stride(d2) = s0 * s1
stride(d_k) = product(s0 .. s_{k-1})
```

Step in bytes when one core's slice index for `d_k` increases by 1:
```
step(d_k) = stride(d_k) * ss_[d_k] * wordLength
```

### `scale_` in `labeledDs_`

For each dim in `layoutDimOrder_`:
- `1`  → the core computes its own slice of this dim (address varies per core)
- `-1` → reduced/broadcast dim (ignored in address offset formula)
- `-2` → stick-reduction dim

---

## Four Invariants

### 1. N_ / ss_ / numWkSlicesPerDim_ consistency

```
ss_[d] == N_[d] // numWkSlicesPerDim_[d]    for every dimension d
numCoresUsed_ == product(numWkSlicesPerDim_.values())
```

### 2. coreIdToWkSlice_ mapping

Derived by the modular formula (from `_get_core_to_slice_mapping`, `superdsc.py:132`):

```
inner = 1
for each dim d (in iteration_space order):
  if splits[d] == 1:
    slice_idx[d] = 0
  elif inner == 1:
    slice_idx[d] = core_id % splits[d]
  else:
    slice_idx[d] = (core_id // inner) % splits[d]
  inner *= splits[d]
```

### 3. startAddressCoreCorelet_ offsets

For each tensor, find pairs of cores that differ by exactly 1 in one dimension's slice
index. The byte difference must equal `step(d_k)` as defined above.

Scales with value ≤ 0 do not contribute to the address offset.

### 4. Coordinate fold factors

**Non-stick dim** (`elemArr == 1`):
```
dim_prop_attr factors: [nsplits, 1, 1, per_core_size]
nsplits == numWkSlicesPerDim_[d]
per_core_size == ss_[d] // nsplits
product == ss_[d]
```

**Stick dim, not reduction** (`elemArr == 2`, last alpha ≠ 0):
```
dim_prop_attr factors: [nsplits, 1, 1, num_sticks, stick_size]
nsplits == numWkSlicesPerDim_[d]
num_sticks == ss_[d] // nsplits // stick_size
product == ss_[d]
alpha on core_fold == ss_[d] // nsplits
alpha on elem_arr_1 == stick_size
alpha on elem_arr_0 == 1
```

**Stick dim, reduction** (`elemArr == 2`, last alpha == 0, i.e. `is_stick_reduction`):
```
nsplits == 1 (core_fold forced to 1 in codegen)
dim_prop_attr factors: [1, 1, 1, 1, stick_size]
alpha on core_fold == stick_size, alpha on last == 0
```

---

## Using validator.py in Claude Sessions

Instead of re-deriving formulas each time, do:

```python
# Read and run the validator inline
exec(open('.claude/skills/diagnose-sdsc/validator.py').read())
issues = diagnose_sdsc('path/to/sdsc_0.json')
```

Or import individual check functions:

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "validator", ".claude/skills/diagnose-sdsc/validator.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
issues = v.check_start_addresses(dsc, top_level, word_length=2)
```
