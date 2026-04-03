#!/usr/bin/env python3
# SDSC JSON diagnostic validator.
#
# Checks four invariants:
#   1. N_ / ss_ / numWkSlicesPerDim_ consistency
#   2. coreIdToWkSlice_ mapping matches the modular formula
#   3. startAddressCoreCorelet_ byte offsets match stride formula
#   4. Coordinate fold factor products equal N_[dim]
#
# Usage:
#   python3 validator.py <sdsc.json>

import argparse
import json
import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    severity: str  # "ERROR" or "WARN"
    location: str  # e.g. "Tensor1/in" or "core 3/mb"
    message: str


# ---------------------------------------------------------------------------
# SDSC loader + helpers
# ---------------------------------------------------------------------------


def load_sdsc(path: str) -> tuple[str, dict, dict]:
    """Return (op_name, top_level_dict, inner_dsc_dict).

    top_level: numWkSlicesPerDim_, coreIdToWkSlice_, numCoresUsed_, etc.
    dsc: inner dict under dscs_[0][op_name].
    """
    with open(path) as f:
        root = json.load(f)
    op_name = next(iter(root))
    top_level = root[op_name]
    dsc = top_level["dscs_"][0][op_name]
    return op_name, top_level, dsc


def _n(dsc: dict) -> dict[str, int]:
    return {k.rstrip("_"): v for k, v in dsc["N_"].items() if k != "name_"}


def _ss(dsc: dict) -> dict[str, int]:
    return {
        k.rstrip("_"): v
        for k, v in dsc["dataStageParam_"]["0"]["ss_"].items()
        if k != "name_"
    }


def _primary_info(dsc: dict, layout_label: str) -> dict:
    return dsc.get("primaryDsInfo_", {}).get(layout_label, {})


def _stick_dims_and_size(primary_info: dict) -> tuple[set[str], int | None]:
    stick_dims = set(primary_info.get("stickDimOrder_", []))
    sizes = primary_info.get("stickSize_", [None])
    return stick_dims, (sizes[0] if sizes else None)


# ---------------------------------------------------------------------------
# Stride computation
# ---------------------------------------------------------------------------


def _compute_strides(
    layout_dim_order: list[str],
    n: dict[str, int],
    stick_dims: set[str],
    stick_size: int,
) -> dict[str, int]:
    """Compute per-element strides from layoutDimOrder_ (innermost first).

    Spyre memory layout: iterate over each dimension in layoutDimOrder_ order
    (d0 innermost). Each "unit" is one stick = stick_size elements.

    device_dim_size[d] = N_[d] // stick_size  if d is a stick dim
                       = N_[d]                 otherwise

    stride(d_k) in elements = prod(device_dim_size[d_i], i<k) * stick_size
    """
    strides: dict[str, int] = {}
    product_in_sticks = 1
    for dim in layout_dim_order:
        strides[dim] = product_in_sticks * stick_size
        dev_size = n[dim] // stick_size if dim in stick_dims else n[dim]
        product_in_sticks *= dev_size
    return strides


# ---------------------------------------------------------------------------
# Check 1: N_ / ss_ / numWkSlicesPerDim_ consistency
# ---------------------------------------------------------------------------


def check_n_ss_slices(dsc: dict, top_level: dict) -> list[Issue]:
    issues: list[Issue] = []
    n = _n(dsc)
    ss = _ss(dsc)
    slices = top_level["numWkSlicesPerDim_"]
    num_cores = top_level["numCoresUsed_"]

    for dim in n:
        if dim not in slices:
            issues.append(
                Issue(
                    "ERROR",
                    f"N_/{dim}",
                    f"dim '{dim}' in N_ but missing from numWkSlicesPerDim_",
                )
            )
            continue
        if dim not in ss:
            issues.append(
                Issue("ERROR", f"ss_/{dim}", f"dim '{dim}' in N_ but missing from ss_")
            )
            continue
        expected_ss = n[dim] // slices[dim]
        if ss[dim] != expected_ss:
            issues.append(
                Issue(
                    "ERROR",
                    f"ss_/{dim}",
                    f"ss_={ss[dim]} but N_={n[dim]} // slices={slices[dim]} "
                    f"=> expected {expected_ss}",
                )
            )

    expected_cores = math.prod(slices.values())
    if num_cores != expected_cores:
        issues.append(
            Issue(
                "ERROR",
                "numCoresUsed_",
                f"numCoresUsed_={num_cores} but "
                f"product(numWkSlicesPerDim_)={expected_cores}",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check 2: coreIdToWkSlice_ mapping
# ---------------------------------------------------------------------------


def _expected_wk_slice(
    core_id: int, dims: list[str], splits: dict[str, int]
) -> dict[str, int]:
    """Reproduce the modular formula from _get_core_to_slice_mapping (superdsc.py:132)."""
    result = {}
    inner = 1
    for dim in dims:
        n = splits[dim]
        if n == 1:
            result[dim] = 0
        elif inner == 1:
            result[dim] = core_id % n
        else:
            result[dim] = (core_id // inner) % n
        inner *= n
    return result


def check_wk_slice_mapping(top_level: dict) -> list[Issue]:
    issues: list[Issue] = []
    slices = top_level["numWkSlicesPerDim_"]
    dims = list(slices.keys())
    wk_map = top_level["coreIdToWkSlice_"]
    num_cores = top_level["numCoresUsed_"]

    for c in range(num_cores):
        expected = _expected_wk_slice(c, dims, slices)
        actual = {k: int(v) for k, v in wk_map[str(c)].items()}
        for dim in dims:
            if actual.get(dim) != expected.get(dim):
                issues.append(
                    Issue(
                        "ERROR",
                        f"core {c}/{dim}",
                        f"coreIdToWkSlice_={actual.get(dim)} "
                        f"but formula gives {expected.get(dim)}",
                    )
                )
    return issues


# ---------------------------------------------------------------------------
# Check 3: startAddressCoreCorelet_ offsets
# ---------------------------------------------------------------------------


def check_start_addresses(dsc: dict, top_level: dict) -> list[Issue]:
    """For each tensor and each split dimension, find a representative pair of cores
    that differ by exactly 1 in that dimension's slice index, and verify the byte
    step equals stride(dim) * word_length.
    """
    issues: list[Issue] = []
    wk_map = top_level["coreIdToWkSlice_"]
    num_cores = top_level["numCoresUsed_"]
    n = _n(dsc)
    ss = _ss(dsc)
    slices = top_level["numWkSlicesPerDim_"]

    for i, node in enumerate(dsc["scheduleTree_"]):
        tensor_name = f"Tensor{i}"
        labeled = dsc["labeledDs_"][i]
        word_length = labeled["wordLength"]
        scales = labeled["scale_"]
        layout_dim_order = node["layoutDimOrder_"]
        layout_label = labeled["dsType_"]
        primary_info = _primary_info(dsc, layout_label)
        stick_dims, stick_size = _stick_dims_and_size(primary_info)
        if stick_size is None:
            continue

        dim_scale = {dim: scales[j] for j, dim in enumerate(layout_dim_order)}
        strides = _compute_strides(layout_dim_order, n, stick_dims, stick_size)
        addr_data = node["startAddressCoreCorelet_"]["data_"]
        addrs = {c: int(addr_data[f"[{c}, 0, 0]"]) for c in range(num_cores)}

        for dim in layout_dim_order:
            if dim_scale.get(dim, 1) <= 0:
                continue
            if slices.get(dim, 1) <= 1:
                continue

            # Step in bytes when the slice index of dim increases by 1:
            #   For stick dims: (ss_[dim] / stick_size) sticks * stride * wordLen
            #     (ss_[dim] is in elements; stride already carries one stick_size factor)
            #   For non-stick dims: ss_[dim] * stride * wordLen
            ss_dim = ss.get(dim, 1)
            is_stick = dim in stick_dims
            if is_stick:
                num_sticks = ss_dim // stick_size
                expected_step = num_sticks * strides[dim] * word_length
            else:
                expected_step = ss_dim * strides[dim] * word_length

            # Find one representative pair of cores differing only in this dim by 1
            found = False
            for c in range(num_cores):
                ws_c = {k: int(v) for k, v in wk_map[str(c)].items()}
                for c2 in range(num_cores):
                    ws_c2 = {k: int(v) for k, v in wk_map[str(c2)].items()}
                    if ws_c2.get(dim, 0) - ws_c.get(dim, 0) != 1:
                        continue
                    if any(ws_c2.get(d) != ws_c.get(d) for d in slices if d != dim):
                        continue
                    actual_step = addrs[c2] - addrs[c]
                    found = True
                    if actual_step != expected_step:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"address step (core {c}→{c2}, "
                                f"slice {ws_c[dim]}→{ws_c2[dim]}) = {actual_step} bytes, "
                                f"expected {'(ss/stickSize)' if is_stick else 'ss'}"
                                f"*stride*wordLen = {expected_step} "
                                f"(stride({dim})={strides[dim]}, ss={ss_dim}"
                                f"{', stickSize=' + str(stick_size) if is_stick else ''})",
                            )
                        )
                    break
                if found:
                    break

    return issues


# ---------------------------------------------------------------------------
# Check 4: Coordinate fold factors
# ---------------------------------------------------------------------------


def _fold_factors(folds: dict) -> list[int]:
    return [entry["factor_"] for entry in folds["dim_prop_attr"]]


def _fold_alphas(folds: dict) -> list[int]:
    return [
        list(entry.values())[0].get("alpha_", 0) for entry in folds["dim_prop_func"]
    ]


def check_coord_folds(dsc: dict, top_level: dict) -> list[Issue]:
    """Fold factor products must equal N_[dim] for all coordinate dims.

    For non-stick dims: [nsplits, 1, 1, per_core_size] where
      nsplits * per_core_size = N_[dim].

    For stick dims (not reduction): [nsplits, 1, 1, num_sticks, stick_size] where
      nsplits * num_sticks * stick_size = N_[dim],
      alpha_core_fold = per_core_size = N_[dim] // nsplits.

    For stick dims (reduction, scale==-2): nsplits forced to 1,
      alpha_core_fold = stick_size, last alpha = 0.
    """
    issues: list[Issue] = []
    n = _n(dsc)
    slices = top_level["numWkSlicesPerDim_"]

    for i, node in enumerate(dsc["scheduleTree_"]):
        tensor_name = f"Tensor{i}"
        labeled = dsc["labeledDs_"][i]
        scales = labeled["scale_"]
        layout_dim_order = node["layoutDimOrder_"]
        layout_label = labeled["dsType_"]
        primary_info = _primary_info(dsc, layout_label)
        stick_dims, stick_size = _stick_dims_and_size(primary_info)

        dim_scale = {dim: scales[j] for j, dim in enumerate(layout_dim_order)}
        coord_info = node["coordinates_"]["coordInfo"]

        for dim, info in coord_info.items():
            elem_arr = info["elemArr"]
            folds = info["folds"]
            factors = _fold_factors(folds)
            alphas = _fold_alphas(folds)
            product = math.prod(factors)
            scale = dim_scale.get(dim, 1)

            # Skip reduced/broadcast dims (scale != 1): their fold structure is
            # implementation-defined and not covered by our invariants.
            if scale != 1:
                continue

            # Fold products always span N_[dim] (the full problem size for this dim).
            expected_product = n.get(dim, 1)

            if product != expected_product:
                issues.append(
                    Issue(
                        "ERROR",
                        f"{tensor_name}/{dim}",
                        f"fold factor product={product} but expected N_={expected_product} "
                        f"(factors={factors})",
                    )
                )

            nsplits = slices.get(dim, 1) if scale == 1 else 1
            per_core = expected_product // nsplits if nsplits else expected_product

            if dim in (stick_dims or set()) and stick_size is not None:
                if elem_arr != 2:
                    issues.append(
                        Issue(
                            "WARN",
                            f"{tensor_name}/{dim}",
                            f"stick dim should have elemArr=2, got {elem_arr}",
                        )
                    )
                is_reduction = scale == -2
                if is_reduction:
                    if factors[0] != 1:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"stick-reduction core_fold factor={factors[0]}, expected 1",
                            )
                        )
                    if alphas[0] != stick_size:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"stick-reduction core_fold alpha={alphas[0]}, "
                                f"expected stick_size={stick_size}",
                            )
                        )
                    if alphas[-1] != 0:
                        issues.append(
                            Issue(
                                "WARN",
                                f"{tensor_name}/{dim}",
                                f"stick-reduction last alpha={alphas[-1]}, expected 0",
                            )
                        )
                else:
                    if factors[0] != nsplits:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"core_fold factor={factors[0]}, expected nsplits={nsplits}",
                            )
                        )
                    if alphas[0] != per_core:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"core_fold alpha={alphas[0]}, "
                                f"expected per_core={per_core} "
                                f"(N_={expected_product} // nsplits={nsplits})",
                            )
                        )
                    if factors[-1] != stick_size:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"innermost fold factor={factors[-1]}, "
                                f"expected stick_size={stick_size}",
                            )
                        )
                    expected_num_sticks = per_core // stick_size
                    if factors[-2] != expected_num_sticks:
                        issues.append(
                            Issue(
                                "ERROR",
                                f"{tensor_name}/{dim}",
                                f"num_sticks fold factor={factors[-2]}, "
                                f"expected {expected_num_sticks} "
                                f"(per_core={per_core} // stick_size={stick_size})",
                            )
                        )
            else:
                if elem_arr != 1:
                    issues.append(
                        Issue(
                            "WARN",
                            f"{tensor_name}/{dim}",
                            f"non-stick dim should have elemArr=1, got {elem_arr}",
                        )
                    )
                if factors[0] != nsplits:
                    issues.append(
                        Issue(
                            "ERROR",
                            f"{tensor_name}/{dim}",
                            f"core_fold factor={factors[0]}, expected nsplits={nsplits}",
                        )
                    )
                if factors[-1] != per_core:
                    issues.append(
                        Issue(
                            "ERROR",
                            f"{tensor_name}/{dim}",
                            f"elem_arr_0 factor={factors[-1]}, "
                            f"expected per_core={per_core} "
                            f"(N_={expected_product} // nsplits={nsplits})",
                        )
                    )

    return issues


# ---------------------------------------------------------------------------
# Top-level runner + report
# ---------------------------------------------------------------------------


def diagnose_sdsc(path: str) -> list[Issue]:
    op_name, top_level, dsc = load_sdsc(path)

    all_issues: list[Issue] = []
    all_issues.extend(check_n_ss_slices(dsc, top_level))
    all_issues.extend(check_wk_slice_mapping(top_level))
    all_issues.extend(check_start_addresses(dsc, top_level))
    all_issues.extend(check_coord_folds(dsc, top_level))

    _print_report(op_name, path, top_level, dsc, all_issues)
    return all_issues


def _print_report(
    op_name: str, path: str, top_level: dict, dsc: dict, issues: list[Issue]
) -> None:
    n = _n(dsc)
    ss = _ss(dsc)
    slices = top_level["numWkSlicesPerDim_"]

    print(f"\n{'=' * 60}")
    print(f"SDSC Diagnostic: {path}")
    print(f"  op: {op_name}  cores: {top_level['numCoresUsed_']}")
    print(f"  N_:               {n}")
    print(f"  ss_:              {ss}")
    print(f"  numWkSlicesPerDim_: {slices}")
    print(f"{'=' * 60}")

    checks: list[tuple[str, list[Issue]]] = [
        (
            "N_/ss_/slices consistency",
            [
                i
                for i in issues
                if not i.location.startswith("Tensor")
                and not i.location.startswith("core ")
            ],
        ),
        (
            "coreIdToWkSlice_ mapping",
            [i for i in issues if i.location.startswith("core ")],
        ),
        (
            "startAddressCoreCorelet_ offsets",
            [
                i
                for i in issues
                if i.location.startswith("Tensor") and "address step" in i.message
            ],
        ),
        (
            "Coordinate fold factors",
            [
                i
                for i in issues
                if i.location.startswith("Tensor") and "address step" not in i.message
            ],
        ),
    ]

    printed: set[int] = set()
    for check_name, check_issues in checks:
        new = [i for i in check_issues if id(i) not in printed]
        status = "PASS" if not new else f"FAIL ({len(new)} issue(s))"
        print(f"\n[{status}] {check_name}")
        for issue in new:
            print(f"  {issue.severity:5s}  {issue.location}: {issue.message}")
            printed.add(id(issue))

    remaining = [i for i in issues if id(i) not in printed]
    if remaining:
        print("\n[Other issues]")
        for issue in remaining:
            print(f"  {issue.severity:5s}  {issue.location}: {issue.message}")

    total_errors = sum(1 for i in issues if i.severity == "ERROR")
    total_warns = sum(1 for i in issues if i.severity == "WARN")
    print(f"\n{'=' * 60}")
    print(f"Summary: {total_errors} error(s), {total_warns} warning(s)")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an SDSC JSON file.")
    parser.add_argument("sdsc_json", help="Path to the SDSC JSON file")
    args = parser.parse_args()
    issues = diagnose_sdsc(args.sdsc_json)
    raise SystemExit(0 if not any(i.severity == "ERROR" for i in issues) else 1)
