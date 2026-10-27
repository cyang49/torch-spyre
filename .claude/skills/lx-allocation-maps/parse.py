# Copyright 2026 The Torch-Spyre Authors.
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

"""Validate CP-SAT allocation JSONL and write machine-readable summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCHEMA = "spyre-lx-allocation-graph-trace-v2"


def _load_records(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
        if record.get("schema") != SCHEMA:
            raise SystemExit(
                f"{path}:{line_number}: expected {SCHEMA!r}, "
                f"got {record.get('schema')!r}"
            )
        records.append(record)
    if not records:
        raise SystemExit(f"No allocation records in {path}")
    return records


def _validate_record(record: dict) -> list[str]:
    problems = []
    operations = record.get("operations") or []
    buffers = record.get("buffers") or []
    resident = [buffer for buffer in buffers if buffer.get("address_bytes") is not None]
    if record.get("solver") != "CpSatLayoutSolver":
        problems.append(f"solver={record.get('solver')!r}")
    if record.get("operation_count") != len(operations):
        problems.append("operation_count does not match operations array")
    if record.get("buffer_count") != len(buffers):
        problems.append("buffer_count does not match buffers array")
    if record.get("resident_count") != len(resident):
        problems.append("resident_count does not match addressed buffers")

    capacity = record.get("capacity_bytes")
    watermark = max(
        (
            buffer["address_bytes"] + buffer["reserved_size_bytes"]
            for buffer in resident
        ),
        default=0,
    )
    if watermark != record.get("global_watermark_bytes"):
        problems.append(
            "global_watermark_bytes does not match resident address extents"
        )
    if record.get("timeline_matches_global_watermark") is not True:
        problems.append("timeline_matches_global_watermark is false")
    if record.get("all_resident_spans_within_capacity") is not True:
        problems.append("all_resident_spans_within_capacity is false")
    for buffer in resident:
        start = buffer["address_bytes"]
        size = buffer["reserved_size_bytes"]
        if start < 0 or size < 0 or start + size > capacity:
            problems.append(f"resident buffer {buffer['name']} exceeds capacity")
            break
    timeline = record.get("timeline") or []
    if len(timeline) != len(operations):
        problems.append("timeline length does not match operations")
    for index, tick in enumerate(timeline):
        if tick.get("op_index") != index:
            problems.append("timeline indices are not contiguous from zero")
            break
        active = [
            buffer
            for buffer in resident
            if buffer["start_time"] <= index < buffer["end_time"]
        ]
        tick_watermark = max(
            (
                buffer["address_bytes"] + buffer["reserved_size_bytes"]
                for buffer in active
            ),
            default=0,
        )
        if tick.get("watermark_bytes") != tick_watermark:
            problems.append(f"operation {index} timeline watermark mismatch")
            break
        if tick.get("active_resident_buffers") != [buffer["name"] for buffer in active]:
            problems.append(f"operation {index} active resident list mismatch")
            break
    if (
        max((tick.get("watermark_bytes", 0) for tick in timeline), default=0)
        != watermark
    ):
        problems.append(
            "per-operation watermark timeline does not match global watermark"
        )
    return problems


def _summary(record: dict, index: int) -> dict:
    operations = record["operations"]
    buffers = record["buffers"]
    resident = [buffer for buffer in buffers if buffer.get("address_bytes") is not None]
    source_ops = sum(
        any(
            (origin.get("source") or {}).get("file") for origin in op.get("origins", [])
        )
        for op in operations
    )
    fx_ops = sum(bool(op.get("origins")) for op in operations)
    watermark = record["global_watermark_bytes"]
    capacity = record["capacity_bytes"]
    return {
        "capture_record_index": index,
        "pid": record.get("pid"),
        "aot_graph_name": record.get("aot_graph_name"),
        "aot_graph_path": record.get("aot_graph_path"),
        "inductor_graph_id": record.get("inductor_graph_id"),
        "inductor_post_grad_graph_id": record.get("inductor_post_grad_graph_id"),
        "allocator": record.get("allocator"),
        "solver": record.get("solver"),
        "operation_count": len(operations),
        "operations_with_fx_origins": fx_ops,
        "operations_with_source_file": source_ops,
        "buffer_count": len(buffers),
        "resident_count": len(resident),
        "spilled_count": len(buffers) - len(resident),
        "global_watermark_bytes": watermark,
        "capacity_bytes": capacity,
        "watermark_fraction": watermark / capacity if capacity else None,
        "timeline_matches_global_watermark": record.get(
            "timeline_matches_global_watermark"
        ),
        "all_resident_spans_within_capacity": record.get(
            "all_resident_spans_within_capacity"
        ),
        "validation_errors": _validate_record(record),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="allocations.<pid>.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = _load_records(args.capture)
    summaries = [_summary(record, index) for index, record in enumerate(records)]
    failures = [
        (summary["capture_record_index"], summary["validation_errors"])
        for summary in summaries
        if summary["validation_errors"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "spyre-lx-allocation-summary-v1",
                "capture": str(args.capture.resolve()),
                "record_count": len(records),
                "valid": not failures,
                "records": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    columns = list(summaries[0])
    with (args.output_dir / "graphs.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["validation_errors"] = "; ".join(summary["validation_errors"])
            writer.writerow(row)

    print(f"Records: {len(records)}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Graph table: {args.output_dir / 'graphs.tsv'}")
    if failures:
        for index, errors in failures:
            print(f"INVALID record {index}: {'; '.join(errors)}")
        raise SystemExit(1)
    print("All records pass CP-SAT, watermark, and capacity checks.")
    for summary in summaries:
        print(
            f"{summary['capture_record_index']:>3} "
            f"{summary['aot_graph_name']} "
            f"watermark={summary['global_watermark_bytes']:,}/"
            f"{summary['capacity_bytes']:,} B "
            f"ops={summary['operation_count']} resident={summary['resident_count']}"
        )


if __name__ == "__main__":
    main()
