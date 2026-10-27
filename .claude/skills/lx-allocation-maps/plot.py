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

"""Render one LX address/lifetime map and operation ledger per captured graph."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

SCHEMA = "spyre-lx-allocation-graph-trace-v2"
PALETTE = (
    "#397db8",
    "#e58a32",
    "#45a78a",
    "#8b65b7",
    "#bd5b83",
    "#987b38",
    "#168c9a",
    "#c56f33",
)


def _render_one(record: dict, index: int, output_dir: Path, dpi: int) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    operations = record["operations"]
    timeline = record["timeline"]
    resident = [
        buffer
        for buffer in record["buffers"]
        if buffer.get("address_bytes") is not None
    ]
    count = len(operations)
    capacity = record["capacity_bytes"]
    watermark = record["global_watermark_bytes"]
    peak_ticks = {
        tick["op_index"]
        for tick in timeline
        if tick["watermark_bytes"] == watermark and watermark
    }
    peak_buffers = {
        name
        for tick in timeline
        if tick["op_index"] in peak_ticks
        for name in tick.get("buffers_at_watermark", [])
    }

    figure, (allocation_ax, op_ax) = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (4, 0.7)},
        layout="constrained",
    )
    figure.suptitle(
        f"{record.get('aot_graph_name') or 'unnamed graph'} — CP-SAT LX allocation\n"
        f"{count} operations · {record['resident_count']} resident buffers · "
        f"watermark {watermark:,} B / capacity {capacity:,} B "
        f"({watermark / max(capacity, 1):.1%}) · "
        f"Inductor graph ID {record.get('inductor_graph_id')}"
    )
    for buffer in resident:
        start, end = buffer["start_time"], buffer["end_time"]
        address, extent = buffer["address_bytes"], buffer["reserved_size_bytes"]
        is_peak = buffer["name"] in peak_buffers
        allocation_ax.add_patch(
            Rectangle(
                (start, address),
                end - start,
                extent,
                facecolor=PALETTE[1] if is_peak else PALETTE[0],
                edgecolor="#9c4c17" if is_peak else "#24577f",
                alpha=0.72 if is_peak else 0.38,
                linewidth=0.7,
            )
        )
        if end - start >= 1.5 and extent >= 16384:
            allocation_ax.text(
                (start + end) / 2,
                address + extent / 2,
                buffer["name"],
                ha="center",
                va="center",
                fontsize=5.5,
                clip_on=True,
            )
    allocation_ax.step(
        [tick["op_index"] for tick in timeline] + [count],
        [tick["watermark_bytes"] for tick in timeline]
        + ([timeline[-1]["watermark_bytes"]] if timeline else [0]),
        where="post",
        color="#c72d32",
        linewidth=1.8,
        label="Per-operation watermark",
        zorder=5,
    )
    allocation_ax.axhline(
        capacity, color="#666", linestyle="--", linewidth=1, label="Capacity"
    )
    allocation_ax.set(
        xlim=(0, max(count, 1)),
        ylim=(0, max(capacity, 1)),
        ylabel="LX address (KiB)",
        title="Reserved address extents by operation; half-open lifetimes [start, end)",
    )
    allocation_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1024:.0f}")
    )
    allocation_ax.set_xticks(range(0, count + 1, max(1, count // 16 or 1)))
    allocation_ax.set_xticks(range(count + 1), minor=True)
    allocation_ax.set_yticks(range(0, capacity + 1, 32768), minor=True)
    allocation_ax.grid(True, axis="both", alpha=0.2)
    allocation_ax.grid(True, which="minor", axis="both", alpha=0.06)
    allocation_ax.legend(loc="upper left", fontsize=8)
    allocation_ax.text(
        max(0.5, count - 0.5),
        watermark,
        f"{watermark / 1024:.0f} KiB",
        color="#c72d32",
        va="bottom",
        ha="right",
        fontsize=8,
    )

    op_types = list(
        dict.fromkeys(str(op.get("op_type") or "unknown") for op in operations)
    )
    type_colors = {
        name: PALETTE[position % len(PALETTE)] for position, name in enumerate(op_types)
    }
    for operation in operations:
        op_type = str(operation.get("op_type") or "unknown")
        x = operation["op_index"]
        op_ax.add_patch(
            Rectangle(
                (x, 0),
                1,
                1,
                facecolor=type_colors[op_type],
                edgecolor="white",
                linewidth=0.4,
            )
        )
        if count <= 160:
            op_ax.text(
                x + 0.5,
                0.5,
                str(x),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white",
            )
    op_ax.set(xlim=(0, max(count, 1)), ylim=(0, 1), title="Operation type by tick")
    op_ax.set_xticks(range(0, count + 1, max(1, count // 16 or 1)))
    op_ax.set_yticks([])
    op_ax.grid(True, axis="x", alpha=0.12)
    op_ax.set_xlabel("Graph operation index")

    stem = f"{record.get('aot_graph_name') or f'graph-{index}'}-solve-{index}"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(png_path, dpi=dpi)
    plt.close(figure)

    ledger_path = output_dir / f"{stem}-ledger.html"
    tsv_path = output_dir / f"{stem}.tsv"
    rows = []
    columns = [
        "op_index",
        "op_type",
        "operation_name",
        "buffer_name",
        "watermark_bytes",
        "buffers_at_watermark",
        "active_resident_buffers",
        "origins",
    ]
    for operation in operations:
        tick = timeline[operation["op_index"]]
        origin_text = []
        for origin in operation.get("origins", []):
            source = origin.get("source") or {}
            if source.get("file"):
                origin_text.append(
                    f"{source['file']}:{source.get('line', '?')}"
                    + (f" — {source['code']}" if source.get("code") else "")
                )
            elif origin.get("target"):
                origin_text.append(str(origin["target"]))
        row = {
            "op_index": operation["op_index"],
            "op_type": operation.get("op_type"),
            "operation_name": operation.get("operation_name"),
            "buffer_name": operation.get("buffer_name"),
            "watermark_bytes": tick["watermark_bytes"],
            "buffers_at_watermark": ", ".join(tick.get("buffers_at_watermark", [])),
            "active_resident_buffers": ", ".join(
                tick.get("active_resident_buffers", [])
            ),
            "origins": "\n".join(dict.fromkeys(origin_text)),
        }
        rows.append(row)
    with tsv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    html_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[column] or ''))}</td>" for column in columns
        )
        + "</tr>"
        for row in rows
    )
    ledger_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(stem)}</title>"
        "<style>body{font:14px/1.4 system-ui,sans-serif;margin:1rem}"
        "table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #aaa;padding:.3rem;vertical-align:top;text-align:left}"
        "thead{position:sticky;top:0;background:#eee}td{white-space:pre-wrap}</style>"
        f"<h1>{html.escape(stem)}</h1><p>CP-SAT · {len(operations)} operations · "
        f"{record['resident_count']} resident buffers · watermark {watermark:,} B / "
        f"{capacity:,} B · lifetime [start, end)</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        + f"</tr></thead><tbody>{html_rows}</tbody></table></html>\n",
        encoding="utf-8",
    )
    return {
        "capture_record_index": index,
        "aot_graph_name": record.get("aot_graph_name"),
        "inductor_graph_id": record.get("inductor_graph_id"),
        "solver": record.get("solver"),
        "operation_count": len(operations),
        "resident_count": record.get("resident_count"),
        "global_watermark_bytes": watermark,
        "capacity_bytes": capacity,
        "watermark_fraction": watermark / capacity if capacity else None,
        "png": png_path.name,
        "html_ledger": ledger_path.name,
        "operation_ledger": tsv_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="allocations.<pid>.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--record",
        type=int,
        action="append",
        dest="record_indices",
        help="Capture record index to render; repeat for several. Default: all records.",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    records = [
        json.loads(line)
        for line in args.capture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"No allocation records in {args.capture}")
    selected = (
        list(range(len(records)))
        if args.record_indices is None
        else list(dict.fromkeys(args.record_indices))
    )
    if not selected:
        parser.error("at least one --record is required")
    if any(index < 0 or index >= len(records) for index in selected):
        parser.error(f"--record values must be in [0, {len(records) - 1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index in selected:
        record = records[index]
        if record.get("schema") != SCHEMA:
            raise SystemExit(f"Record {index} has unexpected schema")
        if record.get("solver") != "CpSatLayoutSolver":
            raise SystemExit(f"Record {index} is not CP-SAT")
        buffers = record.get("buffers") or []
        resident = [
            buffer for buffer in buffers if buffer.get("address_bytes") is not None
        ]
        capacity = record.get("capacity_bytes")
        if not isinstance(capacity, int) or capacity < 0:
            raise SystemExit(f"Record {index} has invalid capacity")
        for buffer in resident:
            address = buffer["address_bytes"]
            size = buffer["reserved_size_bytes"]
            if address < 0 or size < 0 or address + size > capacity:
                raise SystemExit(
                    f"Record {index} buffer {buffer['name']} exceeds capacity"
                )
        watermark = max(
            (
                buffer["address_bytes"] + buffer["reserved_size_bytes"]
                for buffer in resident
            ),
            default=0,
        )
        operations = record.get("operations") or []
        timeline = record.get("timeline") or []
        if record.get("operation_count") != len(operations):
            raise SystemExit(f"Record {index} operation count mismatch")
        if len(timeline) != len(operations):
            raise SystemExit(f"Record {index} timeline length mismatch")
        if any(tick.get("op_index") != i for i, tick in enumerate(timeline)):
            raise SystemExit(f"Record {index} timeline indices are not contiguous")
        for tick in timeline:
            tick_index = tick["op_index"]
            active = [
                buffer
                for buffer in resident
                if buffer["start_time"] <= tick_index < buffer["end_time"]
            ]
            tick_watermark = max(
                (
                    buffer["address_bytes"] + buffer["reserved_size_bytes"]
                    for buffer in active
                ),
                default=0,
            )
            if tick.get("watermark_bytes") != tick_watermark:
                raise SystemExit(
                    f"Record {index} operation {tick_index} watermark mismatch"
                )
            if tick.get("active_resident_buffers") != [
                buffer["name"] for buffer in active
            ]:
                raise SystemExit(
                    f"Record {index} operation {tick_index} active buffer mismatch"
                )
        if (
            max((tick.get("watermark_bytes", 0) for tick in timeline), default=0)
            != watermark
        ):
            raise SystemExit(f"Record {index} timeline watermark mismatch")
        if watermark != record.get("global_watermark_bytes"):
            raise SystemExit(f"Record {index} global watermark mismatch")
        summary = _render_one(record, index, args.output_dir, args.dpi)
        summary["timeline_matches_global_watermark"] = True
        summary["all_resident_spans_within_capacity"] = True
        manifest.append(summary)
        summary_path = args.output_dir / f"{summary['png'][:-4]}.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True), flush=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(manifest)} allocation maps into {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
