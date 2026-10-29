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

"""Render PNG/HTML LX maps and operation ledgers per captured graph."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
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


def _static_preview(record: dict) -> tuple[str, str, str]:
    """Keep the maps and table visible in HTML previews that disable JavaScript."""
    buffers = record["buffers"]
    resident = [b for b in buffers if b["address_bytes"] is not None]
    rejected = [
        b
        for b in buffers
        if b["address_bytes"] is None
        and (
            b.get("is_synthetic_relayout", False)
            or b["name"].startswith("__spyre_lx_relayout__")
        )
    ]
    no_consumers = [
        b
        for b in buffers
        if b["address_bytes"] is None
        and not b.get("is_synthetic_relayout", False)
        and not b["name"].startswith("__spyre_lx_relayout__")
        and isinstance(b.get("consumer_op_indices"), list)
        and not b["consumer_op_indices"]
    ]
    nonresident = [
        b
        for b in buffers
        if b["address_bytes"] is None and b not in rejected and b not in no_consumers
    ]
    ticks = record["timeline"]
    max_tick = max(
        [
            1,
            *(b["end_time"] for b in buffers),
            *(t.get("allocation_tick", t["op_index"]) + 1 for t in ticks),
        ]
    )
    left, span = 75, 1100

    def x(tick: int) -> float:
        return left + tick * span / max_tick

    marks = sorted(
        {0, max_tick, *range(0, max_tick + 1, max(1, math.ceil(max_tick / 12)))}
    )
    lx_axis = "".join(
        f'<line x1="{x(t):.2f}" x2="{x(t):.2f}" y1="506" y2="512" stroke="#64758a"/>'
        f'<text x="{x(t):.2f}" y="527" text-anchor="middle" class="axis-label">{t}</text>'
        for t in marks
    )
    nonresident_axis = "".join(
        f'<line x1="{x(t):.2f}" x2="{x(t):.2f}" y1="259" y2="265" stroke="#64758a"/>'
        f'<text x="{x(t):.2f}" y="281" text-anchor="middle" class="axis-label">{t}</text>'
        for t in marks
    )
    capacity = max(record["capacity_bytes"], 1)
    resident_shapes = "".join(
        f'<rect class="buffer{" relayout" if b.get("is_synthetic_relayout") else ""}'
        f'{" peak" if b["address_bytes"] + b["reserved_size_bytes"] == record["global_watermark_bytes"] and record["global_watermark_bytes"] else ""}" '
        f'x="{x(b["start_time"]):.2f}" '
        f'y="{30 + 430 * (1 - (b["address_bytes"] + b["reserved_size_bytes"]) / capacity):.2f}" '
        f'width="{max(1, x(b["end_time"]) - x(b["start_time"])):.2f}" '
        f'height="{max(1, b["reserved_size_bytes"] * 430 / capacity):.2f}">'
        f"<title>{html.escape(b['name'])}: "
        f"LX {b['address_bytes']:,} B, {b['reserved_size_bytes']:,} B per core</title></rect>"
        for b in resident
    )
    lx_svg = (
        '<line x1="75" y1="500" x2="1175" y2="500" class="axis"/>'
        f"{resident_shapes}{lx_axis}"
        '<text x="75" y="549" class="axis-label">Allocator tick</text>'
    )
    sizes = [
        b.get("allocated_size_bytes", b.get("logical_size_bytes")) for b in nonresident
    ]
    positive = [n for n in sizes if isinstance(n, (int, float)) and n > 0]
    low = math.floor(math.log2(min(positive))) if positive else 0
    high = math.ceil(math.log2(max(positive))) if positive else 0
    spread = max(high - low, 1)
    y_labels = "".join(
        f'<line x1="75" x2="1175" y1="{22 + (high - power) * 210 / spread:.2f}" '
        f'y2="{22 + (high - power) * 210 / spread:.2f}" class="grid"/>'
        f'<text x="68" y="{26 + (high - power) * 210 / spread:.2f}" '
        f'text-anchor="end" class="axis-label">2^{power} B</text>'
        for power in range(low, high + 1)
        if positive
        and ((power - low) % max(1, math.ceil(spread / 6)) == 0 or power == high)
    )
    points = []
    for b in nonresident:
        size = b.get("allocated_size_bytes", b.get("logical_size_bytes"))
        valid = isinstance(size, (int, float)) and size > 0
        y = 22 + (high - math.log2(size)) * 210 / spread if valid else 259
        state = " relayout" if b.get("is_synthetic_relayout") else ""
        uses = sorted(set(b.get("uses") or []))
        markers = "".join(
            f'<circle cx="{x(tick):.2f}" cy="{y:.2f}" r="3" '
            f'class="nonresident-use{state}" aria-hidden="true"/>'
            for tick in uses
        )
        points.append(
            f'<line x1="{x(b["start_time"]):.2f}" x2="{x(b["end_time"]):.2f}" '
            f'y1="{y:.2f}" y2="{y:.2f}" class="nonresident-interval{state}"/>'
            f"{markers}"
            f'<circle cx="{x(b["start_time"]):.2f}" cy="{y:.2f}" r="4" '
            f'class="nonresident-point{" unsized" if not valid else ""}{state}">'
            f"<title>{html.escape(b['name'])}: solver footprint "
            f"{html.escape(str(size))} B; use ticks {html.escape(', '.join(map(str, uses)))}; "
            f"{html.escape(str(b.get('spill_reason') or b.get('residency_reason') or 'unknown'))}"
            "</title></circle>"
        )
    nonresident_svg = (
        '<line x1="75" y1="22" x2="75" y2="259" class="axis"/>'
        '<line x1="75" y1="259" x2="1175" y2="259" class="axis"/>'
        '<text x="68" y="263" text-anchor="end" class="axis-label">unsized</text>'
        + y_labels
        + "".join(points)
        + nonresident_axis
        + '<text x="75" y="302" class="axis-label">Allocator tick</text>'
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(b['name'])}</td>"
        f"<td>{html.escape(str(b.get('spill_reason') or b.get('residency_reason') or 'unknown'))}</td>"
        f"<td>{html.escape(str(b.get('host_shape')))} / {html.escape(str(b.get('host_dtype')))}</td>"
        f"<td>{html.escape(str(b.get('spyre_tensor_layout')))}</td>"
        f"<td>{html.escape(str(b.get('allocated_size_bytes', b.get('logical_size_bytes'))))} B solver footprint</td>"
        f"<td>{html.escape(', '.join(f'op {i}' for i in b.get('consumer_op_indices') or []))}</td>"
        "</tr>"
        for b in sorted(
            nonresident,
            key=lambda b: (
                ", ".join(f"op {i}" for i in b.get("consumer_op_indices") or []),
                b["name"],
            ),
        )
    )
    return lx_svg, nonresident_svg, rows


def _render_interactive_html(record: dict, path: Path) -> None:
    # JSON is embedded in a script element, not HTML attributes or executable JS.
    # Escape script terminators and JS line separators without changing parsed values.
    payload = (
        json.dumps(record, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive LX allocation map</title>
<style>
:root{font:14px/1.45 system-ui,sans-serif;color:#172739;background:#f7f9fc}
body{margin:1rem auto;max-width:1700px;padding:0 1rem}
h1,h2{line-height:1.25}h1{margin-bottom:.25rem}h2{font-size:1.1rem}
.muted{color:#536275}.controls{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:1rem 0}
.controls label{display:flex;gap:.4rem;align-items:center}input,select,button{font:inherit}
.controls output{min-width:3.5rem;text-align:right;font-variant-numeric:tabular-nums}
input[type=search]{min-width:20rem;max-width:100%}
.panel{background:white;border:1px solid #c8d3e0;border-radius:6px;padding:1rem;margin:1rem 0}
.split{display:grid;grid-template-columns:minmax(0,3fr) minmax(19rem,1fr);gap:1rem}
.split.details-closed{grid-template-columns:minmax(0,1fr) auto}
@media(max-width:850px){.split{grid-template-columns:1fr}}
#details-panel summary{cursor:pointer;font-weight:600}
.map-scroll,.table-scroll{overflow:auto;max-width:100%}
.table-scroll{max-height:32rem;overscroll-behavior:contain}
.chart-stack{overflow-x:auto;overflow-y:hidden;max-width:100%;overscroll-behavior-x:contain;scrollbar-width:auto}
.chart-stack::-webkit-scrollbar:vertical{display:none}
.chart-stack .map-scroll{overflow:visible;width:max-content}
.chart-stack h2,.chart-stack p,.chart-stack .legend{position:sticky;left:0;max-width:100%}
svg{display:block;background:white;max-width:none}
svg .grid{stroke:#dfe6ee;stroke-width:1}svg .axis{stroke:#64758a;stroke-width:1}
svg .axis-label{font:12px system-ui,sans-serif;fill:#364a5d}
svg .buffer{fill:#397db8;fill-opacity:.48;stroke:#24577f;stroke-width:1;cursor:pointer}
svg .buffer.peak{fill:#e58a32;stroke:#9c4c17}
svg .buffer.relayout{fill:#168c9a;stroke:#075c66;fill-opacity:.72}
svg .buffer.relayout.peak{stroke:#9c4c17;stroke-width:2.5}
svg .buffer:hover,svg .buffer:focus,svg .buffer.selected,svg .buffer.related{fill-opacity:.85;stroke:#121f32;stroke-width:2}
svg .watermark{fill:none;stroke:#c72d32;stroke-width:2;pointer-events:none}
svg .capacity{stroke:#666;stroke-dasharray:6 4;pointer-events:none}
.legend{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:.5rem 0}
.legend span{display:inline-flex;gap:.4rem;align-items:center}
.swatch{display:inline-block;width:1rem;height:.7rem;border:1px solid #24577f;background:#397db8;opacity:.7}
.swatch.peak{background:#e58a32;border-color:#9c4c17;opacity:1}
.swatch.relayout{background:#168c9a;border-color:#075c66;opacity:1}
.swatch.relayout-peak{background:#168c9a;border:2px solid #9c4c17;opacity:1}
.swatch.watermark{height:0;border:0;border-top:2px solid #c72d32;background:none;opacity:1}
.swatch.capacity{height:0;border:0;border-top:2px dashed #666;background:none;opacity:1}
.swatch.nonresident{background:#8162a6;border-color:#46305f;opacity:1}
.swatch.unsized{background:white;border:2px dashed #46305f;opacity:1}
.swatch.use{width:.45rem;height:.45rem;border-radius:50%;background:#172739;border-color:white;box-shadow:0 0 0 1px #172739;opacity:1}
svg .op-tick{stroke:#fff;stroke-width:.6;cursor:pointer}
svg .op-index{font:10px system-ui,sans-serif;fill:#fff;pointer-events:none}
svg .nonresident-interval{stroke:#8162a6;stroke-width:2;stroke-opacity:.55;pointer-events:none}
svg .nonresident-interval.relayout{stroke:#168c9a}
svg .nonresident-use{fill:#172739;stroke:white;stroke-width:.8;cursor:pointer}
svg .nonresident-use.relayout{fill:#075c66}
svg .nonresident-use:hover,svg .nonresident-use:focus,svg .nonresident-use.selected,svg .nonresident-use.related{stroke:#e58a32;stroke-width:2.5}
svg .nonresident-point{fill:#fff;stroke:#46305f;stroke-width:1.5;cursor:pointer}
svg .nonresident-point.relayout{fill:#168c9a;stroke:#075c66}
svg .nonresident-point.unsized{fill:#fff;stroke-dasharray:2 1}
svg .nonresident-point:hover,svg .nonresident-point:focus,svg .nonresident-point.selected,svg .nonresident-point.related{stroke:#121f32;stroke-width:2.5}
svg .op-tick.associated,svg .op-tick.selected{stroke:#151d32;stroke-width:3}
svg [tabindex]:focus-visible{outline:2px solid #172739;outline-offset:1px}
 .chart-stack:focus-visible{outline:2px solid #14548e;outline-offset:2px}
.details{overflow-wrap:anywhere;max-height:38rem;overflow:auto}
.field{margin:.35rem 0}.field strong{display:inline-block;margin-right:.4rem}
.details button,td button{border:0;background:transparent;color:#14548e;text-decoration:underline;cursor:pointer;padding:0 .2rem}
.details button:focus-visible,td button:focus-visible{outline:2px solid #14548e}
#tooltip{position:fixed;right:1rem;bottom:1rem;z-index:5;max-width:min(32rem,90vw);max-height:55vh;overflow:auto;background:#142b44;color:white;box-shadow:0 3px 15px #555;padding:.6rem 1rem;border-radius:5px;pointer-events:none;overflow-wrap:anywhere}
[hidden]{display:none!important}table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border:1px solid #ccd6e1;padding:.4rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf2f7;position:sticky;top:0}td{max-width:24rem}
tbody tr.selected{background:#fff0d8}tbody tr:hover{background:#eef5fc}
tbody tr.related{background:#e4effa}tbody tr[tabindex]:focus-visible{outline:2px solid #14548e;outline-offset:-2px;cursor:pointer}
</style></head><body>
<h1 id="title">LX allocation map</h1><p id="summary" class="muted"></p>
<p class="muted">Planner allocation, not measured HBM traffic. Address rectangles represent
LX-resident buffers only; lifetimes are half-open [start, end). Select a rectangle,
operation tick, or table entry for details. Tab to focus rectangles and ticks.</p>
<div class="controls"><label><input type="checkbox" id="show-synthetic" checked>
Show selected LX relayout copies <span id="synthetic-count"></span></label></div>
<p id="rejected-count" class="muted">__REJECTED_COUNT__ rejected relayout candidates omitted from buffer plots. <span id="no-consumer-count">__NO_CONSUMER_COUNT__ known no-consumer graph outputs omitted from nonresident plot/table (raw entries retained).</span></p>
<div class="split"><section class="panel"><h2>Resident LX addresses by allocator tick</h2>
<div class="controls" aria-label="Horizontal scale">
<button type="button" id="zoom-fit">Fit full graph</button>
<button type="button" id="zoom-out" aria-label="Zoom out horizontally">−</button>
<label>Horizontal zoom <input type="range" id="zoom" min="0" max="8" step="1" value="0"></label>
<button type="button" id="zoom-in" aria-label="Zoom in horizontally">+</button>
<output id="zoom-label" for="zoom">1×</output></div>
<div class="legend" aria-label="Allocation map legend">
<span><i class="swatch" aria-hidden="true"></i>Other LX-resident buffer</span>
<span><i class="swatch relayout" aria-hidden="true"></i>Synthetic relayout buffer</span>
<span><i class="swatch relayout-peak" aria-hidden="true"></i>Relayout buffer at peak LX address</span>
<span><i class="swatch peak" aria-hidden="true"></i>Buffer reaching the highest LX address</span>
<span><i class="swatch watermark" aria-hidden="true"></i>Highest live LX address at each tick</span>
<span><i class="swatch capacity" aria-hidden="true"></i>Planner LX capacity</span></div>
<p class="muted">The operation strip reuses its palette by operation type; select a tick for its exact type. Synthetic relayout copies are LX planning entries, not standalone graph operations.</p>
<details><summary>Operation strip color key</summary><div id="op-legend" class="legend"></div></details>
<div class="chart-stack" id="charts" tabindex="0" role="region" aria-label="Allocation plots shown full height. Move pointer horizontally to set zoom focus; W/S zoom, A/D pan"><div class="map-scroll"><svg id="map" width="1201" height="557" viewBox="0 0 1201 557" aria-label="LX allocation address and operation timeline">__LX_PREVIEW__</svg></div>
<h2>Nonresident graph-buffer size and lifetime</h2>
<p class="muted">Each small dot marks a recorded buffer reference/use tick; the line spans the half-open lifetime [start, end), and the larger hollow dot marks lifetime start. Y shows log₂ solver footprint bytes. W/S zoom in or out around the pointer's horizontal position; A/D pan horizontally. Both plots display full height and scroll horizontally only.</p>
<div class="legend" aria-label="Nonresident plot legend"><span><i class="swatch nonresident" aria-hidden="true"></i>Positive solver footprint</span><span><i class="swatch unsized" aria-hidden="true"></i>Unknown/nonpositive solver footprint</span><span><i class="swatch use" aria-hidden="true"></i>Recorded reference/use tick</span></div>
<div class="map-scroll"><svg id="nonresident-map" width="1201" height="307" viewBox="0 0 1201 307" aria-label="Nonresident buffer solver footprint and lifetime with a marker at each recorded use tick">__NONRESIDENT_PREVIEW__</svg></div></div>
<p id="resident-count" class="muted"></p></section>
<details id="details-panel" class="panel" open><summary>Selected buffer or operation</summary>
<div id="details" class="details" aria-live="polite">Select a buffer or operation.</div></details></div>
<section class="panel"><h2>Nonresident graph buffers (no LX address)</h2>
<div class="controls"><label>Search <input id="search" type="search" placeholder="Name, shape, dtype, layout, reason…"></label>
<label>Reason <select id="reason"><option value="">All reasons</option></select></label></div>
<p id="nonresident-count" class="muted"></p><div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable nonresident buffer table"><table>
<thead><tr><th>Buffer</th><th>Reason / status</th><th>Host shape / dtype</th>
<th>Spyre device layout</th><th>Solver footprint</th><th>Operation links</th></tr></thead>
<tbody id="nonresident-rows">__ROWS_PREVIEW__</tbody></table></div></section>
<div id="tooltip" role="tooltip" hidden></div>
<noscript>Enable JavaScript to inspect this standalone capture. The PNG and operation ledger remain available.</noscript>
<script id="capture-data" type="application/json">__CAPTURE_DATA__</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("capture-data").textContent);
const capturedBuffers = data.buffers || [], operations = data.operations || [], timeline = data.timeline || [];
const rejectedRelayouts = capturedBuffers.filter(b =>
  b.address_bytes == null && (b.is_synthetic_relayout === true ||
    String(b.name || "").startsWith("__spyre_lx_relayout__")));
const noConsumerBuffers = capturedBuffers.filter(b => b.address_bytes == null &&
  !rejectedRelayouts.includes(b) && Array.isArray(b.consumer_op_indices) &&
  b.consumer_op_indices.length === 0);
const buffers = capturedBuffers.filter(b =>
  !rejectedRelayouts.includes(b) && !noConsumerBuffers.includes(b));
const resident = buffers.filter(b => b.address_bytes !== null && b.address_bytes !== undefined);
const synthetic = b => b.is_synthetic_relayout === true || String(b.name || "").startsWith("__spyre_lx_relayout__");
const nonresident = buffers.filter(b => b.address_bytes == null);
// The allocator inserts transfer ticks before operation ticks when relayouts exist.
// Old captures without a time unit only imply doubled ticks if uses exceed
// the graph's operation range; a synthetic name alone is not proof.
const doubled = data.allocation_time_unit === "half_operation" ||
  (data.allocation_time_unit == null &&
    buffers.some(b => (b.uses || []).some(t => t >= operations.length)));
const opTick = i => timeline[i]?.allocation_tick ?? (doubled ? 2 * i + 1 : i);
const tickOp = t => {
  const i = doubled ? Math.floor(t / 2) : t;
  return Number.isInteger(i) && i >= 0 && i < operations.length ? i : null;
};
const tickLabel = t => doubled && t % 2 === 0
  ? `transfer before operation ${t / 2} (tick ${t})` : `tick ${t}`;
const maxTick = Math.max(operations.length ? opTick(operations.length - 1) + 1 : 0,
  buffers.reduce((max, b) => Math.max(max, Number(b.end_time) || 0), 0), 1);
const tickStep = Math.max(1, Math.ceil(maxTick / 12));
const axisTicks = [...new Set([...Array.from(
  {length: Math.floor(maxTick / tickStep) + 1}, (_, i) => i * tickStep), maxTick])];
const $ = id => document.getElementById(id);
const zoom = $("zoom");
const mapScroll = $("charts");
const scale = () => 2 ** (Number(zoom.value) / 2);
let chartPointerX = null;
const plotWidthAtScale = () => Math.max(1, (mapScroll.clientWidth || 1001) - 101) * scale();
const svg = (tag, attrs) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
};
const format = v => v == null ? "unknown" : typeof v === "object" ? JSON.stringify(v) : String(v);
const bytes = v => v == null ? "unknown" : `${format(v)} B`;
const layout = b => b.spyre_tensor_layout || {};
const layoutText = b => {
  const l = layout(b);
  return `device_size ${format(l.device_size)} · stride_map ${format(l.stride_map)} · device_dtype ${format(l.device_dtype)} · element_arrangement ${format(l.element_arrangement)} · valid_elements ${format(l.valid_elements)}`;
};
const sizeText = b => b.allocated_size_bytes != null ? bytes(b.allocated_size_bytes)
  : b.logical_size_bytes != null ? `${bytes(b.logical_size_bytes)} (legacy solver footprint)` : "unknown";
const footprint = b => b.allocated_size_bytes ?? b.logical_size_bytes;
const reason = b => format(b.spill_reason ?? b.residency_reason);
const operationIndices = b => {
  const result = new Set();
  if (Number.isInteger(b.producer_op_index) && b.producer_op_index >= 0 && b.producer_op_index < operations.length)
    result.add(b.producer_op_index);
  if (Array.isArray(b.consumer_op_indices)) for (const i of b.consumer_op_indices)
    if (Number.isInteger(i) && i >= 0 && i < operations.length) result.add(i);
  for (const tick of b.uses || []) {
    const i = tickOp(tick);
    if (i !== null) result.add(i);
  }
  return [...result].sort((a, b) => a - b);
};
const linkText = b => operationIndices(b).map(i => `op ${i}`).join(", ");
let selectedBuffer = null, selectedOp = null;
const rects = new Map(), opRects = new Map(), rows = new Map(), points = new Map(), usePoints = new Map();
function field(parent, label, value) {
  const line = document.createElement("div");
  line.className = "field";
  const strong = document.createElement("strong");
  strong.textContent = `${label}:`;
  line.append(strong, document.createTextNode(` ${format(value)}`));
  parent.append(line);
}
function opButton(parent, i) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = `op ${i}`;
  button.addEventListener("click", () => selectOp(i));
  parent.append(button);
}
function bufferButton(parent, b) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = format(b.name);
  button.addEventListener("click", () => selectBuffer(b));
  parent.append(button);
}
function describeBuffer(b, parent, links) {
  field(parent, "Buffer", b.name);
  field(parent, "Residency", b.address_bytes == null ? "nonresident" : "LX resident");
  field(parent, "Reason", reason(b));
  field(parent, "Spill reason", b.spill_reason);
  field(parent, "Residency reason", b.residency_reason);
  field(parent, "Metadata status", b.metadata_status);
  field(parent, "Host shape", b.host_shape);
  field(parent, "Host strides", b.host_strides);
  field(parent, "Host dtype", b.host_dtype);
  field(parent, "Spyre device size", layout(b).device_size);
  field(parent, "Spyre stride map", layout(b).stride_map);
  field(parent, "Spyre device dtype", layout(b).device_dtype);
  field(parent, "Element arrangement", layout(b).element_arrangement);
  field(parent, "Valid elements", layout(b).valid_elements);
  field(parent, "Logical tensor size", bytes(b.logical_tensor_bytes));
  field(parent, "Total solver buffer footprint", sizeText(b));
  if (b.address_bytes !== null && b.address_bytes !== undefined) {
    field(parent, "Reserved per-core LX size", bytes(b.reserved_size_bytes));
    field(parent, "LX address", bytes(b.address_bytes));
  }
  field(parent, "Lifetime", `[${format(b.start_time)}, ${format(b.end_time)}) allocator ticks`);
  field(parent, "Use ticks", (b.uses || []).map(tickLabel).join(", ") || "unknown");
  field(parent, "Producer operation", b.producer_op_index);
  field(parent, "Consumer operations", b.consumer_op_indices);
  field(parent, "Chosen division", b.chosen_division_label ?? b.chosen_division);
  // Source metadata belongs to the relayout input, never to the destination layout.
  for (const [key, value] of Object.entries(b))
    if (key.startsWith("source_") || key === "lx_view") field(parent, key, value);
  if (links) {
    const line = document.createElement("div");
    line.className = "field";
    line.textContent = "Linked operations: ";
    const indices = operationIndices(b);
    for (const i of indices) opButton(line, i);
    if (!indices.length) line.append(document.createTextNode("none in this graph"));
    parent.append(line);
  }
}
function showTooltip(b) {
  const tooltip = $("tooltip");
  tooltip.replaceChildren();
  describeBuffer(b, tooltip, false);
  tooltip.hidden = false;
}
function refreshSelection() {
  for (const [b, rect] of rects) {
    rect.classList.toggle("selected", selectedBuffer === b);
    rect.classList.toggle("related", selectedOp !== null && operationIndices(b).includes(selectedOp));
  }
  for (const [i, rect] of opRects) {
    rect.classList.toggle("selected", selectedOp === i);
    rect.classList.toggle("associated", selectedBuffer !== null && operationIndices(selectedBuffer).includes(i));
  }
  for (const [b, point] of points) {
    point.classList.toggle("selected", selectedBuffer === b);
    point.classList.toggle("related", selectedOp !== null && operationIndices(b).includes(selectedOp));
  }
  for (const [b, uses] of usePoints) for (const use of uses) {
    use.classList.toggle("selected", selectedBuffer === b);
    use.classList.toggle("related", selectedOp !== null && operationIndices(b).includes(selectedOp));
  }
  for (const [b, row] of rows) {
    row.classList.toggle("selected", selectedBuffer === b);
    row.classList.toggle("related", selectedOp !== null && operationIndices(b).includes(selectedOp));
  }
}
function selectBuffer(b) {
  if (synthetic(b) && !$("show-synthetic").checked) {
    $("show-synthetic").checked = true;
    filterBuffers();
  }
  selectedBuffer = b;
  selectedOp = null;
  const details = $("details");
  $("details-panel").open = true;
  details.replaceChildren();
  describeBuffer(b, details, true);
  refreshSelection();
  const target = rects.get(b) || points.get(b) || rows.get(b);
  if (target) target.scrollIntoView({block: "nearest", inline: "nearest"});
}
function selectOp(i) {
  selectedBuffer = null;
  selectedOp = i;
  const op = operations[i], tick = timeline[i] || {};
  const details = $("details");
  $("details-panel").open = true;
  details.replaceChildren();
  field(details, "Operation / allocator tick", `${i} / ${opTick(i)}`);
  field(details, "Type", op.op_type);
  field(details, "Name", op.operation_name);
  field(details, "Class", op.class);
  field(details, "Buffer name", op.buffer_name);
  field(details, "Watermark at operation", bytes(tick.watermark_bytes));
  field(details, "Active resident buffers", tick.active_resident_buffers);
  field(details, "Buffers at watermark", tick.buffers_at_watermark);
  field(details, "FX origins", op.origins);
  const related = buffers.filter(b => operationIndices(b).includes(i));
  const line = document.createElement("div");
  line.className = "field";
  line.textContent = `Linked buffers (${related.length}): `;
  for (const b of related) bufferButton(line, b);
  details.append(line);
  refreshSelection();
  opRects.get(i)?.scrollIntoView({block: "nearest", inline: "nearest"});
}
function drawMap() {
  const map = $("map"), left = 75, top = 30, plotHeight = 430, railY = 476;
  const plotWidth = plotWidthAtScale();
  const width = left + plotWidth + 26, height = 557;
  const capacity = Number(data.capacity_bytes) || 0;
  const x = t => left + t * plotWidth / maxTick;
  const y = address => top + plotHeight * (1 - address / Math.max(capacity, 1));
  map.replaceChildren();
  rects.clear();
  opRects.clear();
  map.setAttribute("width", width);
  map.setAttribute("height", height);
  map.setAttribute("viewBox", `0 0 ${width} ${height}`);
  for (let j = 0; j <= 5; j++) {
    const value = capacity * j / 5, lineY = y(value);
    map.append(svg("line", {x1:left,y1:lineY,x2:left+plotWidth,y2:lineY,class:"grid"}));
    const text = svg("text", {x:left-7,y:lineY+4,"text-anchor":"end",class:"axis-label"});
    text.textContent = `${Math.round(value / 1024)} KiB`;
    map.append(text);
  }
  map.append(svg("line", {x1:left,y1:top,x2:left,y2:top+plotHeight,class:"axis"}));
  map.append(svg("line", {x1:left,y1:y(capacity),x2:left+plotWidth,y2:y(capacity),class:"capacity"}));
  map.append(svg("line", {x1:left,y1:top+plotHeight,x2:left+plotWidth,y2:top+plotHeight,class:"axis"}));
  const peakBuffers = new Set(resident.filter(b => b.address_bytes + b.reserved_size_bytes === data.global_watermark_bytes && data.global_watermark_bytes).map(b => b.name));
  for (const b of resident) {
    const rect = svg("rect", {x:x(b.start_time),y:y(b.address_bytes + b.reserved_size_bytes),
      width:Math.max(1,x(b.end_time)-x(b.start_time)),
      height:Math.max(1,b.reserved_size_bytes * plotHeight / Math.max(capacity,1)),
      class:`buffer${synthetic(b) ? " relayout" : ""}${peakBuffers.has(b.name) ? " peak" : ""}`,tabindex:0,role:"button",
      "aria-label":`${format(b.name)} at LX address ${bytes(b.address_bytes)}; select for details`});
    rect.addEventListener("pointerenter", () => showTooltip(b));
    rect.addEventListener("pointerleave", () => $("tooltip").hidden = true);
    rect.addEventListener("focus", () => showTooltip(b));
    rect.addEventListener("blur", () => $("tooltip").hidden = true);
    rect.addEventListener("click", () => selectBuffer(b));
    rect.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuffer(b); }
    });
    map.append(rect);
    rects.set(b, rect);
  }
  // Address watermark includes transfer-only ticks, which may be absent from
  // the per-operation timeline. It is an extent, not summed occupancy.
  let previous = 0;
  let outline = `M ${x(0)} ${y(0)}`;
  for (let tick = 0; tick < maxTick; tick++) {
    const level = resident.reduce((peak, b) =>
      b.start_time <= tick && tick < b.end_time
        ? Math.max(peak, b.address_bytes + b.reserved_size_bytes) : peak, 0);
    if (level !== previous) {
      outline += ` L ${x(tick)} ${y(previous)} L ${x(tick)} ${y(level)}`;
      previous = level;
    }
  }
  outline += ` L ${x(maxTick)} ${y(previous)} L ${x(maxTick)} ${y(0)}`;
  map.append(svg("path", {d:outline,class:"watermark"}));
  const types = [...new Set(operations.map(op => String(op.op_type || "unknown")))];
  const palette = ["#397db8","#e58a32","#45a78a","#8b65b7","#bd5b83","#987b38","#168c9a","#c56f33"];
  const opLegend = $("op-legend");
  opLegend.replaceChildren();
  for (const [i, type] of types.entries()) {
    const item = document.createElement("span"), swatch = document.createElement("i");
    swatch.className = "swatch";
    swatch.style.background = palette[i % palette.length];
    swatch.setAttribute("aria-hidden", "true");
    item.append(swatch, document.createTextNode(type));
    opLegend.append(item);
  }
  for (const [i, op] of operations.entries()) {
    const tick = opTick(i), px = x(tick);
    const rect = svg("rect", {x:px,y:railY,width:Math.max(1,plotWidth/maxTick),height:24,
      fill:palette[types.indexOf(String(op.op_type || "unknown")) % palette.length],
      class:"op-tick",role:"button",tabindex:0,"aria-label":`Operation ${i}: ${format(op.operation_name || op.op_type)}`});
    const title = svg("title", {});
    title.textContent = `Operation ${i} at tick ${tick}: ${format(op.operation_name || op.op_type)}`;
    rect.append(title);
    rect.addEventListener("click", () => selectOp(i));
    rect.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectOp(i); }
    });
    map.append(rect);
    opRects.set(i, rect);
    if (i % Math.max(1, Math.ceil(operations.length / 28)) === 0) {
      const label = svg("text", {x:px+2,y:railY+16,class:"op-index"});
      label.textContent = String(i);
      map.append(label);
    }
  }
  for (const tick of axisTicks) {
    const mark = x(tick);
    map.append(svg("line", {x1:mark,x2:mark,y1:railY+24,y2:railY+30,class:"axis"}));
    const label = svg("text", {x:mark,y:railY+47,"text-anchor":"middle",class:"axis-label"});
    label.textContent = String(tick);
    map.append(label);
  }
  const xlabel = svg("text", {x:left,y:height-8,class:"axis-label"});
  xlabel.textContent = `Allocator tick (operation ${doubled ? "2i+1; preceding transfer 2i" : "i"})`;
  map.append(xlabel);
}
function drawNonresidentMap() {
  const map = $("nonresident-map"), left = 75, top = 22, plotHeight = 210;
  const gutter = top + plotHeight + 27, height = gutter + 48;
  const plotWidth = plotWidthAtScale(), width = left + plotWidth + 26;
  const x = tick => left + tick * plotWidth / maxTick;
  const sizes = nonresident.map(footprint).filter(size => Number.isFinite(size) && size > 0);
  const lower = Math.floor(Math.log2(sizes.length ? Math.min(...sizes) : 1));
  const upper = Math.ceil(Math.log2(sizes.length ? Math.max(...sizes) : 1));
  const span = Math.max(upper - lower, 1);
  const y = size => top + (upper - Math.log2(size)) * plotHeight / span;
  map.replaceChildren();
  points.clear();
  usePoints.clear();
  map.setAttribute("width", width);
  map.setAttribute("height", height);
  map.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (sizes.length) {
    for (let power = lower; power <= upper; power++) {
      if ((power - lower) % Math.max(1, Math.ceil(span / 6)) !== 0 && power !== upper) continue;
      const level = 2 ** power, lineY = y(level);
      map.append(svg("line", {x1:left,y1:lineY,x2:left+plotWidth,y2:lineY,class:"grid"}));
      const label = svg("text", {x:left-7,y:lineY+4,"text-anchor":"end",class:"axis-label"});
      label.textContent = `2^${power} B`;
      map.append(label);
    }
  }
  map.append(svg("line", {x1:left,y1:top,x2:left,y2:gutter,class:"axis"}));
  map.append(svg("line", {x1:left,y1:gutter,x2:left+plotWidth,y2:gutter,class:"axis"}));
  const unknown = svg("text", {x:left-7,y:gutter+4,"text-anchor":"end",class:"axis-label"});
  unknown.textContent = "unsized";
  map.append(unknown);
  for (const b of nonresident) {
    const size = footprint(b), valid = Number.isFinite(size) && size > 0;
    const lineY = valid ? y(size) : gutter;
    const start = Math.max(0, Number(b.start_time) || 0);
    const end = Math.max(start, Number(b.end_time) || start);
    const interval = svg("line", {x1:x(start),y1:lineY,x2:x(end),y2:lineY,
      class:`nonresident-interval${synthetic(b) ? " relayout" : ""}`});
    map.append(interval);
    const bufferUses = [];
    const uses = [...new Set((b.uses || []).map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
    for (const tick of uses) {
      const use = svg("circle", {cx:x(tick),cy:lineY,r:3,
        class:`nonresident-use${synthetic(b) ? " relayout" : ""}`,tabindex:0,role:"button",
        "aria-label":`${format(b.name)} reference at ${tickLabel(tick)}`});
      use.addEventListener("pointerenter", () => showTooltip(b));
      use.addEventListener("pointerleave", () => $("tooltip").hidden = true);
      use.addEventListener("focus", () => showTooltip(b));
      use.addEventListener("blur", () => $("tooltip").hidden = true);
      use.addEventListener("click", () => selectBuffer(b));
      use.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuffer(b); }
      });
      map.append(use);
      bufferUses.push(use);
    }
    usePoints.set(b, bufferUses);
    const point = svg("circle", {cx:x(start),cy:lineY,r:4,
      class:`nonresident-point${synthetic(b) ? " relayout" : ""}${valid ? "" : " unsized"}`,tabindex:0,role:"button",
      "aria-label":`${format(b.name)}: nonresident, solver footprint ${sizeText(b)}`});
    point.addEventListener("pointerenter", () => showTooltip(b));
    point.addEventListener("pointerleave", () => $("tooltip").hidden = true);
    point.addEventListener("focus", () => showTooltip(b));
    point.addEventListener("blur", () => $("tooltip").hidden = true);
    point.addEventListener("click", () => selectBuffer(b));
    point.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectBuffer(b); }
    });
    map.append(point);
    points.set(b, point);
    point.interval = interval;
  }
  for (const tick of axisTicks) {
    const mark = x(tick);
    map.append(svg("line", {x1:mark,x2:mark,y1:gutter,y2:gutter+6,class:"axis"}));
    const label = svg("text", {x:mark,y:gutter+21,"text-anchor":"middle",class:"axis-label"});
    label.textContent = String(tick);
    map.append(label);
  }
  const xlabel = svg("text", {x:left,y:height-8,class:"axis-label"});
  xlabel.textContent = "Allocator tick (aligned with LX address plot)";
  map.append(xlabel);
}
function setZoom(value, fit = false, anchorX = null) {
  const oldWidth = Number($("map").getAttribute("width")) - 101 || plotWidthAtScale();
  const x = anchorX ?? mapScroll.clientWidth / 2;
  const position = (mapScroll.scrollLeft + x - 75) / oldWidth;
  zoom.value = String(Math.max(0, Math.min(8, Number(value))));
  $("zoom-label").textContent = `${Number(scale().toFixed(2))}×`;
  drawMap();
  drawNonresidentMap();
  filterBuffers();
  refreshSelection();
  mapScroll.scrollLeft = fit ? 0 : 75 + position * plotWidthAtScale() - x;
}
function fillTable() {
  const reasons = [...new Set(nonresident.map(reason))].sort();
  for (const name of reasons) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    $("reason").append(option);
  }
  const body = $("nonresident-rows");
  body.replaceChildren();
  for (const b of [...nonresident].sort((a, b) =>
    linkText(a).localeCompare(linkText(b)) ||
    String(a.name).localeCompare(String(b.name)))) {
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.addEventListener("click", e => { if (e.target.tagName !== "BUTTON") selectBuffer(b); });
    row.addEventListener("keydown", e => {
      if (e.target === row && (e.key === "Enter" || e.key === " ")) {
        e.preventDefault(); selectBuffer(b);
      }
    });
    const cell = value => {
      const td = document.createElement("td");
      td.textContent = format(value);
      row.append(td);
      return td;
    };
    const name = cell("");
    bufferButton(name, b);
    cell(`${reason(b)} · metadata: ${format(b.metadata_status)}`);
    cell(`${format(b.host_shape)} / ${format(b.host_dtype)}`);
    cell(layoutText(b));
    cell(sizeText(b));
    const opLinks = cell("");
    const indices = operationIndices(b);
    for (const i of indices) opButton(opLinks, i);
    if (!indices.length) opLinks.textContent = "No graph operation link";
    body.append(row);
    rows.set(b, row);
  }
}
function filterBuffers() {
  const show = $("show-synthetic").checked;
  const query = $("search").value.toLowerCase().trim();
  const selectedReason = $("reason").value;
  for (const [b, rect] of rects) rect.style.display = synthetic(b) && !show ? "none" : "";
  let visible = 0;
  for (const [b, row] of rows) {
    const searchable = [format(b.name), reason(b), format(b.metadata_status),
      format(b.host_shape), format(b.host_strides), format(b.host_dtype),
      layoutText(b), sizeText(b), bytes(b.reserved_size_bytes),
      ...(b.uses || []).map(tickLabel), ...operationIndices(b).map(i => `op ${i}`)].join(" ").toLowerCase();
    row.hidden = (synthetic(b) && !show) || (selectedReason && reason(b) !== selectedReason) ||
      (query && !searchable.includes(query));
    if (!row.hidden) visible++;
    const point = points.get(b);
    point.style.display = row.hidden ? "none" : "";
    point.interval.style.display = row.hidden ? "none" : "";
    for (const use of usePoints.get(b) || []) use.style.display = row.hidden ? "none" : "";
  }
  const hiddenResident = resident.filter(synthetic).length;
  const hiddenNonresident = nonresident.filter(synthetic).length;
  $("resident-count").textContent = `${resident.length - (show ? 0 : hiddenResident)} of ${resident.length} resident rectangles shown${show ? "" : `; ${hiddenResident} synthetic hidden`}.`;
  $("nonresident-count").textContent = `${visible} of ${nonresident.length} nonresident buffers shown${show ? "" : `; ${hiddenNonresident} synthetic hidden`}.`;
}
$("title").textContent = `${format(data.aot_graph_name || "unnamed graph")} — interactive LX allocation map`;
$("summary").textContent = `${operations.length} operations · ${resident.length} LX-resident · ${nonresident.length} nonresident graph buffers · watermark ${bytes(data.global_watermark_bytes)} / capacity ${bytes(data.capacity_bytes)} · Inductor graph ID ${format(data.inductor_graph_id)}`;
$("no-consumer-count").textContent = `${noConsumerBuffers.length} known no-consumer buffers omitted from the nonresident plot and table; retained in raw capture.`;
$("show-synthetic").addEventListener("change", filterBuffers);
$("search").addEventListener("input", filterBuffers);
$("reason").addEventListener("change", filterBuffers);
const chartStack = $("charts");
chartStack.addEventListener("pointerenter", () => chartStack.focus({preventScroll: true}));
chartStack.addEventListener("pointerdown", () => chartStack.focus({preventScroll: true}));
chartStack.addEventListener("pointermove", event => {
  const bounds = chartStack.getBoundingClientRect();
  chartPointerX = Math.max(0, Math.min(chartStack.clientWidth, event.clientX - bounds.left));
});
document.addEventListener("keydown", event => {
  if (!chartStack.contains(event.target) && !chartStack.contains(document.activeElement)) return;
  if (event.ctrlKey || event.metaKey || event.altKey ||
      event.target?.closest?.("input, select, textarea, button, a, [contenteditable='true']")) return;
  const key = event.key.toLowerCase();
  if (key === "a" || key === "d") {
    event.preventDefault();
    chartStack.scrollLeft += key === "a" ? -Math.max(48, chartStack.clientWidth * 0.12)
      : Math.max(48, chartStack.clientWidth * 0.12);
    return;
  }
  if (key !== "w" && key !== "s") return;
  event.preventDefault();
  setZoom(Number(zoom.value) + (key === "w" ? 1 : -1), false, chartPointerX);
}, true);
$("zoom-fit").addEventListener("click", () => setZoom(0, true));
$("zoom-out").addEventListener("click", () => setZoom(Number(zoom.value) - 1));
$("zoom-in").addEventListener("click", () => setZoom(Number(zoom.value) + 1));
zoom.addEventListener("input", e => setZoom(e.target.value));
window.addEventListener("resize", () => setZoom(zoom.value, Number(zoom.value) === 0));
$("details-panel").addEventListener("toggle", () => {
  $("details-panel").parentElement.classList.toggle("details-closed", !$("details-panel").open);
  setZoom(zoom.value, Number(zoom.value) === 0);
});
drawMap();
drawNonresidentMap();
fillTable();
filterBuffers();
</script></body></html>
"""
    preview_lx, preview_nonresident, preview_rows = _static_preview(record)
    rejected = sum(
        b["address_bytes"] is None
        and (
            b.get("is_synthetic_relayout")
            or b["name"].startswith("__spyre_lx_relayout__")
        )
        for b in record["buffers"]
    )
    no_consumers = sum(
        b["address_bytes"] is None
        and not b.get("is_synthetic_relayout")
        and not b["name"].startswith("__spyre_lx_relayout__")
        and isinstance(b.get("consumer_op_indices"), list)
        and not b["consumer_op_indices"]
        for b in record["buffers"]
    )
    path.write_text(
        template.replace("__CAPTURE_DATA__", payload)
        .replace("__LX_PREVIEW__", preview_lx)
        .replace("__NONRESIDENT_PREVIEW__", preview_nonresident)
        .replace("__ROWS_PREVIEW__", preview_rows)
        .replace("__REJECTED_COUNT__", str(rejected))
        .replace("__NO_CONSUMER_COUNT__", str(no_consumers)),
        encoding="utf-8",
    )


def _render_one(record: dict, index: int, output_dir: Path, dpi: int) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
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
    max_tick = max(
        max((buffer["end_time"] for buffer in record["buffers"]), default=0),
        max(
            (tick.get("allocation_tick", tick["op_index"]) + 1 for tick in timeline),
            default=0,
        ),
    )
    peak_buffers = {
        buffer["name"]
        for buffer in resident
        if buffer["address_bytes"] + buffer["reserved_size_bytes"] == watermark
        and watermark
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
                facecolor=PALETTE[6]
                if buffer.get("is_synthetic_relayout")
                else (PALETTE[1] if is_peak else PALETTE[0]),
                edgecolor="#9c4c17"
                if is_peak
                else ("#075c66" if buffer.get("is_synthetic_relayout") else "#24577f"),
                alpha=0.72 if is_peak or buffer.get("is_synthetic_relayout") else 0.38,
                linewidth=2 if is_peak and buffer.get("is_synthetic_relayout") else 0.7,
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
        [tick.get("allocation_tick", tick["op_index"]) for tick in timeline]
        + [max_tick],
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
        xlim=(0, max(max_tick, 1)),
        ylim=(0, max(capacity, 1)),
        ylabel="LX address (KiB)",
        title="Reserved address extents by allocator tick; half-open lifetimes [start, end)",
    )
    allocation_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1024:.0f}")
    )
    allocation_ax.set_xticks(range(0, max_tick + 1, max(1, max_tick // 16)))
    allocation_ax.set_xticks(range(max_tick + 1), minor=True)
    allocation_ax.set_yticks(range(0, capacity + 1, 32768), minor=True)
    allocation_ax.grid(True, axis="both", alpha=0.2)
    allocation_ax.grid(True, which="minor", axis="both", alpha=0.06)
    allocation_ax.legend(
        handles=[
            Patch(facecolor=PALETTE[0], alpha=0.38, label="Other LX-resident buffer"),
            Patch(facecolor=PALETTE[6], alpha=0.72, label="Synthetic relayout buffer"),
            Patch(
                facecolor=PALETTE[1],
                alpha=0.72,
                label="Buffer reaching the highest LX address",
            ),
            Line2D([], [], color="#c72d32", label="Highest live LX address"),
            Line2D([], [], color="#666", linestyle="--", label="Planner LX capacity"),
        ],
        loc="upper left",
        fontsize=8,
    )
    allocation_ax.text(
        max(0.5, max_tick - 0.5),
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
        x = timeline[operation["op_index"]].get(
            "allocation_tick", operation["op_index"]
        )
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
    op_ax.set(
        xlim=(0, max(max_tick, 1)),
        ylim=(0, 1),
        title="Operation type by allocator tick",
    )
    op_ax.set_xticks(range(0, max_tick + 1, max(1, max_tick // 16)))
    op_ax.set_yticks([])
    op_ax.grid(True, axis="x", alpha=0.12)
    op_ax.set_xlabel("Allocator tick (operation indices are labeled on color blocks)")
    op_ax.text(
        0.01,
        -0.5,
        "Operation strip colors repeat by op type; numbers are graph op indices.",
        transform=op_ax.transAxes,
        fontsize=8,
    )

    stem = f"{record.get('aot_graph_name') or f'graph-{index}'}-solve-{index}"
    interactive_path = output_dir / f"{stem}-interactive.html"
    _render_interactive_html(record, interactive_path)
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
        "interactive_html": interactive_path.name,
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
            tick_index = tick.get("allocation_tick", tick["op_index"])
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
        operation_watermark = max(
            (tick.get("watermark_bytes", 0) for tick in timeline), default=0
        )
        if operation_watermark > watermark:
            raise SystemExit(f"Record {index} operation watermark exceeds global")
        if (
            record.get("max_operation_watermark_bytes", operation_watermark)
            != operation_watermark
        ):
            raise SystemExit(f"Record {index} maximum operation watermark mismatch")
        if watermark != record.get("global_watermark_bytes"):
            raise SystemExit(f"Record {index} global watermark mismatch")
        summary = _render_one(record, index, args.output_dir, args.dpi)
        summary["timeline_matches_global_watermark"] = operation_watermark == watermark
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
