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

"""Capture graph-operation and LX-allocation records from allocator solves."""

from __future__ import annotations
# ruff: noqa: E402, I001, RUF100

import json
import linecache
import os
import threading
import time
from pathlib import Path

import regex

_capture_dir_raw = os.environ.get("SPYRE_GRAPH_CAPTURE_DIR")
if not _capture_dir_raw:
    raise RuntimeError("SPYRE_GRAPH_CAPTURE_DIR must name an output directory")
_CAPTURE_DIR = Path(_capture_dir_raw).resolve()
_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_STATUS_PATH = _CAPTURE_DIR / f"processes.{os.getpid()}.jsonl"
with _STATUS_PATH.open("a", encoding="utf-8") as stream:
    stream.write(
        json.dumps({"event": "sitecustomize_started", "pid": os.getpid()}) + "\n"
    )

# Import torch before torch-spyre. torch backend autoload resolves torch-spyre.
import torch
from torch._functorch._aot_autograd.logging_utils import (
    get_aot_compilation_context,
    get_aot_graph_name,
)

from torch_spyre._inductor.pass_utils import op_short_name
from torch_spyre._inductor.scratchpad import allocator as allocator_module

_FRAME_RE = regex.compile(r'File "([^"]+)", line (\d+)')
_ORIGINAL_POST_SOLVE = allocator_module.CoOptimizingAllocator._post_solve


def _node_record(node: torch.fx.Node) -> dict:
    meta = getattr(node, "meta", None) or {}
    frames = _FRAME_RE.findall(meta.get("stack_trace") or "")
    user_frames = [frame for frame in frames if "/torch/" not in frame[0]]
    frame = (user_frames or frames)[-1] if (user_frames or frames) else None
    source = None
    if frame is not None:
        source_file, line_text = frame
        line = int(line_text)
        source = {
            "file": source_file,
            "line": line,
            "code": linecache.getline(source_file, line).strip() or None,
        }
    module_stack = []
    raw_stack = meta.get("nn_module_stack") or {}
    if isinstance(raw_stack, dict):
        for key, entry in raw_stack.items():
            module = entry[1] if len(entry) > 1 else None
            module_stack.append(
                {
                    "node": str(key),
                    "fqn": str(entry[0]) if entry else None,
                    "type": getattr(module, "__qualname__", type(module).__qualname__),
                }
            )
    aten = meta.get("original_aten")
    return {
        "name": node.name,
        "fx_op": node.op,
        "target": str(node.target),
        "original_aten": str(aten) if aten is not None else None,
        "source": source,
        "module_stack": module_stack,
    }


def _origins(graph, operation) -> list[dict]:
    nodes = []
    for carrier in (operation, getattr(operation, "data", None)):
        if carrier is None:
            continue
        nodes.extend(
            node
            for node in getattr(carrier, "origins", ()) or ()
            if isinstance(node, torch.fx.Node)
        )
        origin = getattr(carrier, "origin_node", None)
        if isinstance(origin, torch.fx.Node):
            nodes.append(origin)

    try:
        buffer_name = operation.get_name()
    except (AttributeError, RuntimeError):
        buffer_name = getattr(operation, "name", None)
    if buffer_name:
        for node, tensor in graph.env.items():
            if not isinstance(node, torch.fx.Node):
                continue
            try:
                if tensor.get_name() == buffer_name:
                    nodes.append(node)
            except (AttributeError, RuntimeError, TypeError):
                continue
    unique = {node.name: node for node in nodes}
    return [_node_record(unique[name]) for name in sorted(unique)]


def _capture_post_solve(allocator, graph, allocation, accepted_lx_relayouts):
    try:
        aot_path, model_name, nth_graph = get_aot_compilation_context()
        aot_graph_name = get_aot_graph_name()
    except (AttributeError, RuntimeError):
        aot_path, model_name, nth_graph, aot_graph_name = [], None, None, None

    operations = []
    for index, operation in enumerate(graph.operations):
        try:
            name = operation.get_name()
        except (AttributeError, RuntimeError):
            name = getattr(operation, "name", None)
        try:
            operation_name = operation.get_operation_name()
        except (AttributeError, RuntimeError):
            operation_name = None
        operations.append(
            {
                "op_index": index,
                "buffer_name": name,
                "operation_name": operation_name,
                "op_type": op_short_name(operation),
                "class": type(operation).__name__,
                "origins": _origins(graph, operation),
            }
        )

    buffers = []
    resident = []
    for buffer in allocation:
        division = getattr(buffer, "chosen_division", None)
        reserved_size = (
            buffer.per_core_size(division) if division is not None else buffer.size
        )
        record = {
            "name": buffer.name,
            "logical_size_bytes": buffer.size,
            "reserved_size_bytes": reserved_size,
            "start_time": buffer.start_time,
            "end_time": buffer.end_time,
            "uses": list(buffer.uses),
            "address_bytes": buffer.address,
            "chosen_division": division,
            "chosen_division_label": (
                buffer.core_divisions[division].label if division is not None else None
            ),
            "spill_reason": buffer.spill_reason,
            "residency_reason": buffer.residency_reason,
            "is_synthetic_relayout": buffer.name.startswith("__spyre_lx_relayout__"),
        }
        buffers.append(record)
        if buffer.address is not None:
            resident.append(record)

    timeline = []
    for operation in operations:
        index = operation["op_index"]
        active = [
            buffer
            for buffer in resident
            if buffer["start_time"] <= index < buffer["end_time"]
        ]
        watermark = max(
            (
                buffer["address_bytes"] + buffer["reserved_size_bytes"]
                for buffer in active
            ),
            default=0,
        )
        timeline.append(
            {
                **operation,
                "watermark_bytes": watermark,
                "capacity_bytes": allocator.size,
                "active_resident_buffers": [buffer["name"] for buffer in active],
                "buffers_at_watermark": [
                    buffer["name"]
                    for buffer in active
                    if buffer["address_bytes"] + buffer["reserved_size_bytes"]
                    == watermark
                ]
                if watermark
                else [],
            }
        )

    global_watermark = max(
        (
            buffer["address_bytes"] + buffer["reserved_size_bytes"]
            for buffer in resident
        ),
        default=0,
    )
    record = {
        "schema": "spyre-lx-allocation-graph-trace-v2",
        "pid": os.getpid(),
        "timestamp_unix": time.time(),
        "aot_graph_name": aot_graph_name,
        "aot_graph_path": list(aot_path) if aot_path is not None else None,
        "aot_model_name": model_name,
        "aot_nth_graph": nth_graph,
        "inductor_graph_name": getattr(graph, "name", None),
        "inductor_graph_id": getattr(graph, "graph_id", None),
        "inductor_post_grad_graph_id": getattr(graph, "post_grad_graph_id", None),
        "allocator": type(allocator).__name__,
        "solver": (
            allocator.layout_planning.__name__
            if allocator.layout_planning is not None
            else None
        ),
        "capacity_bytes": allocator.size,
        "global_watermark_bytes": global_watermark,
        "max_operation_watermark_bytes": max(
            (row["watermark_bytes"] for row in timeline), default=0
        ),
        "timeline_matches_global_watermark": (
            max((row["watermark_bytes"] for row in timeline), default=0)
            == global_watermark
        ),
        "all_resident_spans_within_capacity": all(
            buffer["address_bytes"] + buffer["reserved_size_bytes"] <= allocator.size
            for buffer in resident
        ),
        "operation_count": len(operations),
        "buffer_count": len(buffers),
        "resident_count": len(resident),
        "accepted_lx_relayouts": [repr(plan) for plan in accepted_lx_relayouts],
        "operations": operations,
        "timeline": timeline,
        "buffers": buffers,
    }
    path = _CAPTURE_DIR / f"allocations.{os.getpid()}.jsonl"
    with _LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
        stream.flush()
    return _ORIGINAL_POST_SOLVE(allocator, graph, allocation, accepted_lx_relayouts)


allocator_module.CoOptimizingAllocator._post_solve = _capture_post_solve
