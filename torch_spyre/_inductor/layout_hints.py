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

"""Compiler-only tensor layout hint plumbing."""

import torch


REQUIRE_LAYOUT_KEY = "require_layout"


_VIEW_OPS = {
    torch.ops.aten.expand.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
}


def _matmul_producer(source: torch.fx.Node) -> torch.fx.Node:
    """Walk output-only views back to their BMM/MM producer."""
    while source.target in _VIEW_OPS:
        source = source.args[0]
        if not isinstance(source, torch.fx.Node):
            raise TypeError("require_layout expects a tensor producer")
    return source


def apply_require_layout(graph: torch.fx.Graph) -> None:
    """Move static marker layout request onto its matmul producer, then erase it."""
    for node in list(graph.nodes):
        if node.target != torch.ops.spyre.require_layout.default:
            continue
        source, device_size, stride_map = node.args
        if not isinstance(source, torch.fx.Node):
            raise TypeError("require_layout expects a tensor producer")
        if not all(isinstance(v, int) for v in (*device_size, *stride_map)):
            raise TypeError("require_layout layout must be static")
        source = _matmul_producer(source)
        source.meta.setdefault("custom", {})[REQUIRE_LAYOUT_KEY] = (
            list(device_size),
            list(stride_map),
        )
        node.replace_all_uses_with(node.args[0])
        graph.erase_node(node)
