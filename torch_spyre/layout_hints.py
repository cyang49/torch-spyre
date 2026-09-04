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

"""Compiler-directed tensor-layout hints."""

import torch


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def require_layout(x: torch.Tensor, layout):
    """Require ``layout`` from compiled matmul or pointwise producer.

    Compiled producers support float16, bfloat16, and float32. Unsupported
    producer/layout combinations raise during compilation. Eager execution uses
    ``to(device_layout=...)``.
    """
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"require_layout supports {_SUPPORTED_DTYPES}; got {x.dtype}")
    if torch.compiler.is_compiling():
        return torch.ops.spyre.require_layout(
            x, list(layout.device_size), list(layout.stride_map)
        )
    return x.to(device_layout=layout)
