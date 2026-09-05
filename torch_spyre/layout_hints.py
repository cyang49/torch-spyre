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


def require_layout(
    x: torch.Tensor, device_size: list[int], stride_map: list[int]
) -> torch.Tensor:
    """Require physical output geometry from a compiled producer.

    Args:
      x: FP16, BF16, or FP32 output from a supported compiled producer.
      device_size: Static physical device extents.
      stride_map: Static logical strides corresponding to ``device_size``.

    This compiler-only constraint uses ``x.dtype`` and
    ``ElementArrangement.STANDARD``. It supports matmul and tensor
    add/sub/mul/div producers, including output-only view chains. Unsupported
    producers or illegal geometry raise during compilation. For eager conversion
    or non-STANDARD layouts, use ``x.to(device_layout=layout)`` instead.
    """
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"require_layout supports {_SUPPORTED_DTYPES}; got {x.dtype}")
    if len(device_size) != len(stride_map) or not device_size:
        raise ValueError(
            "require_layout device_size and stride_map must have equal nonzero lengths"
        )
    if any(extent <= 0 for extent in device_size):
        raise ValueError("require_layout device_size extents must be positive")
    if not torch.compiler.is_compiling():
        raise RuntimeError("require_layout is available only inside torch.compile")
    return torch.ops.spyre.require_layout(x, list(device_size), list(stride_map))
