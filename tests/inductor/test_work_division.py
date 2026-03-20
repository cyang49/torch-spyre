# Copyright 2025 The Torch-Spyre Authors.
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

"""
Tests for pointwise ops with broadcast inputs under multi-core work division.

Verifies that core_idx_to_slice_offset correctly computes memory offsets for
both broadcast and non-broadcast inputs when the op is split across multiple
dimensions simultaneously (e.g., SENCORES=12 with shape [6, 17, 32, 64] gives
op_dim_splits=[6, 1, 2, 1]).
"""

import unittest

import torch

from utils_inductor import compare_with_cpu

SHAPE_4D = (6, 17, 32, 64)
BROADCAST_SHAPE_4D = (1, 17, 32, 64)

SHAPE_3D = (8, 6, 1088)
BROADCAST_SHAPE_3D = (1, 6, 1088)


def _make_tensor(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float16)


class TestPointwiseWorkDivision(unittest.TestCase):
    def test_add_4d_no_broadcast(self):
        a = _make_tensor(SHAPE_4D)
        b = _make_tensor(SHAPE_4D)
        compare_with_cpu(lambda x, y: torch.add(x, y), a, b)

    def test_add_4d_broadcast(self):
        a = _make_tensor(SHAPE_4D)
        b = _make_tensor(BROADCAST_SHAPE_4D)
        compare_with_cpu(lambda x, y: torch.add(x, y), a, b)

    def test_add_3d_no_broadcast(self):
        a = _make_tensor(SHAPE_3D)
        b = _make_tensor(SHAPE_3D)
        compare_with_cpu(lambda x, y: torch.add(x, y), a, b)

    def test_add_3d_broadcast(self):
        a = _make_tensor(SHAPE_3D)
        b = _make_tensor(BROADCAST_SHAPE_3D)
        compare_with_cpu(lambda x, y: torch.add(x, y), a, b)


if __name__ == "__main__":
    unittest.main()
