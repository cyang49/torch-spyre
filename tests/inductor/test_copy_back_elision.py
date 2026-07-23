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

from unittest.mock import patch

import pytest
import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config as inductor_config
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import propagate_layouts


DEVICE = torch.device("spyre")
SIZE = 128


def _compile_and_source(fn, *args, device_layouts=None):
    torch._dynamo.reset()
    if device_layouts is None:
        device_layouts = [None] * len(args)
    device_args = tuple(
        arg.to(DEVICE) if layout is None else arg.to(device_layout=layout)
        for arg, layout in zip(args, device_layouts)
    )
    compiled_out, code = run_and_get_code(torch.compile(fn), *device_args)
    return compiled_out, code[0], device_args


def _assert_copy_back_elided(source: str) -> None:
    assert "sdsc_fused_copy" not in source


def _assert_copy_back_preserved(source: str) -> None:
    assert any("copy_" in line for line in source.splitlines() if "sdsc_fused_" in line)


@pytest.mark.parametrize("global_stick_optimizer", [True, False])
def test_mm_out_copy_back_into_input_is_elided(global_stick_optimizer):
    torch.manual_seed(0xAFFE)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)
    w = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def fn(x, y, z, w):
        torch.mm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    with patch.object(
        inductor_config, "global_stick_optimizer", global_stick_optimizer
    ):
        actual, source, device_args = _compile_and_source(fn, x, y, z, w)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


@pytest.mark.parametrize("policy", ["prefer", "force"])
@pytest.mark.parametrize("global_stick_optimizer", [True, False])
def test_batchmatmul_out_prefers_preallocated_output_layout(
    policy, global_stick_optimizer
):
    torch.manual_seed(0xB17)
    shape = (2, SIZE, SIZE)
    x = torch.randn(*shape, dtype=torch.float16)
    y = torch.randn(*shape, dtype=torch.float16)
    z = torch.randn(*shape, dtype=torch.float16)
    w = torch.randn(*shape, dtype=torch.float16)
    z_layout = SpyreTensorLayout(
        list(shape),
        [SIZE * SIZE, SIZE, 1],
        torch.float16,
        [1, 0, 2],
    )

    def fn(x, y, z, w):
        torch.bmm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    with inductor_config.patch(
        {
            "global_stick_optimizer": global_stick_optimizer,
            "preallocated_output_layout_policy": policy,
            "preallocated_output_layout_ops": "batchmatmul",
        }
    ):
        actual, source, device_args = _compile_and_source(
            fn, x, y, z, w, device_layouts=[None, None, z_layout, None]
        )

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    assert device_args[2].device_tensor_layout() == z_layout
    _assert_copy_back_elided(source)


def test_batchmatmul_out_preallocated_layout_respects_whitelist():
    torch.manual_seed(0xB18)
    shape = (2, SIZE, SIZE)
    x = torch.randn(*shape, dtype=torch.float16)
    y = torch.randn(*shape, dtype=torch.float16)
    z = torch.randn(*shape, dtype=torch.float16)
    w = torch.randn(*shape, dtype=torch.float16)
    z_layout = SpyreTensorLayout(
        list(shape),
        [SIZE * SIZE, SIZE, 1],
        torch.float16,
        [1, 0, 2],
    )

    def fn(x, y, z, w):
        torch.bmm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    with inductor_config.patch({"preallocated_output_layout_ops": ""}):
        actual, source, device_args = _compile_and_source(
            fn, x, y, z, w, device_layouts=[None, None, z_layout, None]
        )

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_preserved(source)


def test_preallocated_output_layout_ops_rejects_unsupported_ops():
    with inductor_config.patch({"preallocated_output_layout_ops": "add,batchmatmul"}):
        with pytest.raises(ValueError, match="Invalid preallocated_output_layout_ops"):
            propagate_layouts._preallocated_output_layout_ops()


def test_batchmatmul_out_infeasible_preallocated_layout_preserves_copy_back():
    torch.manual_seed(0xB19)
    shape = (2, SIZE, SIZE)
    x = torch.randn(*shape, dtype=torch.float16)
    y = torch.randn(*shape, dtype=torch.float16)
    z = torch.randn(*shape, dtype=torch.float16)
    w = torch.randn(*shape, dtype=torch.float16)
    z_layout = SpyreTensorLayout(
        list(shape),
        [SIZE * SIZE, SIZE, 1],
        torch.float16,
        [0, 2, 1],
    )

    def fn(x, y, z, w):
        torch.bmm(x, y, out=z)
        return z + w

    expected_z = z.clone()
    expected = fn(x, y, expected_z, w)
    with inductor_config.patch(
        {
            "preallocated_output_layout_policy": "force",
            "preallocated_output_layout_ops": "batchmatmul",
        }
    ):
        actual, source, device_args = _compile_and_source(
            fn, x, y, z, w, device_layouts=[None, None, z_layout, None]
        )

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_preserved(source)


@pytest.mark.parametrize("shape", [(SIZE, SIZE), (2, 65, 130)])
def test_pointwise_out_copy_back_into_input_is_elided(shape):
    torch.manual_seed(1)
    x = torch.randn(*shape, dtype=torch.float16)
    y = torch.randn(*shape, dtype=torch.float16)
    z = torch.randn(*shape, dtype=torch.float16)
    tail = torch.randn(*shape, dtype=torch.float16)

    def fn(x, y, z, tail):
        torch.add(x, y, out=z)
        return z * tail

    expected_z = z.clone()
    expected = fn(x, y, expected_z, tail)
    actual, source, device_args = _compile_and_source(fn, x, y, z, tail)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_elided(source)


def test_pointwise_out_copy_back_is_preserved_when_span_overflow_hints_enabled():
    torch.manual_seed(11)
    shape = (SIZE, SIZE)
    x = torch.randn(*shape, dtype=torch.float16)
    y = torch.randn(*shape, dtype=torch.float16)
    z = torch.randn(*shape, dtype=torch.float16)
    tail = torch.randn(*shape, dtype=torch.float16)

    def fn(x, y, z, tail):
        torch.add(x, y, out=z)
        return z * tail

    expected_z = z.clone()
    expected = fn(x, y, expected_z, tail)
    with inductor_config.patch({"ignore_span_overflow_hints": False}):
        actual, source, device_args = _compile_and_source(fn, x, y, z, tail)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_preserved(source)


def test_required_copy_backs_are_preserved():
    torch.manual_seed(2)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def reads_old_destination(x, y, z):
        old_z = z + 1.0
        torch.mm(x, y, out=z)
        return old_z + z

    def returns_destination(x, y, z):
        torch.mm(x, y, out=z)
        return z

    for fn in (reads_old_destination, returns_destination):
        expected_z = z.clone()
        expected = fn(x, y, expected_z)
        actual, source, device_args = _compile_and_source(fn, x, y, z)

        torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
        torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
        _assert_copy_back_preserved(source)


def test_infeasible_target_layout_preserves_copy_back():
    torch.manual_seed(3)
    x = torch.randn(SIZE, SIZE, dtype=torch.float16)
    y = torch.randn(SIZE, SIZE, dtype=torch.float16)
    z = torch.randn(SIZE, SIZE, dtype=torch.float16)

    def fn(x, y, z):
        torch.add(x, y, out=z)
        return z + 1.0

    def reverse_target_layout(target, name):
        layout = target.get_layout()
        return SpyreTensorLayout(
            [int(s) for s in layout.size],
            [int(s) for s in layout.stride],
            layout.dtype,
            [1, 0],
        )

    expected_z = z.clone()
    expected = fn(x, y, expected_z)
    with patch.object(
        propagate_layouts, "_target_device_layout", reverse_target_layout
    ):
        actual, source, device_args = _compile_and_source(fn, x, y, z)

    torch.testing.assert_close(actual.cpu(), expected, atol=0.1, rtol=0.1)
    torch.testing.assert_close(device_args[2].cpu(), expected_z, atol=0.1, rtol=0.1)
    _assert_copy_back_preserved(source)
