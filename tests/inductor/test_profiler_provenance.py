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

"""Device-free tests for profiler-facing kernel provenance identity."""

import ctypes
import dataclasses
from unittest.mock import patch

import pytest

import torch  # noqa: F401
from sympy import Integer

from torch_spyre._inductor.op_spec import DebugHandle, LoopSpec, OpSpec, SourceLoc
from torch_spyre._inductor.profiler_provenance import (
    AIUPTI_ACTIVITY_NAME_MAX_BYTES,
    build_kernel_provenance_descriptor,
    extract_kernel_provenance_key,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile
from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner


def _handle(
    handle_id: int,
    *,
    aten_op: str | None = "aten.mm.default",
    source: SourceLoc | None = SourceLoc("/workspace/model.py", 117),
    fused_from: tuple[DebugHandle, ...] = (),
) -> DebugHandle:
    return DebugHandle(
        id=handle_id,
        source=source,
        aten_op=aten_op,
        ir_chain=(f"op{handle_id}",),
        fused_from=fused_from,
    )


def _op(handle: DebugHandle | None) -> OpSpec:
    return OpSpec(
        op="identity",
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
        debug_handle=handle,
    )


class TestKernelProvenanceDescriptor:
    def test_returns_none_without_handles(self):
        specs = [
            _op(None),
            LoopSpec(count=Integer(2), body=[_op(None)]),
        ]

        assert build_kernel_provenance_descriptor(specs) is None

    def test_collects_nested_handles_in_order_and_deduplicates_ids(self):
        first = _handle(9)
        second = _handle(12, aten_op="aten.add.Tensor")
        specs = [
            _op(first),
            LoopSpec(
                count=Integer(4),
                body=[
                    _op(second),
                    LoopSpec(count=Integer(2), body=[_op(first)]),
                ],
            ),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor is not None
        assert descriptor.debug_handle_ids == ("9", "12")
        assert extract_kernel_provenance_key(descriptor.event_name) == descriptor.key
        assert not hasattr(descriptor, "fusion_context")

    def test_is_deterministic_and_order_sensitive(self):
        first = _handle(9)
        second = _handle(12)

        forward = build_kernel_provenance_descriptor([_op(first), _op(second)])
        repeated = build_kernel_provenance_descriptor([_op(first), _op(second)])
        reverse = build_kernel_provenance_descriptor([_op(second), _op(first)])

        assert forward is not None
        assert repeated == forward
        assert reverse is not None
        assert forward.key == "336396603bb9d63e"
        assert reverse.key != forward.key
        assert len(forward.key) == 16
        assert int(forward.key, 16) < 1 << 63

    def test_descriptor_is_frozen(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(1))])

        assert descriptor is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            descriptor.key = "0" * 16  # type: ignore[misc]


class TestKernelProvenanceEventName:
    def test_uses_agreed_source_and_aten_headline(self):
        handle = _handle(42)

        descriptor = build_kernel_provenance_descriptor([_op(handle), _op(handle)])

        assert descriptor is not None
        assert descriptor.event_name == (
            f"spyre_kernel_aten_mm_default_at_model_py_117_{descriptor.key}"
        )

    def test_does_not_choose_a_primary_for_conflicting_handles(self):
        specs = [
            _op(_handle(1, source=SourceLoc("first.py", 10))),
            _op(
                _handle(
                    2,
                    aten_op="aten.add.Tensor",
                    source=SourceLoc("second.py", 20),
                )
            ),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor is not None
        assert descriptor.event_name == (
            f"spyre_kernel_fused_at_unknown_0_{descriptor.key}"
        )

    def test_single_fused_handle_without_headline_is_labeled_fused(self):
        constituent = _handle(1, source=SourceLoc("first.py", 10))
        fused = _handle(
            2,
            aten_op=None,
            source=None,
            fused_from=(constituent,),
        )

        descriptor = build_kernel_provenance_descriptor([_op(fused)])

        assert descriptor is not None
        assert f"_fused_at_unknown_0_{descriptor.key}" in descriptor.event_name

    def test_sanitizes_and_bounds_name_with_step_suffix_reservation(self):
        long_component = "α/" + "very-long-name." * 20
        handle = _handle(
            7,
            aten_op=f"aten.{long_component}.default",
            source=SourceLoc(f"/tmp/{long_component}.py", 123456),
        )

        descriptor = build_kernel_provenance_descriptor([_op(handle)])

        assert descriptor is not None
        assert descriptor.event_name.isascii()
        # PR #2930 uses size_t for this JobPlan command index. Match the local
        # extension ABI instead of assuming that every target uses 64-bit size_t.
        size_t_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        largest_step_suffix = f"#{(1 << size_t_bits) - 1}"
        final_name = f"{descriptor.event_name}{largest_step_suffix}"
        assert len(final_name.encode("ascii")) <= AIUPTI_ACTIVITY_NAME_MAX_BYTES
        assert descriptor.key in final_name

    def test_rejects_unrelated_or_malformed_event_names(self):
        assert extract_kernel_provenance_key("sdsc_mm_0") is None
        assert extract_kernel_provenance_key("spyre_kernel_not-hex_fused") is None
        assert extract_kernel_provenance_key("xspyre_kernel_0123456789abcdef") is None


class TestKernelProvenancePropagation:
    def test_async_compile_builds_descriptor_from_finalized_specs(self):
        specs = [
            _op(_handle(9)),
            LoopSpec(count=Integer(2), body=[_op(_handle(12))]),
        ]
        runner = object()

        with (
            patch(
                "torch_spyre.execution.async_compile.get_output_dir",
                return_value="/tmp/kernel",
            ),
            patch("torch_spyre.execution.async_compile.generate_bundle"),
            patch("torch_spyre.execution.async_compile.subprocess.run"),
            patch(
                "torch_spyre.execution.async_compile.SpyreSDSCKernelRunner",
                return_value=runner,
            ) as runner_type,
        ):
            result = SpyreAsyncCompile().sdsc("sdsc_fused_mm_0", specs)

        assert result is runner
        descriptor = runner_type.call_args.kwargs["profiler_provenance"]
        assert descriptor.debug_handle_ids == ("9", "12")
        assert extract_kernel_provenance_key(descriptor.event_name) == descriptor.key

    def test_runner_retains_descriptor_for_runtime_forwarding(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(9))])
        assert descriptor is not None

        with patch(
            "torch_spyre.execution.kernel_runner.prepare_kernel",
            return_value="jobplan",
        ):
            runner = SpyreSDSCKernelRunner(
                "sdsc_fused_mm_0",
                "/tmp/kernel",
                profiler_provenance=descriptor,
            )

        assert runner.profiler_provenance is descriptor
        assert runner.jobplan == "jobplan"
