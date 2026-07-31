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

"""Device-free tests for kernel provenance identity and event transport."""

import copy
import ctypes
import dataclasses
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import types
from unittest.mock import patch

from jsonschema import ValidationError
from jsonschema.validators import validator_for
import pytest
import regex as re

import torch  # noqa: F401
from sympy import Integer, Symbol, sympify
from torch._inductor.utils import IndentedBuffer

from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    extract_kernel_provenance_key as extract_kernel_provenance_key_cpp,
)
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    IndirectAccess,
    LoopSpec,
    OpSpec,
    SourceLoc,
    TensorArg,
)
from torch_spyre._inductor.kernel_provenance import (
    build_kernel_provenance_descriptor,
    KernelProvenanceDescriptor,
    KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
    KERNEL_PROVENANCE_KEY_VERSION,
)
from torch_spyre._inductor.profiler_event import (
    extract_kernel_provenance_key,
    format_kernel_provenance_event_name,
)
from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list
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


def _op(
    handle: DebugHandle | None,
    *,
    op: str = "identity",
    iteration_space=None,
    args=(),
    op_info=None,
    tiled_symbols=None,
    tiled_symbol_trip_counts=None,
    symbolic_dim_bounds=None,
    node_output_ranges=None,
) -> OpSpec:
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={} if iteration_space is None else iteration_space,
        args=list(args),
        op_info={} if op_info is None else op_info,
        tiled_symbols=[] if tiled_symbols is None else tiled_symbols,
        tiled_symbol_trip_counts=(
            {} if tiled_symbol_trip_counts is None else tiled_symbol_trip_counts
        ),
        symbolic_dim_bounds=(
            {} if symbolic_dim_bounds is None else symbolic_dim_bounds
        ),
        node_output_ranges=node_output_ranges,
        debug_handle=handle,
    )


def _event_name(
    descriptor: KernelProvenanceDescriptor,
) -> str:
    return format_kernel_provenance_event_name(descriptor)


def _generated_wrapper_roundtrip(specs):
    """Serialize and reconstruct specs through the generated-wrapper seam."""

    def sympy_str(value):
        if isinstance(value, IndirectAccess):
            name_sym = value.args[0]
            return f"IndirectAccess('{name_sym}')"
        return "sympify('" + str(value) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(specs, buf, sympy_str)
    buf.writeline("]")
    namespace = {
        "DataFormats": DataFormats,
        "DebugHandle": DebugHandle,
        "ElementArrangement": ElementArrangement,
        "IndirectAccess": IndirectAccess,
        "LoopSpec": LoopSpec,
        "OpSpec": OpSpec,
        "SourceLoc": SourceLoc,
        "TensorArg": TensorArg,
        "sympify": sympify,
    }
    return eval(buf.getvalue(), namespace)  # noqa: S307


class TestKernelProvenanceDescriptor:
    def test_builds_bundle_identity_without_handles(self):
        specs = [
            _op(None),
            LoopSpec(count=Integer(2), body=[_op(None)]),
        ]

        descriptor = build_kernel_provenance_descriptor(specs)

        assert descriptor.debug_handle_ids == ()
        assert descriptor.aten_ops == ()
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_unknown_{descriptor.key}"
        )

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
        assert descriptor.aten_ops == ("aten.add.Tensor", "aten.mm.default")
        assert extract_kernel_provenance_key(_event_name(descriptor)) == descriptor.key
        assert not hasattr(descriptor, "fusion_context")

    def test_is_deterministic_and_order_sensitive(self):
        first = _handle(9)
        second = _handle(12)
        forward = build_kernel_provenance_descriptor([_op(first), _op(second)])
        independently_reconstructed = build_kernel_provenance_descriptor(
            [_op(_handle(9)), _op(_handle(12))]
        )
        reverse = build_kernel_provenance_descriptor([_op(second), _op(first)])

        assert independently_reconstructed == forward
        assert forward.key == "vsancadvtjfcq6cv"
        assert reverse.key != forward.key
        assert len(forward.key) == KERNEL_PROVENANCE_KEY_BASE32_WIDTH
        assert set(forward.key) <= set("abcdefghijklmnopqrstuvwxyz234567")

    def test_distinguishes_structures_with_the_same_handle_set(self):
        handle = _handle(9)
        flat = build_kernel_provenance_descriptor([_op(handle)])
        different_op = build_kernel_provenance_descriptor([_op(handle, op="add")])
        looped = build_kernel_provenance_descriptor(
            [LoopSpec(count=Integer(2), body=[_op(handle)])]
        )
        repeated_op = build_kernel_provenance_descriptor([_op(handle), _op(handle)])

        descriptors = (flat, different_op, looped, repeated_op)
        assert all(descriptor is not None for descriptor in descriptors)
        assert len({descriptor.key for descriptor in descriptors if descriptor}) == 4
        assert all(
            descriptor.debug_handle_ids == ("9",)
            for descriptor in descriptors
            if descriptor
        )

    def test_canonicalizes_real_tensor_args_and_unordered_metadata(self):
        handle = _handle(9)
        c0 = Symbol("c0")
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0},
            name="arg0",
        )
        first = _op(
            handle,
            iteration_space={c0: (Integer(128), 1)},
            args=(arg,),
            op_info={"constants": {"alpha": 1.0, "beta": Integer(2)}},
        )
        reordered_metadata = _op(
            handle,
            iteration_space={c0: (Integer(128), 1)},
            args=(arg,),
            op_info={"constants": {"beta": Integer(2), "alpha": 1.0}},
        )
        changed_shape = dataclasses.replace(
            first,
            args=[dataclasses.replace(arg, device_size=[4, 64])],
        )
        changed_arrangement = dataclasses.replace(
            first,
            args=[
                dataclasses.replace(arg, element_arrangement=ElementArrangement.EXX2)
            ],
        )

        first_descriptor = build_kernel_provenance_descriptor([first])
        reordered_descriptor = build_kernel_provenance_descriptor([reordered_metadata])
        changed_descriptor = build_kernel_provenance_descriptor([changed_shape])
        changed_arrangement_descriptor = build_kernel_provenance_descriptor(
            [changed_arrangement]
        )

        assert first_descriptor is not None
        assert reordered_descriptor is not None
        assert changed_descriptor is not None
        assert changed_arrangement_descriptor is not None
        assert reordered_descriptor.key == first_descriptor.key
        assert changed_descriptor.key != first_descriptor.key
        assert changed_arrangement_descriptor.key != first_descriptor.key

    def test_pins_rich_canonical_bundle_key(self):
        c0 = Symbol("c0")
        index = Symbol("index")
        constituent = _handle(8, aten_op="aten.permute.default")
        handle = _handle(9, fused_from=(constituent,))
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[IndirectAccess(index), c0],
            allocation={"hbm_pool": {"offset": 4096}},
            element_arrangement=ElementArrangement.EXX2,
            name="arg0",
            device_tile_advance_expr=64 * c0,
        )
        spec = _op(
            handle,
            op="add",
            iteration_space={c0: (Integer(128), 2)},
            args=(arg,),
            op_info={
                "alpha": 1.5,
                "mask": b"\x00\xff",
                "expression": c0 + 1,
            },
            tiled_symbols=[[c0]],
            tiled_symbol_trip_counts={c0: 4},
            symbolic_dim_bounds={"s0": (128, 64)},
            node_output_ranges=(Integer(1), Integer(2), c0, Integer(64)),
        )

        descriptor = build_kernel_provenance_descriptor(
            [LoopSpec(count=Integer(4), body=[spec])]
        )

        assert descriptor.debug_handle_ids == ("9",)
        assert descriptor.aten_ops == (
            "aten.mm.default",
            "aten.permute.default",
        )
        assert descriptor.key == "2y3kqixggcejkcas"

    def test_generated_wrapper_roundtrip_reproduces_descriptor(self):
        constituent = _handle(8, aten_op="aten.permute.default")
        handle = _handle(9, fused_from=(constituent,))
        c0 = Symbol("c0")
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0},
            element_arrangement=ElementArrangement.DL16_TO_FP32,
            name="arg0",
        )
        specs = [
            LoopSpec(
                count=Integer(2),
                body=[
                    _op(
                        handle,
                        op="add",
                        iteration_space={c0: (Integer(128), 1)},
                        args=(arg,),
                        op_info={"constants": {"alpha": 1.0}},
                        tiled_symbols=[[c0]],
                        tiled_symbol_trip_counts={c0: 2},
                        symbolic_dim_bounds={"s0": (128, 64)},
                        node_output_ranges=(Integer(1), Integer(2), c0),
                    )
                ],
            )
        ]

        original = build_kernel_provenance_descriptor(specs)
        reconstructed = build_kernel_provenance_descriptor(
            _generated_wrapper_roundtrip(specs)
        )

        assert original is not None
        assert reconstructed == original

    @pytest.mark.parametrize("changed_schema", [OpSpec, TensorArg, LoopSpec])
    def test_rejects_finalized_schema_drift(self, changed_schema):
        real_fields = dataclasses.fields

        def fields_with_future_field(schema):
            fields = real_fields(schema)
            if schema is changed_schema:
                return (
                    *fields,
                    types.SimpleNamespace(name="future_field", type="object"),
                )
            return fields

        with (
            patch(
                "torch_spyre._inductor.kernel_provenance.dataclasses.fields",
                side_effect=fields_with_future_field,
            ),
            pytest.raises(
                TypeError,
                match=rf"{changed_schema.__name__} schema changed.*future_field",
            ),
        ):
            build_kernel_provenance_descriptor([_op(_handle(1))])

    @pytest.mark.parametrize(
        ("changed_schema", "field_name"),
        [
            (OpSpec, "iteration_space"),
            (TensorArg, "device_coordinates"),
            (LoopSpec, "body"),
        ],
    )
    def test_rejects_finalized_schema_type_drift(self, changed_schema, field_name):
        real_fields = dataclasses.fields

        def fields_with_changed_type(schema):
            fields = real_fields(schema)
            if schema is not changed_schema:
                return fields
            return tuple(
                types.SimpleNamespace(
                    name=field.name,
                    type="future_type" if field.name == field_name else field.type,
                )
                for field in fields
            )

        with (
            patch(
                "torch_spyre._inductor.kernel_provenance.dataclasses.fields",
                side_effect=fields_with_changed_type,
            ),
            pytest.raises(
                TypeError,
                match=rf"{changed_schema.__name__} schema changed.*{field_name}",
            ),
        ):
            build_kernel_provenance_descriptor([_op(_handle(1))])

    def test_descriptor_is_frozen(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(1))])

        assert descriptor is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            descriptor.key = "a" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH  # type: ignore[misc]


class TestKernelProvenanceEventName:
    def test_uses_aten_only_display_without_source_location(self):
        handle = _handle(
            42,
            source=SourceLoc("/private/workspace/model.py", 117),
        )

        descriptor = build_kernel_provenance_descriptor([_op(handle), _op(handle)])

        assert descriptor is not None
        assert (
            _event_name(descriptor)
            == f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_mm_{descriptor.key}"
        )
        assert "model" not in _event_name(descriptor)
        assert "117" not in _event_name(descriptor)

    def test_summarizes_conflicting_handles_without_choosing_a_primary(self):
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
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_add_mm_{descriptor.key}"
        )

    def test_uses_recursive_fused_constituents_for_display_only(self):
        constituent = _handle(1, source=SourceLoc("first.py", 10))
        fused = _handle(
            2,
            aten_op=None,
            source=None,
            fused_from=(constituent,),
        )

        descriptor = build_kernel_provenance_descriptor([_op(fused)])

        assert descriptor is not None
        assert descriptor.debug_handle_ids == ("2",)
        assert descriptor.aten_ops == ("aten.mm.default",)
        assert (
            _event_name(descriptor)
            == f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_mm_{descriptor.key}"
        )

    def test_uses_unknown_label_when_no_aten_name_exists(self):
        descriptor = build_kernel_provenance_descriptor(
            [_op(_handle(1, aten_op=None, source=None))]
        )

        assert descriptor is not None
        assert _event_name(descriptor) == (
            f"spyre_kernel_v{KERNEL_PROVENANCE_KEY_VERSION}_fused_unknown_{descriptor.key}"
        )

    def test_sanitizes_and_bounds_name_with_step_suffix_reservation(self):
        long_component = "α/" + "very-long-name." * 20
        handle = _handle(
            7,
            aten_op=f"aten.{long_component}.default",
            source=SourceLoc(f"/tmp/{long_component}.py", 123456),
        )

        descriptor = build_kernel_provenance_descriptor([_op(handle)])

        assert descriptor is not None
        assert _event_name(descriptor).isascii()
        # PR #2930 uses size_t for this JobPlan command index. Match the local
        # extension ABI instead of assuming that every target uses 64-bit size_t.
        size_t_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        largest_step_suffix = f"#{(1 << size_t_bits) - 1}"
        final_name = f"{_event_name(descriptor)}{largest_step_suffix}"
        assert len(final_name.encode("ascii")) <= 127
        assert descriptor.key in final_name

    @pytest.mark.parametrize(
        "key",
        [
            "a" * (KERNEL_PROVENANCE_KEY_BASE32_WIDTH - 1),
            "a" * (KERNEL_PROVENANCE_KEY_BASE32_WIDTH + 1),
            "A" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
            "1" * KERNEL_PROVENANCE_KEY_BASE32_WIDTH,
        ],
    )
    def test_rejects_noncanonical_descriptor_key(self, key):
        with pytest.raises(ValueError, match="not canonical lowercase base32"):
            KernelProvenanceDescriptor(key, (), ())

    def test_cpp_parser_literals_match_python_constants(self):
        """The C++ parser hardcodes these; see kernel_provenance_registry.cpp.

        A constant bump must update both sides in the same change.
        """
        assert KERNEL_PROVENANCE_KEY_VERSION == 1
        assert KERNEL_PROVENANCE_KEY_BASE32_WIDTH == 16

    @pytest.mark.parametrize(
        ("event_name", "expected"),
        [
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa", "a" * 16),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa#17", "a" * 16),
            ("sdsc_mm_0", None),
            ("spyre_kernel_fused_mm_aaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v2_fused_mm_aaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mm_short", None),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mmaaaaaaaaaaaaaaaa", None),
            ("spyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa#step", None),
            ("xspyre_kernel_v1_fused_mm_aaaaaaaaaaaaaaaa", None),
        ],
    )
    def test_python_and_cpp_key_parsers_share_contract(self, event_name, expected):
        assert extract_kernel_provenance_key(event_name) == expected
        assert extract_kernel_provenance_key_cpp(event_name) == expected


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
        descriptor = runner_type.call_args.kwargs["kernel_provenance"]
        assert descriptor.debug_handle_ids == ("9", "12")
        assert extract_kernel_provenance_key(_event_name(descriptor)) == descriptor.key

    def test_async_compile_keeps_execution_on_unknown_bundle_value(self):
        specs = [_op(_handle(9), op_info={"future_value": object()})]
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
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            result = SpyreAsyncCompile().sdsc("sdsc_fused_mm_0", specs)

        assert result is runner
        assert runner_type.call_args.kwargs["kernel_provenance"] is None
        warning.assert_called_once()
        assert "continuing without kernel provenance" in warning.call_args.args[0]

    def test_async_compile_aggregates_provenance_failures(self):
        specs = [_op(_handle(9), op_info={"future_value": object()})]
        runner = object()
        compiler = SpyreAsyncCompile()

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
            ),
            patch("torch_spyre.execution.async_compile.AsyncCompile.wait") as base_wait,
            patch("torch_spyre.execution.async_compile.logger.warning") as warning,
        ):
            assert compiler.sdsc("sdsc_fused_mm_0", specs) is runner
            assert compiler.sdsc("sdsc_fused_mm_1", specs) is runner
            compiler.wait({})

        base_wait.assert_called_once_with({})
        assert warning.call_count == 2
        assert warning.call_args_list[0].kwargs["exc_info"] is True
        assert warning.call_args_list[1].args == (
            "kernel provenance disabled for %d/%d compiled Spyre kernels",
            2,
            2,
        )
        assert compiler._provenance_attempt_count == 0
        assert compiler._provenance_failure_count == 0

    def test_runner_retains_descriptor_for_runtime_forwarding(self):
        descriptor = build_kernel_provenance_descriptor([_op(_handle(9))])
        assert descriptor is not None

        with (
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance",
                return_value=True,
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
        ):
            runner = SpyreSDSCKernelRunner(
                "sdsc_fused_mm_0",
                "/tmp/kernel",
                kernel_provenance=descriptor,
            )

        assert runner.kernel_provenance is descriptor
        assert runner.profiler_event_name == _event_name(descriptor)
        assert runner.jobplan == "jobplan"
        register_kernel_provenance.assert_called_once_with(
            _event_name(descriptor), list(descriptor.debug_handle_ids)
        )
        prepare_kernel.assert_called_once_with(
            "/tmp/kernel/spyreCodeDir",
            profiler_name=_event_name(descriptor),
        )

    def test_runner_preserves_legacy_prepare_call_without_descriptor(self):
        with (
            patch(
                "torch_spyre.execution.kernel_runner.register_kernel_provenance"
            ) as register_kernel_provenance,
            patch(
                "torch_spyre.execution.kernel_runner.prepare_kernel",
                return_value="jobplan",
            ) as prepare_kernel,
        ):
            runner = SpyreSDSCKernelRunner("sdsc_fused_mm_0", "/tmp/kernel")

        assert runner.kernel_provenance is None
        assert runner.profiler_event_name is None
        prepare_kernel.assert_called_once_with("/tmp/kernel/spyreCodeDir")
        register_kernel_provenance.assert_not_called()


_SCHEMA = json.loads(
    files("torch_spyre._inductor.schemas")
    .joinpath("spyre_provenance_v1.schema.json")
    .read_text(encoding="utf-8")
)
_SCHEMA_VALIDATOR_TYPE = validator_for(_SCHEMA)
_SCHEMA_VALIDATOR_TYPE.check_schema(_SCHEMA)
_SCHEMA_VALIDATOR = _SCHEMA_VALIDATOR_TYPE(_SCHEMA)
_PROVENANCE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "provenance"


def _load_provenance_fixture(name: str) -> dict:
    return json.loads((_PROVENANCE_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _fixture_compile_id(projection: dict) -> str:
    kernels = [
        [kernel["compilerKernelName"], kernel["identityKey"]]
        for kernel in projection["kernels"]
    ]
    return _canonical_digest(
        {
            "domain": "torch-spyre-compile-v1",
            "kernels": kernels,
        }
    )


def _fixture_occurrence_id(occurrence: dict) -> str:
    return _canonical_digest(
        {
            "domain": "torch-spyre-occurrence-v1",
            "compileId": occurrence["compileId"],
            "compilerKernelName": occurrence["compilerKernelName"],
            "identityKey": occurrence["identityKey"],
        }
    )


def _recursive_fixture_atens(handle_ids: list[str], handles: dict) -> list[str]:
    names: set[str] = set()
    visited: set[str] = set()

    def visit(handle_id: str) -> None:
        assert handle_id in handles
        if handle_id in visited:
            return
        visited.add(handle_id)
        handle = handles[handle_id]
        if handle["aten_op"] is not None:
            names.add(handle["aten_op"])
        for constituent_id in handle["fused_from"]:
            visit(constituent_id)

    for handle_id in handle_ids:
        visit(handle_id)
    return sorted(names)


def _deduplicate_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _assert_sorted_keys(mapping: dict) -> None:
    assert list(mapping) == sorted(mapping)


def _validate_fixture_semantics(document: dict) -> None:
    handles = document["handles"]
    identities = document["kernelIdentities"]
    occurrences = document["kernelOccurrences"]
    projections = document["upstreamProjections"]

    for mapping in (
        document["diagnostics"],
        handles,
        identities,
        occurrences,
        projections,
    ):
        _assert_sorted_keys(mapping)

    for handle_id, handle in handles.items():
        assert handle["id"] == handle_id
        assert all(child_id in handles for child_id in handle["fused_from"])

    for identity_key, identity in identities.items():
        binding_ids = [
            binding["handleId"] for binding in identity["specHandleBindings"]
        ]
        assert all(handle_id in handles for handle_id in binding_ids)
        assert identity["directHandleIds"] == _deduplicate_in_order(binding_ids)
        assert identity["atenOps"] == _recursive_fixture_atens(
            identity["directHandleIds"], handles
        )
        descriptor = KernelProvenanceDescriptor(
            key=identity_key,
            debug_handle_ids=tuple(identity["directHandleIds"]),
            aten_ops=tuple(identity["atenOps"]),
        )
        assert (
            format_kernel_provenance_event_name(descriptor) == identity["eventNameBase"]
        )

    occurrences_by_compile: dict[str, list[tuple[str, dict]]] = {}
    for occurrence_id, occurrence in occurrences.items():
        assert occurrence_id == _fixture_occurrence_id(occurrence)
        assert occurrence["identityKey"] in identities
        assert occurrence["compileId"] in projections
        ordinals = [
            registration["ordinal"] for registration in occurrence["registrations"]
        ]
        assert ordinals == sorted(ordinals)
        assert len(ordinals) == len(set(ordinals))
        for registration in occurrence["registrations"]:
            assert registration["alias"] == (
                f"{occurrence['compilerKernelName']}:{registration['ordinal']}"
            )
        occurrences_by_compile.setdefault(occurrence["compileId"], []).append(
            (occurrence_id, occurrence)
        )

    for compile_id, projection in projections.items():
        assert compile_id == _fixture_compile_id(projection)
        kernels = [
            (kernel["compilerKernelName"], kernel["identityKey"])
            for kernel in projection["kernels"]
        ]
        assert len(kernels) == len(set(kernels))
        assert all(identity_key in identities for _, identity_key in kernels)

        compile_occurrences = occurrences_by_compile.get(compile_id, [])
        occurrence_kernels = {
            (
                occurrence["compilerKernelName"],
                occurrence["identityKey"],
            )
            for _, occurrence in compile_occurrences
        }
        assert occurrence_kernels == set(kernels)

        registrations = [
            registration
            for _, occurrence in compile_occurrences
            for registration in occurrence["registrations"]
        ]
        ordinals = [registration["ordinal"] for registration in registrations]
        assert len(ordinals) == len(set(ordinals))
        registration_aliases = {registration["alias"] for registration in registrations}

        for relation_name in (
            "preToPost",
            "postToPre",
            "cppCodeToPost",
            "postToCppCode",
            "kernelStackTraces",
        ):
            _assert_sorted_keys(projection[relation_name])

        cpp_to_post = projection["cppCodeToPost"]
        assert set(cpp_to_post) <= registration_aliases
        if projection["upstreamJoin"] == "ok":
            assert set(cpp_to_post) == registration_aliases
            assert set(projection["kernelStackTraces"]) == registration_aliases

        expected_post_to_cpp: dict[str, list[str]] = {}
        for alias in sorted(cpp_to_post):
            for post_node in cpp_to_post[alias]:
                expected_post_to_cpp.setdefault(post_node, []).append(alias)
        assert projection["postToCppCode"] == expected_post_to_cpp

        expected_post_to_pre: dict[str, list[str]] = {}
        for pre_node, post_nodes in projection["preToPost"].items():
            for post_node in post_nodes:
                expected_post_to_pre.setdefault(post_node, []).append(pre_node)
        assert projection["postToPre"] == expected_post_to_pre

    assert set(occurrences_by_compile) == set(projections)
    expected_status = (
        "complete"
        if not document["diagnostics"]
        and all(
            projection["upstreamJoin"] == "ok" for projection in projections.values()
        )
        else "partial"
    )
    assert document["status"] == expected_status


def _apply_fixture_mutation(document: dict, mutation: dict) -> dict:
    mutated = copy.deepcopy(document)
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in mutation["path"].lstrip("/").split("/")
    ]
    parent: object = mutated
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    final = tokens[-1]
    if isinstance(parent, list):
        parent[int(final)] = mutation["value"]
    else:
        assert isinstance(parent, dict)
        parent[final] = mutation["value"]
    return mutated


def _fixture_reader_diagnostic(document: dict) -> str | None:
    if document.get("schemaVersion") != _SCHEMA["properties"]["schemaVersion"]["const"]:
        return "unsupported-schema-version"
    try:
        _SCHEMA_VALIDATOR.validate(document)
    except ValidationError:
        return "schema-validation-failure"
    try:
        _validate_fixture_semantics(document)
    except AssertionError:
        return "semantic-validation-failure"
    return None


class TestProvenanceArtifactSchema:
    def test_valid_fixtures_pass_schema_and_semantic_validation(self):
        manifest = _load_provenance_fixture("fixture_manifest.json")

        for fixture in manifest["valid"]:
            document = _load_provenance_fixture(fixture["path"])
            _SCHEMA_VALIDATOR.validate(document)
            _validate_fixture_semantics(document)
            assert _fixture_reader_diagnostic(document) is None

    def test_invalid_fixture_mutations_report_declared_reader_diagnostic(self):
        manifest = _load_provenance_fixture("fixture_manifest.json")

        for fixture in manifest["invalid"]:
            document = _load_provenance_fixture(fixture["base"])
            mutated = _apply_fixture_mutation(document, fixture["mutation"])
            assert (
                _fixture_reader_diagnostic(mutated)
                == fixture["expectedReaderDiagnostic"]
            ), fixture["name"]

    def test_schema_event_contract_matches_production_codec(self):
        event_key_schema = _SCHEMA["properties"]["eventKey"]["$ref"]
        assert event_key_schema == "#/$defs/eventKey"
        assert (
            _SCHEMA["$defs"]["eventKey"]["properties"]["version"]["const"]
            == KERNEL_PROVENANCE_KEY_VERSION
        )
        assert (
            _SCHEMA["$defs"]["eventKey"]["properties"]["width"]["const"]
            == KERNEL_PROVENANCE_KEY_BASE32_WIDTH
        )
        assert _SCHEMA["$defs"]["eventKeyValue"]["pattern"] == (
            rf"^[a-z2-7]{{{KERNEL_PROVENANCE_KEY_BASE32_WIDTH}}}$"
        )

        document = _load_provenance_fixture("valid_v1.json")
        event_pattern = _SCHEMA["$defs"]["kernelIdentity"]["properties"][
            "eventNameBase"
        ]["pattern"]
        assert all(
            re.fullmatch(event_pattern, identity["eventNameBase"])
            for identity in document["kernelIdentities"].values()
        )

    def test_cache_replay_reuses_compile_and_occurrence_ids(self):
        fresh = _load_provenance_fixture("valid_v1.json")
        cached = _load_provenance_fixture("valid_cache_replay_v1.json")
        compile_id = "3acd1120e2323fa1fec28f1ff89a59da68cda82541b0a19815ec47a2c922ac0e"

        assert compile_id in fresh["upstreamProjections"]
        assert compile_id in cached["upstreamProjections"]
        cached_occurrence_ids = set(cached["kernelOccurrences"])
        assert cached_occurrence_ids <= set(fresh["kernelOccurrences"])
        for occurrence_id in cached_occurrence_ids:
            cached_occurrence = cached["kernelOccurrences"][occurrence_id]
            fresh_occurrence = fresh["kernelOccurrences"][occurrence_id]
            assert cached_occurrence["compileId"] == fresh_occurrence["compileId"]
            assert cached_occurrence["identityKey"] == fresh_occurrence["identityKey"]
            assert (
                cached_occurrence["compilerKernelName"]
                == (fresh_occurrence["compilerKernelName"])
            )
            assert cached_occurrence["registrations"] == []

        assert fresh["upstreamProjections"][compile_id]["upstreamJoin"] == "ok"
        assert cached["upstreamProjections"][compile_id]["upstreamJoin"] == (
            "unavailable-cache-replay"
        )

    def test_reserved_occurrence_selector_is_an_opaque_exact_match_token(self):
        document = _load_provenance_fixture("valid_cache_replay_v1.json")
        occurrence = next(iter(document["kernelOccurrences"].values()))
        occurrence["selector"] = "future-event-occurrence-token"
        _SCHEMA_VALIDATOR.validate(document)
        _validate_fixture_semantics(document)
