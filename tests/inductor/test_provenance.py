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

# Unit tests for the source-to-kernel provenance types (SourceLoc, DebugHandle,
# build_debug_handle). Pure-Python logic, so no torch.compile or Spyre device is
# needed; later tasks use lightweight fakes for Inductor ComputedBuffers / FX
# nodes. Tests are grouped one class per unit and parametrized for data-driven
# cases, matching the repo convention (cf. TestLoopSpecDataclass in test_codegen).

import dataclasses

import pytest

from types import SimpleNamespace

from sympy import Integer, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
from torch_spyre._inductor.codegen.superdsc import SDSCSpec, parse_op_spec
from torch_spyre._inductor.op_spec import DebugHandle, OpSpec, SourceLoc, TensorArg
from torch_spyre._inductor.provenance import _stable_id, build_debug_handle


class TestSourceLoc:
    @pytest.mark.parametrize(
        "loc, expected",
        [
            (SourceLoc("model.py", 117, 8), "model.py:117:8"),
            (SourceLoc("model.py", 117), "model.py:117:0"),  # default col
            (SourceLoc("a/b/m.py", 1, 0, 3, 9), "a/b/m.py:1:0"),  # range -> start
        ],
    )
    def test_to_str_renders_file_line_col(self, loc, expected):
        assert loc.to_str() == expected

    @pytest.mark.parametrize(
        "loc, expected",
        [
            (
                SourceLoc("m.py", 1, 2, 3, 4),
                {
                    "file": "m.py",
                    "start_line": 1,
                    "start_col": 2,
                    "end_line": 3,
                    "end_col": 4,
                },
            ),
            (
                SourceLoc("m.py", 42),  # point: end fields default to None
                {
                    "file": "m.py",
                    "start_line": 42,
                    "start_col": 0,
                    "end_line": None,
                    "end_col": None,
                },
            ),
        ],
    )
    def test_to_dict_roundtrips(self, loc, expected):
        d = loc.to_dict()
        assert d == expected
        assert SourceLoc(**d) == loc

    def test_is_frozen_hashable_and_value_equal(self):
        a = SourceLoc("m.py", 1, 2)
        b = SourceLoc("m.py", 1, 2)
        assert a == b
        # frozen -> hashable: usable in sets / dict keys (DebugHandle relies on this)
        assert len({a, b}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.start_line = 99  # type: ignore[misc]


class TestStableId:
    def test_is_deterministic_and_nonnegative(self):
        s = SourceLoc("m.py", 1)
        a = _stable_id(s, "aten.mm.default", ("n0", "op0"))
        b = _stable_id(s, "aten.mm.default", ("n0", "op0"))
        assert a == b
        assert a >= 0

    @pytest.mark.parametrize(
        "a, b",
        [
            # each component must affect the hash: aten_op, source line, ir_chain
            (
                (SourceLoc("m.py", 1), "aten.mm.default", ("n0",)),
                (SourceLoc("m.py", 1), "aten.add.Tensor", ("n0",)),
            ),
            (
                (SourceLoc("m.py", 1), "aten.mm.default", ("n0",)),
                (SourceLoc("m.py", 2), "aten.mm.default", ("n0",)),
            ),
            (
                (SourceLoc("m.py", 1), "aten.mm.default", ("n0",)),
                (SourceLoc("m.py", 1), "aten.mm.default", ("n1",)),
            ),
        ],
    )
    def test_distinguishes_content(self, a, b):
        assert _stable_id(*a) != _stable_id(*b)


class TestDebugHandle:
    def test_to_dict_is_structured_and_nested(self):
        child = DebugHandle(
            id=1,
            source=SourceLoc("m.py", 5),
            aten_op="aten.permute.default",
            ir_chain=("permute",),
        )
        h = DebugHandle(
            id=2,
            source=SourceLoc("m.py", 5),
            aten_op="aten.mm.default",
            ir_chain=("mm_default_1", "op0"),
            fused_from=(child,),
        )
        d = h.to_dict()
        assert d["source"] == {
            "file": "m.py",
            "start_line": 5,
            "start_col": 0,
            "end_line": None,
            "end_col": None,
        }
        assert d["aten_op"] == "aten.mm.default"
        assert d["ir_chain"] == ["mm_default_1", "op0"]
        assert d["fused_from"][0]["aten_op"] == "aten.permute.default"
        assert d["fusion_context"] is None


def _node(name, file=None, line=None, aten=None):
    """Fake FX node: .name + .meta with stack_trace/original_aten."""
    trace = None
    if file is not None:
        trace = f'  File "{file}", line {line}, in forward\n    x = f(x)\n'
    return SimpleNamespace(
        name=name, meta={"stack_trace": trace, "original_aten": aten}
    )


def _buffer(origins, origin_node=None, name="op0"):
    """Fake ComputedBuffer: .origins, optional .origin_node, .get_name().

    The real ``origins`` is a set of (hashable) fx.Node; our fake nodes are
    ``SimpleNamespace`` (unhashable — it defines ``__eq__``), so we hold them as
    a tuple. ``build_debug_handle`` only iterates/sorts ``origins``, never relies
    on set semantics, so this is faithful for the logic under test.
    """
    return SimpleNamespace(
        origins=tuple(origins), origin_node=origin_node, get_name=lambda: name
    )


class TestBuildDebugHandle:
    def test_empty_origins_returns_none(self):
        assert build_debug_handle(_buffer([])) is None

    def test_single_op_with_origin_node(self):
        # Inductor's clean 1:1 case (e.g. pointwise relu): use origin_node.
        relu = _node("relu", "/home/u/model.py", 42, "aten.relu.default")
        h = build_debug_handle(_buffer([relu], origin_node=relu, name="op2"))
        assert h.source.to_str() == "/home/u/model.py:42:0"
        assert h.aten_op == "aten.relu.default"
        assert h.fused_from == ()  # single origin -> no fusion set

    def test_origin_node_set_without_trace_borrows_source(self):
        # origin_node present but trace-less: aten from it, source from a sibling.
        mm = _node("mm_default_1", aten="aten.mm.default")  # no stack_trace
        permute = _node("permute", "/home/u/model.py", 117, "aten.permute.default")
        h = build_debug_handle(_buffer([mm, permute], origin_node=mm, name="op0"))
        assert h.aten_op == "aten.mm.default"  # from origin_node
        assert h.source.to_str() == "/home/u/model.py:117:0"  # borrowed
        assert "mm_default_1" in h.ir_chain and "op0" in h.ir_chain

    def test_fused_view_no_origin_node_is_ambiguous(self):
        # Realistic fused matmul: Inductor leaves origin_node None (a view fused
        # in). No single primary -> headline aten_op is None; fused_from is truth.
        mm = _node("mm_default_1", aten="aten.mm.default")  # no stack_trace
        permute = _node("permute", "/home/u/model.py", 117, "aten.permute.default")
        h = build_debug_handle(_buffer([mm, permute], name="op0"))  # origin_node=None
        assert h.source.to_str() == "/home/u/model.py:117:0"  # borrowed headline
        assert h.aten_op is None  # two distinct atens -> do not guess
        assert {c.aten_op for c in h.fused_from} == {
            "aten.mm.default",
            "aten.permute.default",
        }

    def test_fused_from_lists_all_origins(self):
        add = _node("add", "/m.py", 10, "aten.add.Tensor")
        relu = _node("relu", "/m.py", 20, "aten.relu.default")
        h = build_debug_handle(_buffer([add, relu], name="op1"))
        assert len(h.fused_from) == 2
        assert {c.source.to_str() for c in h.fused_from} == {
            "/m.py:10:0",
            "/m.py:20:0",
        }
        assert h.aten_op is None  # distinct atens

    def test_skips_torch_internal_frame(self):
        n = _node("x", aten="aten.linear.default")
        n.meta["stack_trace"] = (
            '  File "/usr/lib/torch/_ops.py", line 5, in f\n'
            '  File "/home/u/model.py", line 42, in forward\n'
        )
        h = build_debug_handle(_buffer([n]))
        assert h.source.to_str() == "/home/u/model.py:42:0"


class TestOpSpecDebugHandle:
    def _make(self, **kw):
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={},
            args=[],
            op_info={},
            **kw,
        )

    def test_defaults_to_none(self):
        assert self._make().debug_handle is None

    def test_accepts_debug_handle(self):
        h = DebugHandle(
            id=1,
            source=SourceLoc("m.py", 1),
            aten_op="aten.add.Tensor",
            ir_chain=("add", "op0"),
        )
        spec = self._make(debug_handle=h)
        assert spec.debug_handle is h
        assert spec.debug_handle.aten_op == "aten.add.Tensor"


def _threadable_op_spec(debug_handle=None):
    """Minimal OpSpec that parse_op_spec can process (mirrors test_coarse_tiling)."""
    c0 = Symbol("c0")
    fp16 = DataFormats.SEN169_FP16
    tin = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x1000},
    )
    tout = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x2000},
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={c0: (Integer(128), 1)},
        args=[tin, tout],
        op_info={},
        tiled_symbols=[c0],
        debug_handle=debug_handle,
    )


class TestSDSCSpecDebugHandle:
    def _make(self, **kw):
        return SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=DataFormats.SEN169_FP16,
            num_inputs=1,
            iteration_space={},
            num_cores=1,
            work_slices={},
            core_id_to_work_slice={},
            padding={},
            layouts={},
            args=[],
            constants={},
            coordinate_masking={},
            **kw,
        )

    def test_defaults_to_none(self):
        assert self._make().debug_handle is None

    def test_accepts_debug_handle(self):
        h = DebugHandle(
            id=1,
            source=SourceLoc("m.py", 1),
            aten_op="aten.add.Tensor",
            ir_chain=("add", "op0"),
        )
        assert self._make(debug_handle=h).debug_handle is h

    def test_parse_op_spec_threads_handle(self):
        # parse_op_spec builds the SDSCSpec from an OpSpec; the handle must
        # survive that translation (the OpSpec -> SDSCSpec threading).
        h = DebugHandle(
            id=7,
            source=SourceLoc("model.py", 5),
            aten_op="aten.add.Tensor",
            ir_chain=("add", "op0"),
        )
        sdsc_spec, _ = parse_op_spec(_threadable_op_spec(debug_handle=h))
        assert sdsc_spec.debug_handle is h


class TestGenerateSdscEmit:
    def test_emits_debug_handle(self):
        # Full OpSpec -> parse_op_spec -> SDSCSpec -> generate_sdsc -> JSON chain.
        h = DebugHandle(
            id=7,
            source=SourceLoc("model.py", 5),
            aten_op="aten.add.Tensor",
            ir_chain=("add", "op0"),
        )
        sdsc_spec, _ = parse_op_spec(_threadable_op_spec(debug_handle=h))
        sdsc_json, *_ = generate_sdsc(0, sdsc_spec, [])
        assert sdsc_json[f"0_{sdsc_spec.opfunc}"]["debug_handle_"] == h.to_dict()

    def test_emits_null_when_absent(self):
        sdsc_spec, _ = parse_op_spec(_threadable_op_spec())
        sdsc_json, *_ = generate_sdsc(0, sdsc_spec, [])
        assert sdsc_json[f"0_{sdsc_spec.opfunc}"]["debug_handle_"] is None
