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

from torch_spyre._inductor.op_spec import SourceLoc, DebugHandle
from torch_spyre._inductor.provenance import _stable_id


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
