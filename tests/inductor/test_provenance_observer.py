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

"""Device-free unit tests for the provenance forwarding helpers and observer."""

import regex  # noqa: F401  (repo convention: never import re)

from torch_spyre._inductor.provenance import (
    _SPYRE_PROV_CONTEXT_ATTR,
    preserve_provenance,
    merge_provenance,
    decompose_provenance,
)


class _Buf:
    """Minimal ComputedBuffer stand-in: mutable origins/origin_node."""

    def __init__(self, origins=None, origin_node=None, name="buf"):
        self.origins = set(origins or ())
        self.origin_node = origin_node
        self._name = name

    def get_name(self):
        return self._name


class TestPreserveProvenance:
    def test_copies_origins_and_node(self):
        old = _Buf(origins={"a", "b"}, origin_node="a")
        new = _Buf()
        preserve_provenance(old, new)
        assert new.origins == {"a", "b"}
        assert new.origin_node == "a"

    def test_copies_context(self):
        old = _Buf(origins={"a"})
        setattr(old, _SPYRE_PROV_CONTEXT_ATTR, "ctx")
        new = _Buf()
        preserve_provenance(old, new)
        assert getattr(new, _SPYRE_PROV_CONTEXT_ATTR) == "ctx"

    def test_does_not_clobber_existing_origin_node(self):
        # A pass that already set origin_node on the new buffer keeps its value.
        old = _Buf(origin_node="a")
        new = _Buf(origin_node="b")
        preserve_provenance(old, new)
        assert new.origin_node == "b"

    def test_unions_into_existing_origins(self):
        # origins is unioned in place, not rebound: pre-existing origins survive.
        old = _Buf(origins={"a"})
        new = _Buf(origins={"z"})
        preserve_provenance(old, new)
        assert new.origins == {"a", "z"}


class TestMergeProvenance:
    def test_unions_origins_and_sets_context(self):
        s1, s2 = _Buf(origins={"a"}), _Buf(origins={"b", "c"})
        new = _Buf()
        merge_provenance([s1, s2], new, context="spyre_fuse_nodes")
        assert new.origins == {"a", "b", "c"}
        assert getattr(new, _SPYRE_PROV_CONTEXT_ATTR) == "spyre_fuse_nodes"


class TestDecomposeProvenance:
    def test_each_child_inherits_parent(self):
        old = _Buf(origins={"a"}, origin_node="a")
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], context="split_multi_ops")
        for c in (c0, c1):
            assert c.origins == {"a"}
            assert c.origin_node == "a"
            assert getattr(c, _SPYRE_PROV_CONTEXT_ATTR) == "split_multi_ops"

    def test_children_have_independent_origins(self):
        # Each child gets its own origins set; mutating one must not affect another.
        old = _Buf(origins={"a"})
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], context="split_multi_ops")
        c0.origins.add("x")
        assert c1.origins == {"a"}
