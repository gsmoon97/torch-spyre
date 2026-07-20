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

import logging
import logging.handlers

import pytest
import regex  # noqa: F401  (repo convention: never import re)

from torch_spyre._inductor.loop_info import copy_op_metadata
from torch_spyre._inductor.op_spec import ProvenanceTransform
from torch_spyre._inductor.provenance import (
    _SPYRE_PROV_HISTORY_ATTR,
    preserve_provenance,
    merge_provenance,
    decompose_provenance,
    SpyreGraphTransformObserver,
    reset_provenance_warnings,
)


@pytest.fixture
def prov_logs():
    # Capture WARNING records emitted by the provenance observer.
    reset_provenance_warnings()
    logger = logging.getLogger("spyre.inductor.provenance")
    handler = logging.handlers.MemoryHandler(capacity=10000)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield handler.buffer
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


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

    def test_copies_history(self):
        old = _Buf(origins={"a"})
        history = (ProvenanceTransform("fusion", "fuse"),)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, history)
        new = _Buf()
        preserve_provenance(old, new)
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == history

    def test_does_not_clobber_existing_origin_node(self):
        # A pass that already set origin_node on the new buffer keeps its value.
        old = _Buf(origin_node="a")
        new = _Buf(origin_node="b")
        preserve_provenance(old, new)
        assert new.origin_node == "b"

    def test_combines_existing_histories_without_clobbering(self):
        old = _Buf()
        new = _Buf()
        old_transform = ProvenanceTransform("fusion", "old_fusion")
        new_transform = ProvenanceTransform("rewrite", "new_rewrite")
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, (old_transform,))
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, (new_transform,))

        preserve_provenance(old, new)

        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == (
            old_transform,
            new_transform,
        )

    def test_preserves_legitimate_repeated_records(self):
        old = _Buf()
        new = _Buf()
        repeated = ProvenanceTransform("rewrite", "same_pass")
        history = (repeated, repeated)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, history)
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, history)
        preserve_provenance(old, new)
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == history

    def test_unions_into_existing_origins(self):
        # origins is unioned in place, not rebound: pre-existing origins survive.
        old = _Buf(origins={"a"})
        new = _Buf(origins={"z"})
        preserve_provenance(old, new)
        assert new.origins == {"a", "z"}


class TestCopyOpMetadata:
    def test_does_not_copy_provenance_history(self):
        old = _Buf()
        source_history = (ProvenanceTransform("fusion", "source_fusion"),)
        destination_history = (ProvenanceTransform("rewrite", "destination"),)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, source_history)
        new = _Buf()
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, destination_history)

        copy_op_metadata(old, new)

        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == destination_history


class TestMergeProvenance:
    def test_unions_origins_and_appends_fusion_record(self):
        s1, s2 = _Buf(origins={"a"}), _Buf(origins={"b", "c"})
        new = _Buf()
        merge_provenance(
            [s1, s2],
            new,
            pass_name="spyre_fuse_nodes",
            reason="same tile",
        )
        assert new.origins == {"a", "b", "c"}
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "fusion", "spyre_fuse_nodes", "same tile"
        )

    def test_clears_single_source_origin_node(self):
        s1 = _Buf(origins={"a"}, origin_node="a")
        s2 = _Buf(origins={"b"}, origin_node="b")
        new = _Buf(origin_node="a")
        merge_provenance([s1, s2], new, pass_name="spyre_fuse_nodes")
        assert new.origin_node is None


class TestDecomposeProvenance:
    def test_each_child_inherits_parent(self):
        old = _Buf(origins={"a"}, origin_node="a")
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], pass_name="split_multi_ops")
        for c in (c0, c1):
            assert c.origins == {"a"}
            assert c.origin_node == "a"
            assert getattr(c, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
                "decomposition", "split_multi_ops"
            )

    def test_children_have_independent_origins(self):
        # Each child gets its own origins set; mutating one must not affect another.
        old = _Buf(origins={"a"})
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], pass_name="split_multi_ops")
        c0.origins.add("x")
        assert c1.origins == {"a"}


class _NodeListTarget(list):
    """Stand-in for list[BaseSchedulerNode]; each unit's buffer is .node."""


def _unit(buf):
    class _N:
        def __init__(self, b):
            self.node = b

    return _N(buf)


class TestObserverDetection:
    def test_regression_warns(self, prov_logs):
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "bad_pass_regress", kind="node"):
            b.origins = set()  # pass wrongly drops provenance
        assert any("bad_pass_regress" in r.getMessage() for r in prov_logs)

    def test_preserved_no_warning(self, prov_logs):
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "good_pass_keep", kind="node"):
            pass  # no change
        assert not any("good_pass_keep" in r.getMessage() for r in prov_logs)

    def test_new_unattributed_buffer_warns(self, prov_logs):
        target = _NodeListTarget([])
        with SpyreGraphTransformObserver(target, "creates_bare_buf", kind="node"):
            target.append(_unit(_Buf(origins=set())))
        assert any("creates_bare_buf" in r.getMessage() for r in prov_logs)

    def test_allowlisted_new_buffer_silent(self, prov_logs):
        target = _NodeListTarget([])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            target.append(_unit(_Buf(origins=set())))  # source-less by design
        assert not any("spyre-provenance" in r.getMessage() for r in prov_logs)

    def test_partial_origin_loss_warns(self, prov_logs):
        b = _Buf(origins={"a", "b"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "partial_drop_pass", kind="node"):
            b.origins = {"a"}  # loses "b" but is not empty
        assert any("partial_drop_pass" in r.getMessage() for r in prov_logs)

    def test_sourceless_creation_pass_partial_loss_warns(self, prov_logs):
        b = _Buf(origins={"a", "b"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            b.origins = {"a"}
        assert any("insert_restickify" in r.getMessage() for r in prov_logs)

    def test_disabled_by_env(self, prov_logs, monkeypatch):
        monkeypatch.setenv("TORCH_SPYRE_PROVENANCE", "0")
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "bad_pass_env", kind="node"):
            b.origins = set()
        assert not any("spyre-provenance" in r.getMessage() for r in prov_logs)

    def test_never_raises_on_bad_target(self):
        # Enumeration failure must not propagate.
        with SpyreGraphTransformObserver(object(), "weird_target", kind="node"):
            pass

    def test_replacement_buffer_partial_loss_warns(self, prov_logs):
        # A pass that swaps a buffer for a fresh SAME-NAME object holding a
        # subset of origins ({a,b} -> {a}) must still be flagged; identity-keyed
        # snapshots would miss it because the object id changed.
        target = _NodeListTarget([_unit(_Buf(origins={"a", "b"}, name="x"))])
        with SpyreGraphTransformObserver(target, "replace_pass", kind="node"):
            target[0] = _unit(_Buf(origins={"a"}, name="x"))  # same name, lost "b"
        assert any("replace_pass" in r.getMessage() for r in prov_logs)

    def test_origin_node_loss_warns(self, prov_logs):
        # A pass that keeps origins but clears origin_node still loses the
        # authoritative source/aten pointer; a non-allowlisted pass must warn.
        b = _Buf(origins={"a"}, origin_node="a")
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "drops_origin_node", kind="node"):
            b.origin_node = None  # origins intact, authoritative pointer gone
        assert any("drops_origin_node" in r.getMessage() for r in prov_logs)

    def test_transform_history_loss_warns(self, prov_logs):
        b = _Buf(origins={"a"})
        setattr(b, _SPYRE_PROV_HISTORY_ATTR, (ProvenanceTransform("fusion", "fuse"),))
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "drops_history", kind="node"):
            setattr(b, _SPYRE_PROV_HISTORY_ATTR, ())
        assert any("transformation history" in r.getMessage() for r in prov_logs)

    def test_sourceless_creation_pass_origin_node_loss_warns(self, prov_logs):
        # Source-less helper creation does not excuse provenance loss on an
        # existing buffer reconstructed by the same pass.
        b = _Buf(origins={"a"}, origin_node="a")
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            b.origin_node = None
        assert any("insert_restickify" in r.getMessage() for r in prov_logs)


class TestPipelineWrapping:
    def test_node_pipeline_observes_each_pass(self, prov_logs):
        from torch_spyre._inductor.passes import _SpyreNodePassPipeline

        def dropping_node_pass(nodes):
            for n in nodes:
                n.node.origins = set()  # wrongly clears provenance
            return nodes

        pipeline = _SpyreNodePassPipeline([dropping_node_pass])
        # Force the device guard on so the loop body runs off-device.
        pipeline._has_spyre_device = lambda target: True
        pipeline([_unit(_Buf(origins={"a"}))])
        assert any("dropping_node_pass" in r.getMessage() for r in prov_logs)

    def test_graphlowering_pipeline_observes_each_pass(self, prov_logs, monkeypatch):
        import torch_spyre._inductor.passes as passes_mod

        class _FakeGraph:
            def __init__(self, ops):
                self.operations = ops

        def dropping_gl_pass(graph):
            for op in graph.operations:
                op.origins = set()

        monkeypatch.setattr(
            passes_mod, "_operations_have_spyre_device", lambda ops: True
        )
        pipeline = passes_mod.CustomPreSchedulingPasses()
        pipeline.passes = [dropping_gl_pass]
        pipeline(_FakeGraph([_Buf(origins={"a"})]))
        assert any("dropping_gl_pass" in r.getMessage() for r in prov_logs)

    def test_warning_dedup_resets_per_pipeline_run(self, prov_logs):
        from torch_spyre._inductor.passes import _SpyreNodePassPipeline

        def dedup_reset_pass(nodes):
            for n in nodes:
                n.node.origins = set()  # drop provenance every run
            return nodes

        pipeline = _SpyreNodePassPipeline([dedup_reset_pass])
        pipeline._has_spyre_device = lambda target: True

        # The buffer grows per emission and is never cleared between runs:
        # this is what proves the observer now genuinely re-emits per compile
        # (the cumulative count strictly increases on the second run) rather
        # than being silenced by Python's per-process warning registry, which
        # this test would have caught under the old warnings.warn emission.
        pipeline([_unit(_Buf(origins={"a"}))])
        after_first = sum(1 for r in prov_logs if "dedup_reset_pass" in r.getMessage())
        pipeline([_unit(_Buf(origins={"a"}))])
        after_second = sum(1 for r in prov_logs if "dedup_reset_pass" in r.getMessage())
        assert after_first >= 1
        assert after_second > after_first  # reset -> a new run warns again
