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

"""Source-to-kernel provenance construction for the Spyre Inductor backend.

The provenance *types* (``SourceLoc``, ``DebugHandle``) live in ``op_spec.py``
alongside the other IR-op schema dataclasses. This module holds the *logic* that
builds them from Inductor IR: stable-id hashing here, and ``build_debug_handle``
(reading ``ComputedBuffer.origins``) in a later task.
"""

from __future__ import annotations

import hashlib
from typing import Any

import regex

from torch_spyre._inductor.op_spec import DebugHandle, SourceLoc


_FRAME_RE = regex.compile(r'File "([^"]+)", line (\d+)')


def _stable_id(
    source: SourceLoc | None,
    aten_op: str | None,
    ir_chain: tuple[str, ...],
) -> int:
    """Deterministic content hash. Reproducible across processes (unlike hash())."""
    canonical = "|".join(
        [
            source.to_str() if source is not None else "",
            aten_op or "",
            ",".join(ir_chain),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1  # 63-bit non-negative int


def _source_from_node(node: Any) -> SourceLoc | None:
    """Extract a structured SourceLoc from an FX node's ``stack_trace`` meta."""
    meta = getattr(node, "meta", None) or {}
    trace = meta.get("stack_trace")
    if not trace:
        return None
    matches = _FRAME_RE.findall(trace)
    if not matches:
        return None
    user = [(f, ln) for (f, ln) in matches if "/torch/" not in f]
    file, line = (user or matches)[-1]
    return SourceLoc(file=file, start_line=int(line))


def _aten_from_node(node: Any) -> str | None:
    """Extract the ``original_aten`` op string from an FX node's meta."""
    meta = getattr(node, "meta", None) or {}
    op = meta.get("original_aten")
    return str(op) if op is not None else None


def _headline_source(per_node: list) -> SourceLoc | None:
    """The single distinct source if the origins agree on one, else None.

    Symmetric with the ``aten_op`` rule in ``build_debug_handle``: we never
    present an arbitrary line as *the* source of an op fused across multiple
    distinct source locations. When the origins disagree, the headline is None
    and the full set lives in ``fused_from`` (consumers pick a representative).
    """
    distinct = {s.to_str(): s for (_, s, _) in per_node if s is not None}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def build_debug_handle(buffer: Any) -> DebugHandle | None:
    """Build a DebugHandle from a ComputedBuffer's origins. Best-effort, never raises.

    ``origins`` (the set) is authoritative. ``origin_node`` is used only when
    Inductor set it (the clean 1:1 non-view case); for fused/view ops it is None
    and we do not invent a primary — the full set lives in ``fused_from``.
    """
    origins = getattr(buffer, "origins", None) or set()
    if not origins:
        return None
    # Stable iteration order; NOT a semantic primary pick (full truth in fused_from).
    nodes = sorted(origins, key=lambda n: getattr(n, "name", ""))
    per_node = [
        (getattr(n, "name", ""), _source_from_node(n), _aten_from_node(n))
        for n in nodes
    ]
    origin_node = getattr(buffer, "origin_node", None)

    if origin_node is not None:
        # Inductor's authoritative 1:1 op: take its aten/source directly. If it
        # has no stack_trace, fall back only to a single agreed sibling source.
        aten_op = _aten_from_node(origin_node)
        source = _source_from_node(origin_node) or _headline_source(per_node)
    else:
        # Fused/view: headline source and aten_op are set only when the origins
        # agree on a single distinct value; otherwise None (do not guess) — the
        # full per-origin set is preserved in fused_from. Source and aten are
        # handled symmetrically.
        source = _headline_source(per_node)
        atens = {a for (_, _, a) in per_node if a is not None}
        aten_op = next(iter(atens)) if len(atens) == 1 else None

    buf_name = None
    get_name = getattr(buffer, "get_name", None)
    if callable(get_name):
        buf_name = get_name()
    ir_chain = tuple([nm for (nm, _, _) in per_node] + ([buf_name] if buf_name else []))

    # fused_from: the authoritative set — every origin when the kernel fuses >1.
    fused_from: tuple[DebugHandle, ...] = ()
    if len(nodes) > 1:
        fused_from = tuple(
            DebugHandle(
                id=_stable_id(s, a, (nm,)),
                source=s,
                aten_op=a,
                ir_chain=(nm,),
            )
            for (nm, s, a) in per_node
        )

    return DebugHandle(
        id=_stable_id(source, aten_op, ir_chain),
        source=source,
        aten_op=aten_op,
        ir_chain=ir_chain,
        fused_from=fused_from,
    )
