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

"""Profiler-facing identity for kernels containing provenance-aware OpSpecs.

The constants below follow existing compiler/runtime contracts rather than
choosing a new wire format locally:

* upstream Inductor kernel names are backend-qualified, followed by a
  descriptive fused-op component and then a hash or unique suffix (for example
  ``cpp_*``, ``triton_*``, and ``pallas_*``);
* torch-spyre already follows that layout with ``sdsc_<fused_name>_<suffix>``;
* torch-spyre PR #2930 forwards the selected kernel name through
  ``ComputeParams::kernel_name`` and uses a ``size_t`` JobPlan step index; and
* libaiupti declares ``AIUpti_ActivityCompute.name`` as ``char[128]`` and copies
  at most ``sizeof(name) - 1`` bytes before writing the terminating NUL.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
from collections.abc import Iterator, Sequence

import regex

from torch_spyre._inductor.op_spec import DebugHandle, LoopSpec, OpSpec, SourceLoc


_AIUPTI_ACTIVITY_NAME_BUFFER_BYTES = 128
AIUPTI_ACTIVITY_NAME_MAX_BYTES = _AIUPTI_ACTIVITY_NAME_BUFFER_BYTES - 1
# Payload limit after reserving libaiupti's required terminating NUL.

# PR #2930 represents the JobPlan command position as C++ ``size_t`` and its
# existing fallback spells the disambiguator as ``#<step_idx>``. Python and the
# extension share one process ABI, so derive the maximum suffix width instead
# of assuming that size_t is always 64-bit.
_SIZE_T_BITS = ctypes.sizeof(ctypes.c_size_t) * 8
_MAX_COMPUTE_STEP_SUFFIX_BYTES = len(f"#{(1 << _SIZE_T_BITS) - 1}")
_MAX_EVENT_NAME_BASE_BYTES = (
    AIUPTI_ACTIVITY_NAME_MAX_BYTES - _MAX_COMPUTE_STEP_SUFFIX_BYTES
)

# This domain/version tag is part of the persisted content hash. Domain
# separation prevents an ordered handle-ID tuple from aliasing another Spyre
# hash that happens to use the same values. A future canonicalization change
# must bump the version so existing trace/sidecar joins are not reinterpreted.
_KERNEL_PROVENANCE_KEY_DOMAIN = "spyre-kernel-provenance"
_KERNEL_PROVENANCE_KEY_VERSION = 1

# Keep a public backend-qualified prefix rather than exposing the replaceable
# SDSC IR in the trace name. As in upstream Inductor, the descriptive fused-op
# text precedes the stable identity suffix. The complete 63-bit aggregate
# (matching ``provenance._stable_id``'s signed-int64 contract) is rendered as 16
# hex digits; only the non-authoritative display text may be truncated.
_EVENT_NAME_PREFIX = "spyre_kernel_"
_EVENT_KEY_HEX_WIDTH = 16
_EVENT_KEY_RE = regex.compile(
    rf"\A{regex.escape(_EVENT_NAME_PREFIX)}"
    rf"[A-Za-z0-9_]+_(?P<key>[0-9a-f]{{{_EVENT_KEY_HEX_WIDTH}}})"
    rf"(?:#[0-9]+)?\Z"
)

# Upstream's generated C++ wrapper likewise normalizes kernel names to
# ``[A-Za-z0-9_]``. Restricting our display-only text to that alphabet also
# makes byte truncation exact for libaiupti's fixed buffer; the full provenance
# remains available through the key and handle mapping.
_DISPLAY_COMPONENT_RE = regex.compile(r"[^A-Za-z0-9]+")


@dataclasses.dataclass(frozen=True)
class KernelProvenanceDescriptor:
    """Immutable identity shared by compiler artifacts and profiler events.

    ``key`` identifies the kernel-level ordered handle set. It is intentionally
    distinct from each per-OpSpec ``DebugHandle.id`` in ``debug_handle_ids``.
    ``event_name`` leaves enough space for any local ``size_t`` ``#<step>``
    suffix added by the JobPlan path.
    """

    key: str
    debug_handle_ids: tuple[str, ...]
    event_name: str


def build_kernel_provenance_descriptor(
    specs: Sequence[object],
) -> KernelProvenanceDescriptor | None:
    """Build the profiler identity for finalized OpSpecs in emission order.

    Nested ``LoopSpec`` bodies are traversed depth-first. Repeated IDs are
    deduplicated without sorting so the descriptor retains deterministic
    compiler emission order. Returns ``None`` when no OpSpec has provenance.
    """
    handles = _deduplicate_handles(_iter_debug_handles(specs))
    if not handles:
        return None

    handle_ids = tuple(str(handle.id) for handle in handles)
    key = _kernel_provenance_key(handle_ids)
    event_name = _format_event_name(key, handles)
    return KernelProvenanceDescriptor(
        key=key,
        debug_handle_ids=handle_ids,
        event_name=event_name,
    )


def extract_kernel_provenance_key(event_name: str) -> str | None:
    """Extract a kernel-provenance key from a Spyre device event name."""
    match = _EVENT_KEY_RE.match(event_name)
    return match.group("key") if match is not None else None


def _iter_debug_handles(specs: Sequence[object]) -> Iterator[DebugHandle]:
    for spec in specs:
        if isinstance(spec, OpSpec):
            if spec.debug_handle is not None:
                yield spec.debug_handle
        elif isinstance(spec, LoopSpec):
            yield from _iter_debug_handles(spec.body)


def _deduplicate_handles(handles: Iterator[DebugHandle]) -> tuple[DebugHandle, ...]:
    unique: list[DebugHandle] = []
    seen_ids: set[int] = set()
    for handle in handles:
        if handle.id not in seen_ids:
            unique.append(handle)
            seen_ids.add(handle.id)
    return tuple(unique)


def _kernel_provenance_key(handle_ids: tuple[str, ...]) -> str:
    """Hash the complete ordered handle set into a bounded kernel identity.

    A kernel may contain multiple OpSpecs, so using the first DebugHandle ID
    would lose provenance. NUL separators make decimal ID tuple boundaries
    unambiguous, and the domain/version header keeps this key space distinct
    and explicitly evolvable. The full ID tuple remains authoritative.
    """
    domain_and_version = (
        f"{_KERNEL_PROVENANCE_KEY_DOMAIN}-v{_KERNEL_PROVENANCE_KEY_VERSION}"
    ).encode("ascii")
    payload = (
        domain_and_version
        + b"\0"
        + b"\0".join(handle_id.encode("ascii") for handle_id in handle_ids)
    )
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 1
    return f"{value:0{_EVENT_KEY_HEX_WIDTH}x}"


def _format_event_name(key: str, handles: tuple[DebugHandle, ...]) -> str:
    label, source = _agreed_headline(handles)
    if label is None or source is None:
        has_fusion = len(handles) > 1 or any(handle.fused_from for handle in handles)
        display = "fused_at_unknown_0" if has_fusion else "unknown_at_unknown_0"
    else:
        display = "_at_".join(
            (
                _sanitize_component(label),
                f"{_sanitize_component(_source_basename(source.file))}_{source.start_line}",
            )
        )

    key_suffix = f"_{key}"
    display_budget = _MAX_EVENT_NAME_BASE_BYTES - len(
        f"{_EVENT_NAME_PREFIX}{key_suffix}".encode("ascii")
    )
    display = display[:display_budget].rstrip("_")
    return f"{_EVENT_NAME_PREFIX}{display}{key_suffix}"


def _agreed_headline(
    handles: tuple[DebugHandle, ...],
) -> tuple[str | None, SourceLoc | None]:
    first = handles[0]
    if first.aten_op is None or first.source is None:
        return None, None
    if all(
        handle.aten_op == first.aten_op and handle.source == first.source
        for handle in handles[1:]
    ):
        return first.aten_op, first.source
    return None, None


def _source_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _sanitize_component(value: str) -> str:
    sanitized = _DISPLAY_COMPONENT_RE.sub("_", value).strip("_")
    return sanitized or "unknown"


# TODO(PyTorch 2.12): register ``key -> debug_handle_ids`` with the out-of-tree
# PrivateUse1 activity profiler and emit the IDs as structured Kineto metadata.
# This descriptor and event-name contract are shared by PyTorch 2.11 and 2.12;
# structured metadata is additive, and the name remains the compatibility join.
