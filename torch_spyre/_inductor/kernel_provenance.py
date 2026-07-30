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

"""Stable provenance identity for finalized Spyre kernel bundles.

This module is independent of any profiler transport. It aggregates the
DebugHandles directly attached to finalized OpSpecs and fingerprints the full
OpSpec/LoopSpec tree. Profiler event-name encoding lives in ``profiler_event``;
the provenance sidecar and future structured profiler metadata can therefore
reuse this identity without depending on AIUPTI/Kineto string constraints.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence

import sympy

from torch_spyre._inductor.op_spec import DebugHandle, LoopSpec, OpSpec, TensorArg


# Version 1 defines the first public finalized-bundle fingerprint. Any
# canonicalization change must bump this version so persisted sidecars and
# profiler events are never reinterpreted.
_KERNEL_BUNDLE_KEY_DOMAIN = "spyre-kernel-bundle"
KERNEL_PROVENANCE_KEY_VERSION = 1
KERNEL_PROVENANCE_KEY_BASE32_WIDTH = 52

_EXPECTED_OP_SPEC_FIELDS = frozenset(
    {
        "op",
        "is_reduction",
        "iteration_space",
        "args",
        "op_info",
        "tiled_symbols",
        "tiled_symbol_trip_counts",
        "symbolic_dim_bounds",
        "node_output_ranges",
        "debug_handle",
    }
)
_EXPECTED_TENSOR_ARG_FIELDS = frozenset(
    {
        "is_input",
        "arg_index",
        "device_dtype",
        "device_size",
        "device_coordinates",
        "allocation",
        "per_tile_fixed",
        "name",
        "device_tile_advance_expr",
    }
)
_EXPECTED_LOOP_SPEC_FIELDS = frozenset({"count", "body"})


@dataclasses.dataclass(frozen=True)
class KernelProvenanceDescriptor:
    """Immutable identity and direct provenance links for one kernel bundle.

    ``key`` fingerprints the finalized OpSpec tree, including each OpSpec's
    directly attached DebugHandle ID. ``debug_handle_ids`` is the ordered,
    deduplicated set exposed to profiler consumers; recursive ``fused_from``
    lineage remains in the handle records. Because handle IDs derive from
    source metadata and IR names, key stability is scoped to equivalent
    recompiles in the same source/toolchain context, not source relocation.
    ``aten_ops`` contains sorted,
    deduplicated ATen names for human-readable consumers; it is not identity.
    """

    key: str
    debug_handle_ids: tuple[str, ...]
    aten_ops: tuple[str, ...]


def build_kernel_provenance_descriptor(
    specs: Sequence[object],
) -> KernelProvenanceDescriptor:
    """Build the provenance identity for a finalized OpSpec tree.

    Nested ``LoopSpec`` bodies are traversed depth-first. Repeated IDs are
    deduplicated without sorting so the descriptor retains deterministic
    compiler emission order. A valid bundle always has an identity, even when
    no OpSpec carries a DebugHandle.
    """
    _validate_finalized_schema()
    handles = _deduplicate_handles(_iter_debug_handles(specs))
    handle_ids = tuple(str(handle.id) for handle in handles)
    key = _kernel_bundle_key(specs)
    return KernelProvenanceDescriptor(
        key=key,
        debug_handle_ids=handle_ids,
        aten_ops=_collect_aten_ops(handles),
    )


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


def _collect_aten_ops(handles: tuple[DebugHandle, ...]) -> tuple[str, ...]:
    """Collect recursive ATen lineage once per stable handle ID."""
    names: set[str] = set()
    seen_ids: set[int] = set()

    def collect(handle: DebugHandle) -> None:
        if handle.id in seen_ids:
            return
        seen_ids.add(handle.id)
        if handle.aten_op is not None:
            names.add(handle.aten_op)
        for constituent in handle.fused_from:
            collect(constituent)

    for handle in handles:
        collect(handle)
    return tuple(sorted(names))


def _kernel_bundle_key(specs: Sequence[object]) -> str:
    """Fingerprint the complete finalized OpSpec tree.

    The canonical payload contains execution-relevant OpSpec/LoopSpec structure
    and the directly attached handle ID at every OpSpec position. It excludes
    Python identities, temporary output paths, generated kernel counters, and
    runtime launch state. Any canonicalization change requires a visible format
    version bump.
    """
    payload = {
        "domain": _KERNEL_BUNDLE_KEY_DOMAIN,
        "version": KERNEL_PROVENANCE_KEY_VERSION,
        "specs": [_canonical_spec(spec) for spec in specs],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("ascii")).digest()
    return base64.b32encode(digest).decode("ascii").rstrip("=").lower()


def _validate_finalized_schema() -> None:
    """Reject schema drift that the canonical bundle payload would omit."""
    schemas = (
        (OpSpec, _EXPECTED_OP_SPEC_FIELDS),
        (TensorArg, _EXPECTED_TENSOR_ARG_FIELDS),
        (LoopSpec, _EXPECTED_LOOP_SPEC_FIELDS),
    )
    for schema, expected_fields in schemas:
        actual_fields = frozenset(field.name for field in dataclasses.fields(schema))
        if actual_fields == expected_fields:
            continue
        added = sorted(actual_fields - expected_fields)
        removed = sorted(expected_fields - actual_fields)
        raise TypeError(
            f"{schema.__name__} schema changed; update kernel provenance "
            f"canonicalization (added={added}, removed={removed})"
        )


def _canonical_spec(spec: object) -> object:
    if isinstance(spec, OpSpec):
        return {
            "kind": "op",
            "op": spec.op,
            "is_reduction": spec.is_reduction,
            # Iteration-space order determines generated loop dimensions.
            "iteration_space": [
                [
                    _canonical_value(symbol),
                    _canonical_value(extent),
                    _canonical_value(work_division),
                ]
                for symbol, (extent, work_division) in spec.iteration_space.items()
            ],
            "args": [_canonical_tensor_arg(arg) for arg in spec.args],
            "op_info": _canonical_value(spec.op_info),
            "tiled_symbols": [
                [_canonical_value(symbol) for symbol in level]
                for level in spec.tiled_symbols
            ],
            "tiled_symbol_trip_counts": _canonical_value(spec.tiled_symbol_trip_counts),
            "symbolic_dim_bounds": _canonical_value(spec.symbolic_dim_bounds),
            "node_output_ranges": _canonical_value(spec.node_output_ranges),
            # Preserve position and multiplicity in the bundle fingerprint.
            # The descriptor separately deduplicates its consumer-facing list.
            "debug_handle_id": (
                str(spec.debug_handle.id) if spec.debug_handle is not None else None
            ),
        }
    if isinstance(spec, LoopSpec):
        return {
            "kind": "loop",
            "count": _canonical_value(spec.count),
            "body": [_canonical_spec(child) for child in spec.body],
        }
    raise TypeError(f"Unsupported finalized kernel spec: {type(spec).__qualname__}")


def _canonical_tensor_arg(arg: TensorArg) -> object:
    return {
        "is_input": arg.is_input,
        "arg_index": arg.arg_index,
        "device_dtype": str(arg.device_dtype),
        "device_size": _canonical_value(arg.device_size),
        "device_coordinates": _canonical_value(arg.device_coordinates),
        "allocation": _canonical_value(arg.allocation),
        "per_tile_fixed": arg.per_tile_fixed,
        "name": arg.name,
        "device_tile_advance_expr": _canonical_value(arg.device_tile_advance_expr),
    }


def _canonical_value(value: object) -> object:
    """Convert finalized schema values to deterministic JSON primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # hex() preserves exact finite values and signed zero without depending
        # on JSON float formatting.
        return {"float": value.hex()}
    if isinstance(value, sympy.Basic):
        return {"sympy": sympy.srepr(value)}
    if isinstance(value, Mapping):
        items = [
            [_canonical_value(key), _canonical_value(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"mapping": items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    raise TypeError(
        "Unsupported value in finalized OpSpec bundle: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
