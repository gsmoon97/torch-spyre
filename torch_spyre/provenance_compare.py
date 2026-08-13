# Copyright 2026 The Torch-Spyre Authors.
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

"""Generic compile-variant comparison model for the provenance viewer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from typing import Any


def build_compile_comparison(
    document: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    phase_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build source cohorts and compile-variant summaries from validated evidence."""
    observations_by_identity = {
        event["identityKey"]: list(event["observations"]) for event in events
    }
    phase_ranges = {
        item["id"]: item for item in (phase_analysis or {}).get("phaseRanges", [])
    }
    occurrences_by_compile: dict[str, list[tuple[str, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for occurrence_id, occurrence in document["kernelOccurrences"].items():
        occurrences_by_compile[occurrence["compileId"]].append(
            (occurrence_id, occurrence)
        )

    group_records = [
        _compile_group(
            compile_id,
            sorted(occurrences, key=lambda item: item[0]),
            observations_by_identity,
            phase_ranges,
        )
        for compile_id, occurrences in sorted(occurrences_by_compile.items())
    ]
    _order_and_label_groups(group_records, phase_ranges)
    group_order = {
        group["compileId"]: index for index, group in enumerate(group_records)
    }

    members = _operation_members(document)
    cohorts = _cohorts(members, group_records, observations_by_identity)
    cohorts.sort(key=_cohort_sort_key)

    return {
        "comparisonKind": "compile-variants",
        "ordering": (
            "runtime-chronology"
            if all(group["ordinal"] is not None for group in group_records)
            else "stable-compile-id"
        ),
        "groups": group_records,
        "cohorts": cohorts,
        "groupOrder": [
            compile_id
            for compile_id, _ in sorted(group_order.items(), key=lambda item: item[1])
        ],
    }


def _cohort_sort_key(cohort: Mapping[str, Any]) -> tuple[object, ...]:
    pattern = cohort["bundleCountPattern"]
    returning_split = (
        len(pattern) == 3
        and pattern[0] > 0
        and pattern[0] == pattern[2]
        and pattern[1] > pattern[0]
    )
    changed = len(set(pattern)) > 1
    priority = 0 if returning_split else 1 if changed else 2
    return priority, -max(pattern, default=0), cohort["label"], cohort["id"]


def _compile_group(
    compile_id: str,
    occurrences: Sequence[tuple[str, Mapping[str, Any]]],
    observations_by_identity: Mapping[str, Sequence[Mapping[str, Any]]],
    phase_ranges: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    identities = sorted({occurrence["identityKey"] for _, occurrence in occurrences})
    observations = [
        observation
        for identity_key in identities
        for observation in observations_by_identity.get(identity_key, ())
    ]
    phase_ids = sorted(
        {
            assignment["phaseRangeId"]
            for observation in observations
            if (assignment := observation.get("phaseAssignment"))
            and assignment["status"] == "assigned"
            and assignment["phaseRangeId"] in phase_ranges
        },
        key=lambda phase_id: (
            phase_ranges[phase_id]["startTimestampUs"],
            phase_ranges[phase_id]["traceEventIndex"],
        ),
    )
    valid_durations = [
        observation["durationUs"]
        for observation in observations
        if observation["durationUs"] is not None
    ]
    identity_steps = {
        (identity_key, observation["commandStep"])
        for identity_key in identities
        for observation in observations_by_identity.get(identity_key, ())
    }
    return {
        "compileId": compile_id,
        "label": "Compile variant " + compile_id[:12],
        "ordinal": None,
        "annotation": _phase_annotation(phase_ids, phase_ranges),
        "phaseRangeIds": phase_ids,
        "bundleIdentityCount": len(identities),
        "observationCount": len(observations),
        "identityStepGroupCount": len(identity_steps),
        "validDurationObservationCount": len(valid_durations),
        "excludedDurationObservationCount": len(observations) - len(valid_durations),
        "totalDurationUs": sum(valid_durations),
        "meanDurationUs": (
            sum(valid_durations) / len(valid_durations) if valid_durations else None
        ),
        "maximumDurationUs": max(valid_durations) if valid_durations else None,
    }


def _order_and_label_groups(
    groups: list[dict[str, Any]],
    phase_ranges: Mapping[str, Mapping[str, Any]],
) -> None:
    chronological = all(len(group["phaseRangeIds"]) == 1 for group in groups)
    range_ids = [group["phaseRangeIds"][0] for group in groups] if chronological else []
    chronological = chronological and len(range_ids) == len(set(range_ids))
    if not chronological:
        groups.sort(key=lambda item: item["compileId"])
        return
    groups.sort(
        key=lambda item: (
            phase_ranges[item["phaseRangeIds"][0]]["startTimestampUs"],
            item["compileId"],
        )
    )
    for ordinal, group in enumerate(groups, 1):
        group["ordinal"] = ordinal
        group["label"] = f"Compile variant {ordinal}"


def _phase_annotation(
    phase_ids: Sequence[str],
    phase_ranges: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if len(phase_ids) != 1:
        return None
    phase = phase_ranges[phase_ids[0]]
    if phase["phaseKind"] == "prefill":
        return "Prefill"
    decode_index = phase.get("decodeIndex")
    return "Decode" if decode_index is None else f"Decode {decode_index}"


def _operation_members(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    handles = document["handles"]
    projections = document["upstreamProjections"]
    members: list[dict[str, Any]] = []
    for occurrence_id, occurrence in sorted(document["kernelOccurrences"].items()):
        compile_id = occurrence["compileId"]
        identity_key = occurrence["identityKey"]
        identity = document["kernelIdentities"][identity_key]
        projection = projections[compile_id]
        post_names = sorted(
            {
                post_name
                for registration in occurrence["registrations"]
                for post_name in projection["cppCodeToPost"].get(
                    registration["alias"], []
                )
            }
        )
        bindings = identity["specHandleBindings"] or [
            {"handleId": handle_id, "specPath": []}
            for handle_id in identity["directHandleIds"]
        ]
        for binding_position, binding in enumerate(bindings):
            for leaf_position, handle_id in enumerate(
                _leaf_handle_ids(binding["handleId"], handles)
            ):
                handle = handles[handle_id]
                exact_posts = [
                    name for name in post_names if name in handle["ir_chain"]
                ]
                candidate = _operation_candidate(handle, exact_posts)
                members.append(
                    {
                        "compileId": compile_id,
                        "occurrenceId": occurrence_id,
                        "identityKey": identity_key,
                        "bindingPosition": binding_position,
                        "leafPosition": leaf_position,
                        "specPath": copy.deepcopy(binding["specPath"]),
                        "handleId": handle_id,
                        "source": copy.deepcopy(handle["source"]),
                        "atenOp": handle["aten_op"],
                        "exactPostGradOrigins": [
                            {"compileId": compile_id, "name": name}
                            for name in exact_posts
                        ],
                        "operationMatchKey": candidate,
                        "matchState": (
                            "candidate"
                            if candidate is not None
                            else "ambiguous"
                            if len(exact_posts) > 1
                            else "unavailable"
                        ),
                    }
                )
    return members


def _leaf_handle_ids(
    handle_id: str,
    handles: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    fused_from = handles[handle_id]["fused_from"]
    if not fused_from:
        return [handle_id]
    return [
        leaf for child_id in fused_from for leaf in _leaf_handle_ids(child_id, handles)
    ]


def _operation_candidate(
    handle: Mapping[str, Any],
    exact_posts: Sequence[str],
) -> str | None:
    if handle["source"] is None or handle["aten_op"] is None or len(exact_posts) != 1:
        return None
    payload = {
        "source": handle["source"],
        "atenOp": handle["aten_op"],
        "postGradOrigin": exact_posts[0],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _cohorts(
    members: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    observations_by_identity: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    handle_compiles: dict[str, set[str]] = defaultdict(set)
    for member in members:
        handle_compiles[member["handleId"]].add(member["compileId"])
    cross_compile_handles = {
        handle_id
        for handle_id, compile_ids in handle_compiles.items()
        if len(compile_ids) > 1
    }
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_handle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for member in members:
        if member["handleId"] in cross_compile_handles:
            by_handle[member["handleId"]].append(member)
        elif member["operationMatchKey"] is not None:
            by_candidate[member["operationMatchKey"]].append(member)
        else:
            by_handle[member["handleId"]].append(member)

    cohorts = []
    for candidate, selected in sorted(by_candidate.items()):
        cohorts.append(
            _cohort(
                "operation:" + candidate[:20],
                "candidate",
                selected,
                groups,
                observations_by_identity,
            )
        )
    for handle_id, selected in sorted(by_handle.items()):
        cohorts.append(
            _cohort(
                "handle:" + handle_id,
                "exact-handle",
                selected,
                groups,
                observations_by_identity,
            )
        )
    return cohorts


def _cohort(
    cohort_id: str,
    basis: str,
    members: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    observations_by_identity: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    first = members[0]
    source = first["source"]
    aten_op = first["atenOp"]
    post_names = sorted(
        {
            origin["name"]
            for member in members
            for origin in member["exactPostGradOrigins"]
        }
    )
    group_records = []
    for group in groups:
        compile_id = group["compileId"]
        selected = [member for member in members if member["compileId"] == compile_id]
        identities = sorted({member["identityKey"] for member in selected})
        bundles = []
        for identity_key in identities:
            bundle_members = [
                member for member in selected if member["identityKey"] == identity_key
            ]
            observations = observations_by_identity.get(identity_key, ())
            bundles.append(
                {
                    "identityKey": identity_key,
                    "memberCount": len(bundle_members),
                    "handleIds": sorted(
                        {member["handleId"] for member in bundle_members}
                    ),
                    "traceEventIndices": [
                        observation["traceEventIndex"] for observation in observations
                    ],
                }
            )
        group_records.append(
            {
                "compileId": compile_id,
                "memberCount": len(selected),
                "distinctHandleCount": len({member["handleId"] for member in selected}),
                "bundleCount": len(identities),
                "observationCount": sum(
                    len(observations_by_identity.get(identity_key, ()))
                    for identity_key in identities
                ),
                "collision": len({member["handleId"] for member in selected}) > 1,
                "bundles": bundles,
            }
        )
    return {
        "id": cohort_id,
        "basis": basis,
        "label": _cohort_label(source, aten_op, post_names, first["handleId"]),
        "source": copy.deepcopy(source),
        "atenOp": aten_op,
        "postGradOrigins": post_names,
        "groups": group_records,
        "bundleCountPattern": [group["bundleCount"] for group in group_records],
        "hasCollision": any(group["collision"] for group in group_records),
    }


def _cohort_label(
    source: Mapping[str, Any] | None,
    aten_op: str | None,
    post_names: Sequence[str],
    handle_id: str,
) -> str:
    parts = []
    if source is not None:
        parts.append(f"{source['file']}:{source['start_line']}:{source['start_col']}")
    if aten_op is not None:
        parts.append(aten_op)
    if len(post_names) == 1:
        parts.append(post_names[0])
    return " | ".join(parts) if parts else "Handle " + handle_id


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
