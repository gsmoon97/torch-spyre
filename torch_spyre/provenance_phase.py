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

"""Replaceable phase adapters for provenance-viewer applications."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import regex


_GRANITE_OUTER_RE = regex.compile(
    r"\Agranite\.iteration\.(?P<iteration>0|[1-9][0-9]*)\Z"
)
_GRANITE_PHASE_RE = regex.compile(
    r"\Agranite\.iteration\.(?P<iteration>0|[1-9][0-9]*)\."
    r"(?:(?P<prefill>prefill)|decode\.(?P<decode>0|[1-9][0-9]*))\Z"
)
_GRANITE_PREFIX_RE = regex.compile(r"\Agranite\.iteration\.")
_RUNTIME_LAUNCH_NAME = "aiuLaunchControlBlocks"


def build_phase_analysis(
    trace_events: Sequence[object],
    observations: Sequence[tuple[str, dict[str, Any]]],
    adapter: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Annotate observations and build one phase-aware presentation record."""
    if adapter != "granite":
        raise ValueError(f"unsupported phase adapter: {adapter}")

    phase_ranges, iteration_ranges, diagnostics = _parse_granite_ranges(trace_events)
    launches, launch_diagnostics = _runtime_launches(trace_events)
    diagnostics.extend(launch_diagnostics)

    assignment_counts = {
        "assignedByCorrelation": 0,
        "assignedByKernelStartFallback": 0,
        "ambiguous": 0,
        "unassigned": 0,
    }
    for _, observation in observations:
        assignment, assignment_diagnostics = _assign_observation(
            observation,
            phase_ranges,
            launches,
        )
        observation["phaseAssignment"] = assignment
        diagnostics.extend(assignment_diagnostics)
        if assignment["status"] == "ambiguous":
            assignment_counts["ambiguous"] += 1
        elif assignment["status"] == "unassigned":
            assignment_counts["unassigned"] += 1
        elif assignment["source"] == "correlation":
            assignment_counts["assignedByCorrelation"] += 1
        else:
            assignment_counts["assignedByKernelStartFallback"] += 1

    return (
        {
            "adapter": adapter,
            "phaseRanges": phase_ranges,
            "iterationRanges": iteration_ranges,
            "assignmentSummary": assignment_counts,
            "stageSummaries": _stage_summaries(phase_ranges, observations),
        },
        diagnostics,
    )


def _parse_granite_ranges(
    trace_events: Sequence[object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase_ranges: list[dict[str, Any]] = []
    iteration_ranges: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for trace_index, raw_event in enumerate(trace_events):
        if not isinstance(raw_event, Mapping):
            continue
        name = raw_event.get("name")
        if not isinstance(name, str) or _GRANITE_PREFIX_RE.match(name) is None:
            continue

        outer_match = _GRANITE_OUTER_RE.fullmatch(name)
        phase_match = _GRANITE_PHASE_RE.fullmatch(name)
        if outer_match is None and phase_match is None:
            diagnostics.append(
                _diagnostic(
                    "malformed-phase-label",
                    "warning",
                    "Granite-like trace label is not an anchored supported phase",
                    {"traceEventIndex": trace_index, "eventName": name},
                )
            )
            continue

        bounds = _range_bounds(raw_event)
        if raw_event.get("ph") != "X" or bounds is None:
            diagnostics.append(
                _diagnostic(
                    "malformed-phase-range",
                    "warning",
                    "phase range requires a finite nonnegative complete event",
                    {"traceEventIndex": trace_index, "eventName": name},
                )
            )
            continue
        start, end = bounds
        if outer_match is not None:
            iteration_ranges.append(
                {
                    "id": f"granite-iteration:{trace_index}",
                    "iteration": int(outer_match.group("iteration")),
                    "traceEventIndex": trace_index,
                    "startTimestampUs": start,
                    "endTimestampUs": end,
                    "label": name,
                }
            )
            continue

        assert phase_match is not None
        phase_kind = "prefill" if phase_match.group("prefill") else "decode"
        decode_group = phase_match.group("decode")
        phase_ranges.append(
            {
                "id": f"granite-phase:{trace_index}",
                "iteration": int(phase_match.group("iteration")),
                "phaseKind": phase_kind,
                "decodeIndex": int(decode_group) if decode_group is not None else None,
                "traceEventIndex": trace_index,
                "startTimestampUs": start,
                "endTimestampUs": end,
                "label": name,
            }
        )

    def range_order(item: Mapping[str, Any]) -> tuple[int | float, int | float, int]:
        return (
            item["startTimestampUs"],
            item["endTimestampUs"],
            item["traceEventIndex"],
        )

    phase_ranges.sort(key=range_order)
    iteration_ranges.sort(key=range_order)
    return phase_ranges, iteration_ranges, diagnostics


def _runtime_launches(
    trace_events: Sequence[object],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    launches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    for trace_index, raw_event in enumerate(trace_events):
        if not isinstance(raw_event, Mapping):
            continue
        if raw_event.get("name") != _RUNTIME_LAUNCH_NAME:
            continue
        args = raw_event.get("args")
        args = args if isinstance(args, Mapping) else {}
        correlation = _correlation(args.get("correlation"))
        timestamp = _finite_number(raw_event.get("ts"))
        if correlation is None:
            diagnostics.append(
                _diagnostic(
                    "malformed-runtime-launch-correlation",
                    "warning",
                    "runtime launch has a nonnumeric correlation identifier",
                    {"traceEventIndex": trace_index},
                )
            )
            continue
        if timestamp is None:
            diagnostics.append(
                _diagnostic(
                    "malformed-runtime-launch-timestamp",
                    "warning",
                    "runtime launch has no finite timestamp",
                    {
                        "traceEventIndex": trace_index,
                        "correlation": correlation,
                    },
                )
            )
            continue
        launches[correlation].append(
            {"traceEventIndex": trace_index, "timestampUs": timestamp}
        )

    for correlation, matches in sorted(launches.items()):
        if len(matches) > 1:
            diagnostics.append(
                _diagnostic(
                    "duplicate-runtime-launch-correlation",
                    "warning",
                    "several runtime launches share one correlation identifier",
                    {
                        "correlation": correlation,
                        "traceEventIndices": [
                            item["traceEventIndex"] for item in matches
                        ],
                    },
                )
            )
    return dict(launches), diagnostics


def _assign_observation(
    observation: Mapping[str, Any],
    phase_ranges: Sequence[Mapping[str, Any]],
    launches: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_index = observation["traceEventIndex"]
    event_name = observation["name"]
    details = {"traceEventIndex": trace_index, "eventName": event_name}
    diagnostic_codes: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    correlation = _correlation(observation.get("correlation"))

    if observation.get("correlation") is not None and correlation is None:
        diagnostic_codes.append("malformed-observation-correlation")
        diagnostics.append(
            _diagnostic(
                "malformed-observation-correlation",
                "warning",
                "kernel observation has a nonnumeric correlation identifier",
                details,
            )
        )
    if correlation is not None:
        matches = list(launches.get(correlation, ()))
        if len(matches) > 1:
            diagnostic_codes.append("duplicate-runtime-launch-correlation")
            return (
                _assignment(
                    "ambiguous",
                    "correlation",
                    diagnostic_codes,
                    matched_launch_indices=[
                        item["traceEventIndex"] for item in matches
                    ],
                ),
                diagnostics,
            )
        if len(matches) == 1:
            launch = matches[0]
            return _assignment_from_timestamp(
                launch["timestampUs"],
                phase_ranges,
                "correlation",
                diagnostic_codes,
                diagnostics,
                details,
                matched_launch_index=launch["traceEventIndex"],
            )
        diagnostic_codes.append("missing-runtime-launch")
        diagnostics.append(
            _diagnostic(
                "missing-runtime-launch",
                "warning",
                "no runtime launch matches the kernel correlation identifier",
                {**details, "correlation": correlation},
            )
        )
    elif observation.get("correlation") is None:
        diagnostic_codes.append("missing-observation-correlation")
        diagnostics.append(
            _diagnostic(
                "missing-observation-correlation",
                "warning",
                "kernel observation has no usable correlation identifier",
                details,
            )
        )

    diagnostic_codes.append("kernel-start-fallback")
    diagnostics.append(
        _diagnostic(
            "kernel-start-fallback",
            "warning",
            "phase assignment fell back to the kernel start timestamp",
            details,
        )
    )
    return _assignment_from_timestamp(
        observation.get("timestampUs"),
        phase_ranges,
        "kernel-start-fallback",
        diagnostic_codes,
        diagnostics,
        details,
    )


def _assignment_from_timestamp(
    timestamp: object,
    phase_ranges: Sequence[Mapping[str, Any]],
    source: str,
    diagnostic_codes: list[str],
    diagnostics: list[dict[str, Any]],
    details: Mapping[str, Any],
    *,
    matched_launch_index: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    number = _finite_number(timestamp)
    candidates = _innermost_ranges(number, phase_ranges) if number is not None else []
    if len(candidates) == 1:
        return (
            _assignment(
                "assigned",
                source,
                diagnostic_codes,
                phase_range_id=candidates[0]["id"],
                matched_launch_indices=(
                    [matched_launch_index] if matched_launch_index is not None else []
                ),
            ),
            diagnostics,
        )
    if len(candidates) > 1:
        code = "ambiguous-phase-range"
        diagnostic_codes.append(code)
        diagnostics.append(
            _diagnostic(
                code,
                "warning",
                "several equally specific phase ranges contain the assignment timestamp",
                {
                    **details,
                    "phaseRangeIds": [item["id"] for item in candidates],
                },
            )
        )
        return (
            _assignment(
                "ambiguous",
                source,
                diagnostic_codes,
                candidate_phase_range_ids=[item["id"] for item in candidates],
                matched_launch_indices=(
                    [matched_launch_index] if matched_launch_index is not None else []
                ),
            ),
            diagnostics,
        )

    code = (
        "runtime-launch-outside-phase"
        if source == "correlation"
        else "kernel-start-outside-phase"
    )
    diagnostic_codes.append(code)
    diagnostics.append(
        _diagnostic(
            code,
            "warning",
            "assignment timestamp is outside every supported phase range",
            dict(details),
        )
    )
    return (
        _assignment(
            "unassigned",
            source,
            diagnostic_codes,
            matched_launch_indices=(
                [matched_launch_index] if matched_launch_index is not None else []
            ),
        ),
        diagnostics,
    )


def _assignment(
    status: str,
    source: str,
    diagnostic_codes: Sequence[str],
    *,
    phase_range_id: str | None = None,
    candidate_phase_range_ids: Sequence[str] = (),
    matched_launch_indices: Sequence[int] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "phaseRangeId": phase_range_id,
        "candidatePhaseRangeIds": list(candidate_phase_range_ids),
        "matchedRuntimeLaunchTraceIndices": list(matched_launch_indices),
        "diagnostics": list(diagnostic_codes),
    }


def _innermost_ranges(
    timestamp: int | float,
    phase_ranges: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    candidates = [
        item
        for item in phase_ranges
        if item["startTimestampUs"] <= timestamp < item["endTimestampUs"]
    ]
    if not candidates:
        return []
    shortest = min(
        item["endTimestampUs"] - item["startTimestampUs"] for item in candidates
    )
    return [
        item
        for item in candidates
        if item["endTimestampUs"] - item["startTimestampUs"] == shortest
    ]


def _stage_summaries(
    phase_ranges: Sequence[Mapping[str, Any]],
    observations: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    summaries = []
    for phase_range in phase_ranges:
        selected = [
            (identity_key, observation)
            for identity_key, observation in observations
            if observation["phaseAssignment"]["status"] == "assigned"
            and observation["phaseAssignment"]["phaseRangeId"] == phase_range["id"]
        ]
        durations = [
            observation["durationUs"]
            for _, observation in selected
            if observation["durationUs"] is not None
        ]
        identity_steps = {
            (identity_key, observation["commandStep"])
            for identity_key, observation in selected
        }
        summaries.append(
            {
                "phaseRangeId": phase_range["id"],
                "bundleIdentityCount": len(
                    {identity_key for identity_key, _ in selected}
                ),
                "observationCount": len(selected),
                "identityStepGroupCount": len(identity_steps),
                "validDurationObservationCount": len(durations),
                "excludedDurationObservationCount": len(selected) - len(durations),
                "totalDurationUs": sum(durations),
                "meanDurationUs": (
                    sum(durations) / len(durations) if durations else None
                ),
                "maximumDurationUs": max(durations) if durations else None,
                "assignmentCounts": {
                    "correlation": sum(
                        observation["phaseAssignment"]["source"] == "correlation"
                        for _, observation in selected
                    ),
                    "kernelStartFallback": sum(
                        observation["phaseAssignment"]["source"]
                        == "kernel-start-fallback"
                        for _, observation in selected
                    ),
                },
            }
        )
    return summaries


def _range_bounds(event: Mapping[str, Any]) -> tuple[int | float, int | float] | None:
    start = _finite_number(event.get("ts"))
    duration = _finite_number(event.get("dur"))
    if start is None or duration is None or duration < 0:
        return None
    return start, start + duration


def _correlation(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not float(value).is_integer():
        return None
    return int(value)


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _diagnostic(
    code: str,
    severity: str,
    message: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "details": dict(details),
    }
