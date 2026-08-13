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

"""Device-free tests for Phase 4b phase attribution."""

from __future__ import annotations

import json
from pathlib import Path

from torch_spyre.provenance_phase import build_phase_analysis
from torch_spyre.provenance_viewer import build_provenance_presentation


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "provenance"
_SIDECAR = _FIXTURE_DIR / "valid_v1.json"
_IDENTITY_KEY = "atqydvnuutl766na"
_EVENT_BASE = "spyre_kernel_v1_fused_linear_relu_atqydvnuutl766na"


def _observation(
    trace_index: int,
    timestamp: float,
    correlation: object,
    *,
    duration: float | None = 1.0,
    step: int = 0,
) -> dict:
    return {
        "traceEventIndex": trace_index,
        "name": f"{_EVENT_BASE}#{step}",
        "timestampUs": timestamp,
        "durationUs": duration,
        "commandStep": step,
        "correlation": correlation,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_correlation_first_assignment_and_explicit_failure_states():
    trace_events = [
        {
            "name": "granite.iteration.0",
            "ph": "X",
            "ts": 0.0,
            "dur": 400.0,
        },
        {
            "name": "granite.iteration.0.prefill",
            "ph": "X",
            "ts": 10.0,
            "dur": 80.0,
        },
        {
            "name": "granite.iteration.0.decode.1",
            "ph": "X",
            "ts": 100.0,
            "dur": 100.0,
        },
        {
            "name": "granite.iteration.0.decode.bad",
            "ph": "X",
            "ts": 200.0,
            "dur": 50.0,
        },
        {
            "name": "aiuLaunchControlBlocks",
            "ph": "X",
            "ts": 50.0,
            "dur": 1.0,
            "args": {"correlation": 11},
        },
        {
            "name": "aiuLaunchControlBlocks",
            "ph": "X",
            "ts": 120.0,
            "dur": 1.0,
            "args": {"correlation": 33},
        },
        {
            "name": "aiuLaunchControlBlocks",
            "ph": "X",
            "ts": 130.0,
            "dur": 1.0,
            "args": {"correlation": 33},
        },
    ]
    observations = [
        ("bundle-a", _observation(10, 150.0, 11, duration=3.0, step=1)),
        ("bundle-b", _observation(11, 150.0, 22, duration=4.0, step=2)),
        ("bundle-c", _observation(12, 150.0, 33, step=3)),
        ("bundle-d", _observation(13, 350.0, None, step=4)),
    ]

    analysis, diagnostics = build_phase_analysis(
        trace_events,
        observations,
        "granite",
    )

    assert [item["label"] for item in analysis["phaseRanges"]] == [
        "granite.iteration.0.prefill",
        "granite.iteration.0.decode.1",
    ]
    assert [item["label"] for item in analysis["iterationRanges"]] == [
        "granite.iteration.0"
    ]
    assignments = [item["phaseAssignment"] for _, item in observations]
    assert assignments[0]["status"] == "assigned"
    assert assignments[0]["source"] == "correlation"
    assert assignments[0]["matchedRuntimeLaunchTraceIndices"] == [4]
    assert assignments[1]["status"] == "assigned"
    assert assignments[1]["source"] == "kernel-start-fallback"
    assert assignments[2]["status"] == "ambiguous"
    assert assignments[2]["matchedRuntimeLaunchTraceIndices"] == [5, 6]
    assert assignments[3]["status"] == "unassigned"
    assert analysis["assignmentSummary"] == {
        "assignedByCorrelation": 1,
        "assignedByKernelStartFallback": 1,
        "ambiguous": 1,
        "unassigned": 1,
    }
    assert [
        (
            item["bundleIdentityCount"],
            item["observationCount"],
            item["identityStepGroupCount"],
        )
        for item in analysis["stageSummaries"]
    ] == [(1, 1, 1), (1, 1, 1)]
    codes = {item["code"] for item in diagnostics}
    assert {
        "duplicate-runtime-launch-correlation",
        "kernel-start-outside-phase",
        "kernel-start-fallback",
        "malformed-phase-label",
        "missing-observation-correlation",
        "missing-runtime-launch",
    }.issubset(
        codes | set(assignments[1]["diagnostics"]) | set(assignments[3]["diagnostics"])
    )


def test_innermost_range_and_equal_specificity_ambiguity():
    trace_events = [
        {
            "name": "granite.iteration.0.prefill",
            "ph": "X",
            "ts": 0.0,
            "dur": 100.0,
        },
        {
            "name": "granite.iteration.0.decode.1",
            "ph": "X",
            "ts": 20.0,
            "dur": 40.0,
        },
        {
            "name": "aiuLaunchControlBlocks",
            "ph": "X",
            "ts": 30.0,
            "dur": 1.0,
            "args": {"correlation": 1},
        },
    ]
    observation = _observation(3, 200.0, 1)

    _, _ = build_phase_analysis(
        trace_events,
        [("bundle", observation)],
        "granite",
    )

    assert observation["phaseAssignment"]["status"] == "assigned"
    assert observation["phaseAssignment"]["phaseRangeId"] == "granite-phase:1"

    trace_events[1]["name"] = "granite.iteration.0.prefill"
    trace_events[1]["ts"] = 0.0
    trace_events[1]["dur"] = 100.0
    observation = _observation(3, 200.0, 1)
    analysis, _ = build_phase_analysis(
        trace_events,
        [("bundle", observation)],
        "granite",
    )
    assert observation["phaseAssignment"]["status"] == "ambiguous"
    assert analysis["assignmentSummary"]["ambiguous"] == 1


def test_viewer_adapter_is_opt_in_and_increments_presentation_version(tmp_path):
    trace = tmp_path / "trace.json"
    _write_json(
        trace,
        {
            "traceEvents": [
                {
                    "name": "granite.iteration.0.prefill",
                    "ph": "X",
                    "ts": 0.0,
                    "dur": 100.0,
                },
                {
                    "name": "aiuLaunchControlBlocks",
                    "ph": "X",
                    "ts": 25.0,
                    "dur": 1.0,
                    "args": {"correlation": 7},
                },
                {
                    "name": _EVENT_BASE + "#2",
                    "ph": "X",
                    "ts": 125.0,
                    "dur": 5.0,
                    "args": {
                        "provenance_key": _IDENTITY_KEY,
                        "debug_handles": ["200"],
                        "correlation": 7,
                    },
                },
            ]
        },
    )

    base = build_provenance_presentation(_SIDECAR, kineto_trace=trace)
    phase_aware = build_provenance_presentation(
        _SIDECAR,
        kineto_trace=trace,
        phase_adapter="granite",
    )

    assert base["presentationVersion"] == 1
    assert "phaseAnalysis" not in base
    base_observation = next(
        event["observations"][0] for event in base["events"] if event["observations"]
    )
    assert "phaseAssignment" not in base_observation

    assert phase_aware["presentationVersion"] == 2
    assert phase_aware["phaseAnalysis"]["adapter"] == "granite"
    observation = next(
        event["observations"][0]
        for event in phase_aware["events"]
        if event["observations"]
    )
    assert observation["phaseAssignment"] == {
        "candidatePhaseRangeIds": [],
        "diagnostics": [],
        "matchedRuntimeLaunchTraceIndices": [1],
        "phaseRangeId": "granite-phase:0",
        "source": "correlation",
        "status": "assigned",
    }
    assert phase_aware["phaseAnalysis"]["stageSummaries"][0]["bundleIdentityCount"] == 1
