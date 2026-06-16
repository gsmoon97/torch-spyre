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
"""
pipeline/superdsc.py  —  Stage 6 (SuperDSC JSON) reader.

Reads the emitted ``sdsc_*.json`` bundles from the *exact* per-kernel output
directories captured in-process at compile time (captures.stage6_kernels.
output_dirs), keyed by ``kernel_name``. No /tmp globbing, no mtime guessing,
no iteration-space heuristics — the kernel→dir map is recorded by the compiler
itself ([async_compile.get_output_dir]).

For each kernel we report how many ``sdsc_*.json`` files (one per OpSpec) it
emitted and whether any provenance field appears anywhere in the serialized
JSON. Confirmed absence is the Stage-6 finding.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

# Provenance field names (issue #2574) we scan the serialized JSON for.
PROVENANCE_FIELD_NAMES = [
    "stack_trace",
    "original_aten",
    "from_node",
    "origins",
    "origin_node",
    "traceback",
    "provenance",
    "debug_handle",
    "source",
]


def _scan_json_for_provenance(text: str) -> dict[str, bool]:
    low = text.lower()
    return {name: (name.lower() in low) for name in PROVENANCE_FIELD_NAMES}


def read_bundle(kernel_name: str, output_dir: str) -> dict[str, Any]:
    """Read all sdsc_*.json under one kernel's output_dir."""
    d = pathlib.Path(output_dir)
    rec: dict[str, Any] = {
        "kernel_name": kernel_name,
        "output_dir": output_dir,
        "exists": d.is_dir(),
        "sdsc_files": [],
        "provenance_present": False,
    }
    if not d.is_dir():
        return rec

    for jf in sorted(d.glob("sdsc_*.json")):
        try:
            text = jf.read_text()
            json.loads(text)  # validate
        except Exception as e:
            rec["sdsc_files"].append({"file": jf.name, "parse_error": str(e)})
            continue
        hits = _scan_json_for_provenance(text)
        rec["sdsc_files"].append(
            {
                "file": jf.name,
                "provenance_fields": {k: v for k, v in hits.items() if v},
                "has_provenance": any(hits.values()),
            }
        )

    rec["provenance_present"] = any(f.get("has_provenance") for f in rec["sdsc_files"])
    return rec


def run(output_dirs: dict[str, str]) -> dict[str, Any]:
    """Read every captured kernel bundle.

    Args:
        output_dirs: kernel_name -> exact output directory, as captured by the
            in-process hook on async_compile.get_output_dir during this run.

    Returns:
        {
          "kernels": [read_bundle(...), ...],   # one per kernel_name
          "total_kernels": int,
          "total_sdsc_files": int,
          "provenance_present": bool,            # any field in any bundle
        }
    """
    kernels = [read_bundle(name, d) for name, d in sorted(output_dirs.items())]
    total_files = sum(len(k["sdsc_files"]) for k in kernels)
    return {
        "kernels": kernels,
        "total_kernels": len(kernels),
        "total_sdsc_files": total_files,
        "provenance_present": any(k["provenance_present"] for k in kernels),
    }
