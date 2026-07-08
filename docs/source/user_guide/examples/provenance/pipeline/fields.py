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
pipeline/fields.py — the single source of truth for the provenance field names.

Both the capture layer (captures.py) and the report layer (report.py) key off
these lists, so they are defined once here to prevent drift.
"""

from __future__ import annotations

# FX ``node.meta`` provenance fields (issue #2574), in the order the issue
# lists them.
FX_FIELDS = [
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "original_aten",
    "from_node",
]

# Inductor IR / LoopLevelIR provenance attributes stored on the
# ``ComputedBuffer`` (what the capture reads directly).
IR_ATTR_FIELDS = ["origins", "origin_node", "traceback"]

# IR provenance shown as matrix columns: the stored attributes plus the derived
# ``get_stack_traces()`` accessor (not a stored attribute, so it is separate
# from IR_ATTR_FIELDS).
IR_FIELDS = IR_ATTR_FIELDS + ["get_stack_traces"]

# Provenance fields an ``OpSpec`` instance may carry. Phase 2a (#2575) adds
# ``debug_handle``; the IR attrs / accessor are checked too so the OpSpec column
# can show partial population once a field is declared but not on every op.
OPSPEC_PROVENANCE_FIELDS = IR_ATTR_FIELDS + ["get_stack_traces", "debug_handle"]
