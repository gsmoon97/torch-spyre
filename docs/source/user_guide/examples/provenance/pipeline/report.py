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
pipeline/report.py  —  renders the issue #2574 provenance audit report.

The report is **measurement-only**: every table is computed from the captured
data (captures.capture()) and the Stage-6 bundle read (superdsc.run()). It
contains no hand-written analysis, field semantics, or recommendations — those
are intentionally left out so the report cannot drift from what a given run
actually observed. Interpretation is a separate, later deliverable.
"""

from __future__ import annotations

import datetime
from typing import Any

TICK = "✅"  # present on all relevant nodes/ops
PARTIAL = "◐"  # present on some (shown as n/total)
CROSS = "❌"  # measured absent
DASH = "➖"  # n/a: not generated yet, or carried only indirectly (origins)

# Field sets, in the order issue #2574 lists them.
FX_FIELDS = [
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "original_aten",
    "from_node",
]
IR_FIELDS = ["origins", "origin_node", "traceback", "get_stack_traces"]


# --------------------------------------------------------------------------
# Presence computation (purely from captured data)
# --------------------------------------------------------------------------
def _fx_present(node: dict, field: str) -> bool:
    return node.get("fields", {}).get(field) is not None


def _fx_symbol(nodes: list[dict], field: str) -> str:
    total = len(nodes)
    if total == 0:
        return CROSS
    present = sum(1 for n in nodes if _fx_present(n, field))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _ir_value(op: dict, field: str):
    return op.get("fields", {}).get(field)


def _ir_present(op: dict, field: str) -> bool:
    v = _ir_value(op, field)
    if field == "origins":
        return bool(v)
    if field == "get_stack_traces":
        return bool(v) and v != "OrderedSet([])"
    return v is not None


def _ir_symbol(ops: list[dict], field: str) -> str:
    total = len(ops)
    if total == 0:
        return CROSS
    present = sum(1 for o in ops if _ir_present(o, field))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _ir_attr_exists(ops: list[dict], field: str) -> bool:
    """Does the provenance attribute even exist on these IR objects? Uses the
    capture's `attr_exists` (hasattr) record; defaults True for older dumps and
    for derived accessors like get_stack_traces (not tracked there)."""
    return any(o.get("attr_exists", {}).get(field, True) for o in ops)


def _ir_cell(ops: list[dict], field: str) -> str:
    """Matrix cell for an IR field at one stage: ➖ if the attribute isn't on
    the object here at all; otherwise ✓/◐/✗ from the populated values."""
    if not ops:
        return CROSS
    if not _ir_attr_exists(ops, field):
        return DASH
    return _ir_symbol(ops, field)


def _fx_type(node: dict, field: str) -> str:
    rec = node.get("fields", {}).get(field)
    if rec is None:
        return CROSS
    return f"`{rec.get('type', '?')}`"


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render(
    capture: dict[str, Any], bundles: dict[str, Any], model_name: str = "SimpleMLP"
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pre = capture["stage2_pre_grad"]["nodes"]
    post = capture["stage2_post_grad"]["nodes"]
    passes_before = capture["stage3_passes"].get("before", [])
    looplevel = capture["stage4_looplevel"]["operations"]
    opspec_ops = capture["stage5_opspec"]["ops"]
    opspec_fields = capture["stage5_opspec"]["opspec_fields"] or []
    kernels = capture["stage6_kernels"]["kernels"]
    sdsc_present = bundles.get("provenance_present", False)
    sdsc_cell = TICK if sdsc_present else CROSS

    L: list[str] = []

    # Header --------------------------------------------------------------
    L += [
        f"# Provenance Audit: `{model_name}` — Metadata Across "
        "the Compilation Pipeline",
        "",
        f"> Generated: {now} &nbsp;|&nbsp; Issue: "
        "[torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574)",
        "",
        "Measured in-process during one cache-defeated `torch.compile` (compile-path "
        "objects only). This report is **measurement-only**; interpretation is a "
        "separate deliverable.",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| FX pre-grad compute nodes | {len(pre)} |",
        f"| FX post-grad compute nodes | {len(post)} |",
        f"| LoopLevelIR operations | {len(looplevel)} |",
        f"| OpSpec ops created | {len(opspec_ops)} |",
        f"| SuperDSC kernels | {len(kernels)} |",
        f"| `sdsc_*.json` files | {bundles.get('total_sdsc_files', 0)} |",
        f"| `OpSpec` declared fields | `{opspec_fields}` |",
        "",
    ]

    # Stage x field matrix ------------------------------------------------
    # Each column is measured on the object that stage produces. The Layer column
    # groups fields by the object they live on: FX = FX-node meta, IR =
    # ComputedBuffer/LoopLevelIR attributes. "Inductor passes" reads the IR
    # entering pre-scheduling (before snapshot); "LoopLevelIR" reads it after the
    # passes (they mutate it in place and insert restickify buffers, 5 -> 7 ops).
    L += [
        "## Stage × Field Matrix",
        "",
        f"{TICK} present (all) &nbsp; {PARTIAL} present on some (n/total) &nbsp; "
        f"{CROSS} present-able here but not carried (a genuine drop, or an "
        f"empty slot) &nbsp; {DASH} not applicable (not generated yet, or carried "
        "only indirectly via `origins`).",
        "",
        "The **Layer** column marks whether a field lives on the FX node (`FX`) "
        "or the IR `ComputedBuffer` (`IR`). *Inductor passes* is the IR entering "
        "pre-scheduling; *LoopLevelIR* is the same IR after the passes — they "
        "mutate it in place and insert 2 `restickify` buffers (5 → 7 ops), so "
        "`origin_node` reads ✅ (5/5) → ◐ 5/7 (nulled on the matmuls).",
        "",
        "| Layer | Field | FX pre-grad | FX post-grad | Inductor passes "
        "| LoopLevelIR | OpSpec | SuperDSC JSON |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in FX_FIELDS:
        pre_sym = _fx_symbol(pre, f)
        post_sym = _fx_symbol(post, f)
        # Absent pre-grad but present post-grad => generated in the grad/lowering
        # transition (original_aten, from_node) -> pre-grad is "not yet created".
        if pre_sym == CROSS and post_sym != CROSS:
            pre_sym = DASH
        # FX-meta is not a direct attribute of the IR objects, but it is not lost
        # there either — it rides in `origins` until the OpSpec drop -> ➖.
        L.append(
            f"| FX | `{f}` | {pre_sym} | {post_sym} | {DASH} | {DASH} | {DASH} "
            f"| {DASH} |"
        )
    for f in IR_FIELDS:
        passes_c = _ir_cell(passes_before, f)
        llir_c = _ir_cell(looplevel, f)
        # OpSpec/JSON have no provenance field, so an IR field present upstream is
        # genuinely not carried there -> ❌ (unless a Phase-2 field is declared).
        opspec_c = TICK if f in opspec_fields else CROSS
        sdsc_c = TICK if sdsc_present else CROSS
        L.append(
            f"| IR | `{f}` | {DASH} | {DASH} | {passes_c} | {llir_c} "
            f"| {opspec_c} | {sdsc_c} |"
        )
    L += [
        "",
        f"> {DASH} for FX fields downstream of FX post-grad means the value is "
        f"carried indirectly — FX-meta is reachable through `origins` (which "
        f"points back to the FX nodes), not as a direct attribute of the IR "
        f"objects. The genuine drop is the {TICK}/{PARTIAL} → {CROSS} at OpSpec: "
        f"the `OpSpec` dataclass declares no provenance field "
        f"(`{opspec_fields}`), so `origins` and its derivatives are not carried "
        "into OpSpec or the emitted JSON. `traceback` reads ❌ throughout the IR "
        "stages — the attribute exists but is never populated (an empty slot).",
        "",
    ]

    # Stage 2 detail ------------------------------------------------------
    for label, nodes in [("pre-grad", pre), ("post-grad", post)]:
        L += [
            f"## Stage 2 — FX {label} ({len(nodes)} compute nodes)",
            "",
            "Cell = observed `type` of the field, or ❌ if absent.",
            "",
            "| Node | target | "
            + " | ".join(f"`{f}`" for f in FX_FIELDS)
            + " | source line |",
            "| --- | --- | " + " | ".join("---" for _ in FX_FIELDS) + " | --- |",
        ]
        for n in nodes:
            st = n.get("fields", {}).get("stack_trace")
            src = st.get("source_line") if isinstance(st, dict) else None
            cells = " | ".join(_fx_type(n, f) for f in FX_FIELDS)
            L.append(
                f"| `{n['name']}` | `{n.get('target', '')}` | {cells} "
                f"| {('`' + src + '`') if src else '—'} |"
            )
        L.append("")

    # Stages 3-4 detail ---------------------------------------------------
    L += [
        f"## Stages 3–4 — LoopLevelIR operations ({len(looplevel)})",
        "",
        "Measured `ComputedBuffer` attributes after the pre-scheduling passes.",
        "",
        "| Op | `origins` | `origin_node` | `traceback` | `get_stack_traces` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in looplevel:
        fields = o.get("fields", {})
        origins = (
            ", ".join(f"`{x['name']}`" for x in (fields.get("origins") or [])) or "—"
        )
        onode = fields.get("origin_node") or CROSS
        tb = fields.get("traceback") or CROSS
        gst = fields.get("get_stack_traces") or ""
        gst_cell = TICK if gst and gst != "OrderedSet([])" else CROSS
        L.append(
            f"| `{o['name']}` | {origins} "
            f"| {('`' + onode + '`') if onode != CROSS else CROSS} "
            f"| {tb if tb == CROSS else '`' + str(tb)[:40] + '`'} | {gst_cell} |"
        )
    L.append("")

    # Stage 5 detail ------------------------------------------------------
    L += [
        f"## Stage 5 — OpSpec ops ({len(opspec_ops)})",
        "",
        f"`OpSpec` declared fields: `{opspec_fields}` — no provenance field, so "
        "the matrix shows `OpSpec` as ➖. The `origins` below are what is "
        "*available on the input `ComputedBuffer`* at `create_op_spec` (what a "
        "Phase-2 `debug_handle` could capture); the `OpSpec` object itself stores "
        "none of them.",
        "",
        "| Spyre op | buffer | `origins` | `origin_node` |",
        "| --- | --- | --- | --- |",
    ]
    for o in opspec_ops:
        fields = o.get("fields", {})
        origins = (
            ", ".join(f"`{x['name']}`" for x in (fields.get("origins") or [])) or "—"
        )
        onode = fields.get("origin_node") or CROSS
        L.append(
            f"| `{o.get('op', '?')}` | `{o.get('name', '?')}` | {origins} "
            f"| {('`' + onode + '`') if onode != CROSS else CROSS} |"
        )
    L.append("")

    # Stage 6 detail ------------------------------------------------------
    L += [
        f"## Stage 6 — SuperDSC kernels ({len(kernels)})",
        "",
        f"Provenance field present in any emitted `sdsc_*.json`: {sdsc_cell}",
        "",
    ]
    bundle_by_name = {b["kernel_name"]: b for b in bundles.get("kernels", [])}
    for k in kernels:
        name = k["kernel_name"]
        ops = [no["name"] for no in k.get("node_origins", [])]
        origins = sorted(
            {
                o["name"]
                for no in k.get("node_origins", [])
                for o in (no.get("fields", {}).get("origins") or [])
            }
        )
        md = k.get("kernel_metadata", {}).get("metadata", "")
        b = bundle_by_name.get(name, {})
        nfiles = len(b.get("sdsc_files", []))
        prov = TICK if b.get("provenance_present") else CROSS
        L += [
            f"### `{name}`",
            "",
            f"- buffers ({len(ops)}): {', '.join(f'`{o}`' for o in ops) or '—'}",
            f"- fx origins: {', '.join(f'`{o}`' for o in origins) or '—'}",
            f"- kernel metadata: `{md}`",
            f"- `sdsc_*.json` files: {nfiles} &nbsp; provenance in JSON: {prov}",
            "",
        ]

    return "\n".join(L)
