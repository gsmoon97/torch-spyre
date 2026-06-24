# Provenance Audit — Analysis

> Companion to `provenance_audit.md` (the measurement-only report) for
> [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574).
> Reference model: 1-hidden-layer MLP (`128→256→128`, fp16), torch 2.11.
> This document interprets the measured numbers; all claims trace to the audit
> capture (`provenance_capture_raw.json`), the emitted `sdsc_*.json` bundles, or
> cited source.
>
> **⚠️ Specific to one audit run — `provenance_audit.md` Generated:
> `2026-06-24 16:08`.** That report is regenerated on every run; if its
> timestamp no longer matches the one above, the counts and findings here may be
> stale. Re-read against the current `provenance_audit.md` and update this file.

## 0. Changes since the previous run

Two measured differences from the earlier capture — both explained by `main`
advancing between runs (restickify is a deterministic, layout-driven decision,
so this is a compiler-version effect, not randomness), and neither changes the
core finding:

- **No `restickify` split this run.** LoopLevelIR holds **5** operations; an
  earlier `main` inserted 2 `restickify` buffers (5 → 7), since optimized away
  for this model. So `origins` reads **5/5** here, not 7/7.
- **`origin_node` is fully present (✅ 5/5).** The earlier run showed it nulled
  on both matmul buffers (◐ 5/7); here it survives on all five ops. That its
  coverage shifts across compiler versions is exactly why provenance should ride
  `origins` — the **complete set** of contributing nodes (stable 5/5) — not the
  single optional `origin_node` pointer (`fx.Node | None`, nullable by design).

**Unchanged:** the genuine drop is still **OpSpec → SuperDSC JSON**. The
`OpSpec` dataclass declares no provenance field, so nothing source-related
reaches the emitted JSON.

## 1. Field semantics

How each field is structured and how it comes to exist (copied from an upstream
node, derived on demand, or generated fresh):

| Field | Represents | Observed type | Populated by | Copied / derived / generated |
| --- | --- | --- | --- | --- |
| `stack_trace` | the user source `file:line` that created the op | `str` (traceback; last frame = user line) | Dynamo, trace time | copied across decomposition via `fx_traceback.preserve_node_meta()`; **not** propagated to synthesized `mm`/`add` from `addmm` |
| `nn_module_stack` | the `nn.Module` path owning the op (e.g. `fc1` → `Linear`) | `dict {id: (qual_name, cls)}` | Dynamo | copied; on module-call nodes only (the two `Linear`s, not functional `relu`) |
| `source_fn_stack` | the source function/op that produced the node | `list[(seq_name, fn/cls)]` | Dynamo | copied; tracks the source op, survives on anchor nodes |
| `original_aten` | the ATen op this node lowered from (e.g. `addmm`) | `OpOverload` | Inductor, post-grad | generated per node; present on **all 7** post-grad nodes — the most robust FX field |
| `from_node` | the pass/transform chain that produced the node | `list[NodeSource(name, pass_name, action, graph_id)]` | Inductor `GraphTransformObserver` | generated as nodes are transformed; present on the anchor nodes (3/7 here) and **run-dependent** (see §6) |
| `origins` | the set of FX nodes that lowered into this buffer | `OrderedSet[fx.Node]` | Inductor IR lowering | generated when fx nodes lower to a `ComputedBuffer`; propagated by Spyre passes — robust (present **5/5** at the IR stages this run) |
| `origin_node` | a single representative FX node for this buffer | `fx.Node \| None` | Inductor IR | copied; **run-dependent** — present 5/5 this run, but nulled on both matmul buffers (5/7) in the earlier run |
| `traceback` | the IR node's creation-site traceback | `str \| None` | Inductor IR | the attribute **exists** on the IR node but is never populated for this model — an unused provenance slot (`hasattr` true, value null at every IR stage) |
| `get_stack_traces()` | source lines derived from the buffer's `origins` | `OrderedSet[str]` (accessor) | `ComputedBuffer` | **derived** on demand from `origins[*].meta['stack_trace']`; empty when origins lack a stack trace — here the two bias `add`s (`op1`, `op4`), so 3/5 |

## 2. Metadata flow

```text
  nn.Module source ── file:line, module tree (ground truth)
        │ Dynamo
  FX pre-grad ───────── stack_trace, nn_module_stack, source_fn_stack
        │ AOTAutograd: addmm decomposition (linear → permute + mm + add)
        │   • anchor nodes (permute/relu) keep stack_trace/nn_module_stack/source_fn_stack
        │   • synthesized mm/add keep only original_aten   ← partial loss #1
  FX post-grad ──────── + original_aten (all 7), from_node (3/7, run-dependent)
        │ Inductor lowering → ComputedBuffer.origins
  LoopLevelIR (pre-pass)  origins (5/5), origin_node (5/5) entering the passes
        │            ↓ pre-scheduling passes mutate in place (no restickify this run; 5 ops)
  LoopLevelIR (post-pass) origins (5/5); origin_node (5/5 this run; 5/7 earlier)
        │               get_stack_traces() derived from origins (3/5)
        │ SpyreKernel.create_op_spec
  OpSpec ────────────── the OpSpec dataclass has no provenance field, so origins
        │               (still on the buffer feeding create_op_spec) is discarded ← DROP #2
        │ codegen/superdsc.py (parse_op_spec / compile_op_spec)
  SuperDSC JSON ─────── no provenance field emitted              ← DROP #3
                        (get_kernel_metadata assembles full source mapping at
                         define_kernel, emitted only as a wrapper comment)
```

## 3. Exact drop locations

1. **`addmm` decomposition (AOTAutograd).** `aten.addmm` → `mm` + `add`; the
   synthesized `mm`/`add` nodes carry `original_aten` but not
   `stack_trace`/`nn_module_stack`/`source_fn_stack`. Evidence: post-grad
   `mm_default*`/`add_tensor*` have those three fields `null` (a *partial* loss —
   `original_aten` still resolves them to `aten.addmm`).
2. **OpSpec carries nothing (DROP #2 — the real drop).** `create_op_spec`
   (`torch_spyre/_inductor/spyre_kernel.py:516`) reads the input
   `ComputedBuffer.origins`, but the `OpSpec` dataclass
   (`torch_spyre/_inductor/op_spec.py:67`) declares only
   `op, is_reduction, iteration_space, args, op_info, tiled_symbols` — no field
   to hold provenance. So `origins` (present 5/5 on the buffer) is dropped.
3. **Serialization emits nothing (DROP #3).** `parse_op_spec`/`compile_op_spec`
   (`torch_spyre/_inductor/codegen/superdsc.py`) consume only `OpSpec` fields.
   Confirmed by direct inspection of the emitted bundles (§4):
   `provenance in JSON: ❌` for all 5 `sdsc_*.json` files.

> Note on `origin_node`: in the earlier run it was nulled on the two matmul
> buffers between entry and exit of `CustomPreSchedulingPasses`
> (`torch_spyre/_inductor/passes.py`), reading ◐ 5/7. This run it survives 5/5.
> Because its coverage is schedule-dependent and `origins` strictly supersedes
> it, the audit does not chase the exact nulling pass (see §5/§6).

## 4. What the SuperDSC JSON actually contains

Two kernels are emitted, 5 `sdsc_*.json` files in total (one per OpSpec):

- `sdsc_fused_addmm_linear_relu_0` — 4 buffers (`op0`–`op3`), kernel metadata
  `Original ATen: [aten.linear, aten.addmm, aten.relu]`, source nodes
  `[x, x_1, x_2]`; 4 `sdsc_*.json` files.
- `sdsc_fused_addmm_1` — 1 buffer (`op4`), kernel metadata
  `Original ATen: [aten.addmm]`; 1 `sdsc_*.json` file.

Each file is purely the **device-execution spec** — iteration space, labeled
tensors, the compute op (`opFuncName`/`exUnit`), the schedule tree (HBM
allocations + device addresses), and core/work-slice maps. **No `stack_trace`,
`origins`, `origin_node`, `original_aten`, `from_node`, or any source reference
appears in any file** — the factual confirmation of DROP #3. The only
source-aware artifact at this stage is the wrapper comment that
`get_kernel_metadata(node_schedule)` assembles at `define_kernel` (the
`Original ATen` / source-node line above), which is emitted as a code comment,
not into the SDSC JSON.

## 5. If we add a `debug_handle` at OpSpec, what can we derive it from?

At `create_op_spec` the input `ComputedBuffer` carries `origins`
(`OrderedSet[fx.Node]`), present on **5/5** ops. From each origin node's `.meta`:

- **`original_aten`** — present on all origins → always derivable.
- **`stack_trace` / `nn_module_stack` / `source_fn_stack`** — present on anchor
  origins (matmul, relu); absent on the bias-`add` origins (`op1`, `op4`), which
  can therefore only be sourced to their `original_aten`.

**Recommended carrier: `origins`.** Do not use `origin_node` (run-dependent —
5/5 here but 5/7 earlier) or `from_node` (run-dependent, §6). Equivalently,
`get_kernel_metadata(node_schedule)` at `define_kernel` already aggregates this
per kernel and is keyed by `kernel_name` — e.g. `sdsc_fused_addmm_linear_relu_0`
→ `Original ATen: [aten.linear, aten.addmm, aten.relu]`, source nodes
`[x, x_1, x_2]`. **Phase 2 (#2575):** add a `debug_handle` field to `OpSpec`,
populate it from `origins` in `create_op_spec`, and serialize it through
`codegen/superdsc.py`. This mirrors upstream's
`_inductor_kernel_provenance_debug_handle` (an int handle) in
`torch/_inductor/debug.py`.

## 6. Caveats for implementation

- **`from_node` is run-dependent.** This audit (a normal compile) shows it on
  3/7 post-grad nodes — the anchor nodes; it rises to 7/7 when
  debug/provenance-tracking is active. Do not assume it exists in production
  compiles.
- **`origin_node` is run-dependent.** Present 5/5 this run, but nulled on the
  matmul buffers (5/7) in the earlier run. Coverage depends on the pass
  schedule, so `origins` (5/5 in both runs) is the stable carrier.
- **`traceback` is an unused slot.** The attribute *exists* on the IR node
  (confirmed via `hasattr` in the capture) but is never populated for this
  model — so it reads ❌ (reachable but empty) at the IR stages, not ➖. A
  profiler could populate it, but nothing does today.
- **Op count is schedule-dependent.** This run produced 5 LoopLevelIR ops; the
  earlier run produced 7 (2 `restickify` buffers). A regression suite (#2581)
  should assert provenance *coverage* (every op carries `origins`), not a fixed
  op count.
- **Upstream `inductor_provenance_tracking_node_mappings.json` is not emitted**
  by the Spyre backend today; Phase 3b should emit a compatible artifact so
  `tlparse` tooling works (RFC 0601 §4.7).
