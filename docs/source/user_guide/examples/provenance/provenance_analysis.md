# Provenance Audit — Analysis

> Companion to `provenance_audit.md` (the measurement-only report) for
> [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574).
> Reference model: 1-hidden-layer MLP (`128→256→128`, fp16), torch 2.11.
> This document interprets the measured numbers; all claims trace to the audit
> capture (`provenance_capture_raw.json`), the emitted `sdsc_*.json` bundles, or
> cited source.

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
| `origins` | the set of FX nodes that lowered into this buffer | `OrderedSet[fx.Node]` | Inductor IR lowering | generated when fx nodes lower to a `ComputedBuffer`; propagated by Spyre passes that build buffers — robust (present 7/7 at the IR stages) |
| `origin_node` | a single representative FX node for this buffer | `fx.Node \| None` | Inductor IR | copied; **fragile** — nulled on both matmul buffers during pre-scheduling (5/7) |
| `traceback` | the IR node's creation-site traceback | `str \| None` | Inductor IR — copied onto cloned buffers (see §7) | the attribute **exists** on the IR node but is never populated for this model — an unused provenance slot (`hasattr` true, value null at every IR stage) |
| `get_stack_traces()` | source lines derived from the buffer's `origins` | `OrderedSet[str]` (accessor) | `ComputedBuffer` | **derived** on demand from `origins[*].meta['stack_trace']`; empty when origins lack a stack trace (bias adds, restickify) |

## 2. Metadata flow

```text
  nn.Module source ── file:line, module tree (ground truth)
        │ Dynamo
  FX pre-grad ───────── stack_trace, nn_module_stack, source_fn_stack
        │ AOTAutograd: addmm decomposition (linear → permute + mm + add)
        │   • anchor nodes (permute/relu) keep stack_trace/nn_module_stack/source_fn_stack
        │   • synthesized mm/add keep only original_aten   ← partial loss #1
  FX post-grad ──────── + original_aten (all), from_node (run-dependent)
        │ Inductor lowering → ComputedBuffer.origins
  Inductor passes ───── origins (all 5), origin_node (all 5) entering
        │            ↓ passes mutate in place + insert 2 restickify (5 → 7 ops)
  LoopLevelIR ───────── origins (all 7); origin_node nulled on matmuls (◐ 5/7) ← loss #2
        │               get_stack_traces() derived from origins
        │ SpyreKernel.create_op_spec
  OpSpec ────────────── the OpSpec dataclass has no provenance field, so origins
        │               (still on the buffer feeding create_op_spec) is discarded ← DROP #3
        │ codegen/superdsc.py (parse_op_spec / compile_op_spec)
  SuperDSC JSON ─────── no provenance field emitted              ← DROP #4
                        (get_kernel_metadata assembles full source mapping at
                         define_kernel, emitted only as a wrapper comment)
```

## 3. Exact drop locations

1. **`addmm` decomposition (AOTAutograd).** `aten.addmm` → `mm` + `add`; the
   synthesized `mm`/`add` nodes carry `original_aten` but not
   `stack_trace`/`nn_module_stack`/`source_fn_stack`. Evidence: post-grad
   `mm_default*`/`add_tensor*` have those three fields `null`.
2. **`origin_node` nulled in pre-scheduling.** Between entry and exit of
   `CustomPreSchedulingPasses` ([passes.py](../../torch_spyre/_inductor/passes.py)),
   `origin_node` goes from set → `null` on both matmul buffers (`op0`, `op3`);
   `origins` (the set) survives. Specific pass not isolated (not pursued —
   `origins` supersedes `origin_node`, see §5/§6).
3. **OpSpec carries nothing (DROP #3).** `create_op_spec`
   ([spyre_kernel.py:487](../../torch_spyre/_inductor/spyre_kernel.py#L487)) reads
   `self.current_node.node.origins` but the `OpSpec` dataclass
   ([op_spec.py:51](../../torch_spyre/_inductor/op_spec.py#L51)) declares only
   `op, is_reduction, iteration_space, args, op_info, tiled_symbols` — no field
   to hold provenance.
4. **Serialization emits nothing (DROP #4).** `parse_op_spec`/`compile_op_spec`
   ([codegen/superdsc.py](../../torch_spyre/_inductor/codegen/superdsc.py))
   consume only `OpSpec` fields. Confirmed by direct inspection of the emitted
   bundles (§4).

## 4. What the SuperDSC JSON actually contains

Direct inspection of one `sdsc_*.json` per kernel (each file = one OpSpec):

- `sdsc_fused_addmm_1/sdsc_0.json` (key `0_add`): `N_` iteration space
  (`mb_=2, out_=128`), `labeledDs_` (Tensor0/1/2), `computeOp_`
  (`opFuncName=add`, `exUnit=sfp`), `scheduleTree_` (HBM allocations + device
  addresses), core/work-slice maps.
- `sdsc_fused_addmm_linear_relu_0/sdsc_0.json` (key `0_ReStickifyOpHBM`): same
  shape, `N_` = `mb_=256, out_=128`.

**No `stack_trace`, `origins`, `origin_node`, `original_aten`, `from_node`, or
any source reference appears in either file.** The JSON is purely the
device-execution spec (layout, work division, addresses, op func). This is the
factual confirmation of DROP #4.

## 5. If we add a `debug_handle` at OpSpec, what can we derive it from?

At `create_op_spec` the input `ComputedBuffer` carries `origins`
(`OrderedSet[fx.Node]`), present on **7/7** ops. From each origin node's `.meta`:

- **`original_aten`** — present on all origins → always derivable.
- **`stack_trace` / `nn_module_stack` / `source_fn_stack`** — present on anchor
  origins (matmul, relu); absent on bias-add / restickify origins, which can
  therefore only be sourced to their `original_aten`.

**Recommended carrier: `origins`.** Do not use `origin_node` (fragile, 5/7) or
`from_node` (run-dependent, §6). Equivalently, `get_kernel_metadata(node_schedule)`
at `define_kernel` already aggregates this per kernel and is keyed by
`kernel_name` — e.g. `sdsc_fused_addmm_linear_relu_0` →
`Original ATen: [aten.linear, aten.addmm, aten.relu]`, source nodes `[x, x_1, x_2]`.
Phase 2: add a `debug_handle`/provenance field to `OpSpec`, populate from
`origins` in `create_op_spec`, serialize through `codegen/superdsc.py`.

## 6. Caveats for implementation

- **`from_node` is run-dependent.** This audit (a normal compile) shows it on
  3/7 post-grad nodes — the anchor nodes; it rises to 7/7 when
  debug/provenance-tracking is active. Do not assume it exists in production
  compiles.
- **`origin_node` is fragile.** Nulled on matmul buffers during pre-scheduling;
  not investigated to the exact pass because `origins` (7/7) strictly supersedes
  it as the carrier.
- **`traceback` is an unused slot.** The attribute *exists* on the IR node
  (confirmed via `hasattr` in the capture) but is never populated for this
  model — so it reads ❌ (present-able but empty) at the IR stages, not ➖. A
  profiler could populate it, but nothing does today.
- **Upstream `inductor_provenance_tracking_node_mappings.json` is not emitted**
  for the Spyre backend (`trace.provenance_tracking` is off by default and never
  set; Spyre's custom codegen bypasses the standard highlighter path). The
  `from_node` NodeSource chain and `get_kernel_metadata` provide the equivalent
  mapping in-process.

## 7. Field relationships (what derives from what)

The fields are not independent — several are *views* of `origins`. At the IR
stages, `origins` is the hub the rest hang off:

```text
origins  — OrderedSet[fx.Node]   (the IR-stage carrier; present 7/7)
│
├─ origin_node ........... one distinguished member of origins (a single fx.Node)
│                          — fragile: nulled on the matmul buffers
├─ get_stack_traces() .... DERIVED: origins[*].meta["stack_trace"]
│                          — empty when those origins have no stack_trace
└─ each fx.Node in origins carries its own .meta, reachable only via origins:
   ├─ stack_trace        ┐  Dynamo-set  (survive on anchor nodes;
   ├─ nn_module_stack    │               lost on mm/add decomposition products)
   ├─ source_fn_stack    ┘
   ├─ original_aten      ┐  Inductor-set, post-grad
   └─ from_node          ┘
```

**Why this matters for Phase 2.** Because every other field is reachable from
`origins` (and `get_stack_traces` is literally `origins` × `stack_trace`),
threading **`origins` alone** into the `OpSpec`/`debug_handle` makes all the rest
derivable downstream. `origin_node` and `get_stack_traces` add nothing `origins`
doesn't already contain — they are strictly children of it.

**On "clone paths" (the `traceback` populated-by note).** Some Inductor/Spyre
passes *clone* an IR node — e.g. the scratchpad planner duplicates a
`ComputedBuffer`/loop to insert a copy ([scratchpad/graph_editor.py](../../torch_spyre/_inductor/scratchpad/graph_editor.py)),
copying provenance onto the clone via
`new_loop._post_init_setattr("traceback", old_loop.traceback)` (and likewise for
`origins`/`origin_node`). Those duplication sites are the "clone paths." They
copy whatever the source held — and since no source node ever has `traceback`
set in this model, the clones inherit `None` too. So `traceback` stays an empty
but real slot (see §6).
