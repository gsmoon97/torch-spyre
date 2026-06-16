# Provenance Audit: `SimpleMLP` — Metadata Across the Compilation Pipeline

> Generated: 2026-06-16 18:43 &nbsp;|&nbsp; Issue: [torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574)

Measured in-process during one cache-defeated `torch.compile` (compile-path objects only). This report is **measurement-only**; interpretation is a separate deliverable.

| Quantity | Value |
| --- | --- |
| FX pre-grad compute nodes | 3 |
| FX post-grad compute nodes | 7 |
| LoopLevelIR operations | 7 |
| OpSpec ops created | 7 |
| SuperDSC kernels | 2 |
| `sdsc_*.json` files | 7 |
| `OpSpec` declared fields | `['op', 'is_reduction', 'iteration_space', 'args', 'op_info', 'tiled_symbols']` |

## Stage × Field Matrix

✅ present (all) &nbsp; ◐ present on some (n/total) &nbsp; ❌ present-able here but not carried (a genuine drop, or an empty slot) &nbsp; ➖ not applicable (not generated yet, or carried only indirectly via `origins`).

The **Layer** column marks whether a field lives on the FX node (`FX`) or the IR `ComputedBuffer` (`IR`). *Inductor passes* is the IR entering pre-scheduling; *LoopLevelIR* is the same IR after the passes — they mutate it in place and insert 2 `restickify` buffers (5 → 7 ops), so `origin_node` reads ✅ (5/5) → ◐ 5/7 (nulled on the matmuls).

| Layer | Field | FX pre-grad | FX post-grad | Inductor passes | LoopLevelIR | OpSpec | SuperDSC JSON |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FX | `stack_trace` | ✅ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `nn_module_stack` | ◐ 2/3 | ◐ 2/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `source_fn_stack` | ✅ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| FX | `original_aten` | ➖ | ✅ | ➖ | ➖ | ➖ | ➖ |
| FX | `from_node` | ➖ | ◐ 3/7 | ➖ | ➖ | ➖ | ➖ |
| IR | `origins` | ➖ | ➖ | ✅ | ✅ | ❌ | ❌ |
| IR | `origin_node` | ➖ | ➖ | ✅ | ◐ 5/7 | ❌ | ❌ |
| IR | `traceback` | ➖ | ➖ | ❌ | ❌ | ❌ | ❌ |
| IR | `get_stack_traces` | ➖ | ➖ | ◐ 3/5 | ◐ 3/7 | ❌ | ❌ |

> ➖ for FX fields downstream of FX post-grad means the value is carried indirectly — FX-meta is reachable through `origins` (which points back to the FX nodes), not as a direct attribute of the IR objects. The genuine drop is the ✅/◐ → ❌ at OpSpec: the `OpSpec` dataclass declares no provenance field (`['op', 'is_reduction', 'iteration_space', 'args', 'op_info', 'tiled_symbols']`), so `origins` and its derivatives are not carried into OpSpec or the emitted JSON. `traceback` reads ❌ throughout the IR stages — the attribute exists but is never populated (an empty slot).

## Stage 2 — FX pre-grad (3 compute nodes)

Cell = observed `type` of the field, or ❌ if absent.

| Node | target | `stack_trace` | `nn_module_stack` | `source_fn_stack` | `original_aten` | `from_node` | source line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `x` | `<built-in function linear>` | `str` | `dict` | `list` | ❌ | ❌ | `x = self.fc1(x)` |
| `x_1` | `<built-in method relu of type object at 0x7f3241dc78a0>` | `str` | ❌ | `list` | ❌ | ❌ | `x = torch.relu(x)` |
| `x_2` | `<built-in function linear>` | `str` | `dict` | `list` | ❌ | ❌ | `x = self.fc2(x)` |

## Stage 2 — FX post-grad (7 compute nodes)

Cell = observed `type` of the field, or ❌ if absent.

| Node | target | `stack_trace` | `nn_module_stack` | `source_fn_stack` | `original_aten` | `from_node` | source line |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `permute` | `aten.permute.default` | `str` | `dict` | `list` | `OpOverload` | `list` | `x = self.fc1(x)` |
| `mm_default_1` | `aten.mm.default` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `add_tensor_1` | `aten.add.Tensor` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `relu` | `aten.relu.default` | `str` | ❌ | `list` | `OpOverload` | `list` | `x = torch.relu(x)` |
| `permute_1` | `aten.permute.default` | `str` | `dict` | `list` | `OpOverload` | `list` | `x = self.fc2(x)` |
| `mm_default` | `aten.mm.default` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |
| `add_tensor` | `aten.add.Tensor` | ❌ | ❌ | ❌ | `OpOverload` | ❌ | — |

## Stages 3–4 — LoopLevelIR operations (7)

Measured `ComputedBuffer` attributes after the pre-scheduling passes.

| Op | `origins` | `origin_node` | `traceback` | `get_stack_traces` |
| --- | --- | --- | --- | --- |
| `op5` | `restickify_default` | `restickify_default` | ❌ | ❌ |
| `op0` | `mm_default_1`, `permute` | ❌ | ❌ | ✅ |
| `op1` | `add_tensor_1` | `add_tensor_1` | ❌ | ❌ |
| `op2` | `relu` | `relu` | ❌ | ✅ |
| `op6` | `restickify_default_1` | `restickify_default_1` | ❌ | ❌ |
| `op3` | `mm_default`, `permute_1` | ❌ | ❌ | ✅ |
| `op4` | `add_tensor` | `add_tensor` | ❌ | ❌ |

## Stage 5 — OpSpec ops (7)

`OpSpec` declared fields: `['op', 'is_reduction', 'iteration_space', 'args', 'op_info', 'tiled_symbols']` — no provenance field, so the matrix shows `OpSpec` as ➖. The `origins` below are what is *available on the input `ComputedBuffer`* at `create_op_spec` (what a Phase-2 `debug_handle` could capture); the `OpSpec` object itself stores none of them.

| Spyre op | buffer | `origins` | `origin_node` |
| --- | --- | --- | --- |
| `ReStickifyOpHBM` | `op5` | `restickify_default` | `restickify_default` |
| `batchmatmul` | `op0` | `mm_default_1`, `permute` | ❌ |
| `add` | `op1` | `add_tensor_1` | `add_tensor_1` |
| `relufwd` | `op2` | `relu` | `relu` |
| `ReStickifyOpHBM` | `op6` | `restickify_default_1` | `restickify_default_1` |
| `batchmatmul` | `op3` | `mm_default`, `permute_1` | ❌ |
| `add` | `op4` | `add_tensor` | `add_tensor` |

## Stage 6 — SuperDSC kernels (2)

Provenance field present in any emitted `sdsc_*.json`: ❌

### `sdsc_fused_addmm_linear_relu_0`

- buffers (6): `op5`, `op0`, `op1`, `op2`, `op6`, `op3`
- fx origins: `add_tensor_1`, `mm_default`, `mm_default_1`, `permute`, `permute_1`, `relu`, `restickify_default`, `restickify_default_1`
- kernel metadata: `# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.linear, aten.addmm, aten.relu]`
- `sdsc_*.json` files: 6 &nbsp; provenance in JSON: ❌

### `sdsc_fused_addmm_1`

- buffers (1): `op4`
- fx origins: `add_tensor`
- kernel metadata: `# Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]`
- `sdsc_*.json` files: 1 &nbsp; provenance in JSON: ❌
