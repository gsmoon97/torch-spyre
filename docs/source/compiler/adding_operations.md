# Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the front-end compiler.

## Direct mapping from ATen to OpFunc

If a pointwise ATen operation can be implemented with a single Spyre OpFunc,
then enabling it in our backend only requires
adding a method to `SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py).
Canonical examples are `add` and `softplus` (see `softplus` for an example of using `op_info` for non-tensor arguments).

Note that some pointwise ATen operations that can be be implemented with a single Spyre OpFunc
have default decompositions defined by Inductor. Adding a method to
`SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py)
overrides the default decomposition and thus enables the desired direct mapping.
Canonical examples are `reciprocal` and `sigmoid`.

## Spyre-specific decompositions

We define Spyre-specific decompositions in [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
using the `@register_spyre_decomposition` decorator.  Decompositions are graph transformations
that are performed before the graph is lowered to loop level IR.

## Spyre-specific lowerings

We define Spyre-specific lowerings from ATen operations to Inductor's
loop level IR in [lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py) using the `@register_spyre_lowering` decorator.

## Spyre-specific OpFuncs

For Spyre OpFuncs that do not have corresponding ATen operations, we use
the `@torch.library.custom_op` decorator to define a new operation in
[customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py). This has two pieces:
+ defining the signature of the operation (using `@custom_op`)
+ defining its fake function (using the `@opname.register_fake` that is defined as part of the `@custom_op`)

In addition, when defining a custom op, you will also need to do one of:
+ register a lowering for the custom op in [lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py) and
  add a method to `SpyreOpFuncs` in [spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py).
  A canonical example is `spyre.clamp`.
+ register a decomposition for the custom op in [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
  that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.
+ define a `CustomPrePass` or `CustomPostPass` that implements a more general graph
  rewrite that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.

## Custom ops as CPU fallbacks

When an ATen operation cannot run natively on Spyre for certain dtypes but
can be decomposed on Spyre for others, we use a custom op to route the
unsupported cases to a CPU fallback. The pattern is:

1. Define a custom op in
   [customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py)
   with `@torch.library.custom_op` and a `register_fake` that returns the
   expected output shape/dtype. The implementation body raises `RuntimeError`
   (it is never called directly).
2. Register a Spyre-specific decomposition in
   [decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py)
   that dispatches to either a native Spyre path or the custom fallback op
   based on dtype or other conditions.
3. Register a CPU fallback for the custom op in
   [fallbacks.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/ops/fallbacks.py)
   using `@register_fallback`.

Canonical examples are `spyre::max_dim_int64_fallback` and
`spyre::min_dim_int64_fallback`, which fall back to CPU for int64 reductions
while the fp16/fp32 cases run natively on Spyre via decomposition.

## Modifying existing kernels: wrap, never reconstruct

When a compiler pass needs to alter how an existing `ComputedBuffer` computes
its values, always wrap the original `inner_fn` using a `WrapperHandler`
subclass — never attempt to rebuild `inner_fn` from scratch.

**Why:** `inner_fn` index expressions are symbolic; they are bound to the
specific `sympy` objects created during lowering. Rebuilding the function from
scratch produces fresh symbolic expressions that are structurally similar but
not the same objects, causing silent wrong-code bugs that are hard to detect
(see issue [#2797](https://github.com/torch-spyre/torch-spyre/issues/2797)).

**Correct pattern:**

```python
class _MyHandler(WrapperHandler):
    def load(self, name, index):
        # intercept specific loads; delegate everything else
        return super().load(self._name_map.get(name, name), index)

def new_inner_fn(*args, _orig=orig_inner):
    with V.set_ops_handler(_MyHandler(V.ops, ...)):
        return _orig(*args)
```

Canonical implementations: `NameSwapHandler` in
[insert_restickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/insert_restickify.py),
`_SplitOpsHandler` and `_IntermediateOpHandler` in
[split_multi_ops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/split_multi_ops.py).

## Preserving provenance in passes

Every op carries a `debug_handle`, built at codegen from the buffer's `origins`,
that ties the emitted kernel back to a source line (and, once profiler
integration lands, a profiler event). A pass that creates or rewrites a
`ComputedBuffer` must carry `origins` (and `origin_node`) onto the new buffer, or
that op's `debug_handle` will have no source.

Reuse the existing helpers rather than setting `origins` by hand:

+ `replace_computed_buffer_body(op, new_data, operations)` in
  [pass_utils.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/pass_utils.py)
  when reconstructing a buffer's body: it forwards `operation_name`, `origins`,
  and `origin_node`.
+ `copy_op_metadata(src, dst)` in
  [loop_info.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/loop_info.py)
  to carry Spyre metadata (including the fusion-context tag) across a
  reconstructed buffer.
+ `preserve_provenance`, `merge_provenance`, and `decompose_provenance` in
  [provenance.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/provenance.py)
  at explicit rewrite sites:
  + `preserve_provenance(old, new)` for a 1-to-1 rewrite,
  + `merge_provenance(sources, new, context)` when fusing several buffers into one
    (unions their `origins` and records why),
  + `decompose_provenance(old, news, context)` when splitting one buffer into many
    (each output inherits the parent's `origins`).

`SpyreGraphTransformObserver` wraps every pass in the node and pre-scheduling
pipelines and emits a warning when a pass clears an existing buffer's `origins` or
creates a buffer without any. If you add a pass and see a
`[spyre-provenance] pass '<name>' ...` warning, forward provenance with one of the
helpers above.

Some passes legitimately create buffers with no user source (for example padding
via `constant_pad_nd`). Those pass names are listed in `COMPILER_GENERATED_PASSES`
in `provenance.py` and are not warned; add a pass there only after confirming its
new buffers genuinely have no source. Set `TORCH_SPYRE_PROVENANCE=0` to disable
the observer entirely.
