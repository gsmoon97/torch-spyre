# Compare provenance across compile variants

Presentation version 2 extends the offline Spyre provenance viewer with a
source-first comparison mode. It does not change **spyre_provenance.json**
version 1 or the runtime-event inspector introduced in presentation version 1.

## Compile variants are not compile calls

A sidecar **compileId** is a content hash over the finalized wrapper's ordered
compiler-kernel names and bundle identities. It identifies one compile-content
variant. It is not an invocation ID, timestamp, or ordinal, and equal content
from several `torch.compile` calls can collapse to the same ID.

Consequently, the viewer never labels sidecar-only groups as the first, second,
or nth compile call. Without trace evidence it orders variants by stable
**compileId** and displays the ID prefix. When a workload adapter establishes a
one-to-one relation between variants and chronological trace ranges, the viewer
may display **Compile variant 1**, **Compile variant 2**, and so on. That number
orders observed ranges, not compile invocations.

## Generate a generic comparison

A sidecar is sufficient to compare compile-content membership:

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python -m torch_spyre.provenance_viewer \
      /path/to/spyre_provenance.json \
      --compare-compiles \
      --output /path/to/spyre_provenance_viewer.html

This uses stable compile-ID order and makes no workload-stage claim. A Kineto
trace remains optional for runtime-event inspection.

## Add workload annotations

For a Granite capture with the sidecar and Kineto trace from the same run:

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    python -m torch_spyre.provenance_viewer \
      /path/to/spyre_provenance.json \
      --kineto-trace /path/to/kineto_trace.json \
      --phase-adapter granite \
      --output /path/to/spyre_provenance_viewer.html

The adapter implies compile comparison. It recognizes only these anchored
complete-event labels:

    granite.iteration.<iteration>.prefill
    granite.iteration.<iteration>.decode.<index>

Outer **granite.iteration.<iteration>** ranges are retained separately. Similar
but malformed labels are diagnosed and do not receive semantic meaning.
**Prefill** and **Decode N** are annotations supplied by this adapter; they are
not built into the comparison schema.

Without **--compare-compiles** or **--phase-adapter**, the generator preserves
the presentation version 1 event inspector and its DOM behavior.

## Use the two viewer modes

**Explore runtime event** retains the profiler-event selector, runtime
occurrence selector, compact event facts, six evidence panels, typed row
highlighting, and automatic cross-panel focus.

**Compare compile variants** starts from a source cohort:

1. Filter by source path or range, ATen identity, exact post-grad FX origin,
   evidence basis, or a pattern such as `1 / 2 / 1`.
2. Select one cohort. Divergent returning patterns are listed first.
3. Read each column's global bundle, runtime-observation, and
   bundle-plus-JobPlan-step counts separately.
4. Read the selected cohort's bundle, operation-member, and observation counts.
5. Select a bundle to open its exact runtime occurrence in the six-panel
   inspector.

A column is one compile-content variant. A row inside the selected result is one
finalized bundle. Source, ATen, FX, lower-IR, and SpecPath evidence stays in the
inspector rather than being repeated at mixed granularities in the comparison
cards.

## Source-cohort evidence

The comparison recursively expands each ordered direct SpecPath binding to its
leaf debug handles and preserves repeated membership. Matching uses this
precedence:

1. Reuse an exact debug-handle ID when it appears in several variants.
2. Otherwise derive a presentation-only operation candidate from structured
   **SourceLoc**, ATen identity, and exactly one exact compile-scoped post-grad
   FX origin.
3. Keep a handle in an explicit exact-handle cohort when the candidate cannot be
   derived safely.

Source plus ATen alone is never accepted because repeated model layers can
share both. Generated FX names are resolved inside their compile scope before
they participate in a candidate. Multiple handles sharing one candidate remain
visible as a collision; they are not silently deduplicated.

The bundle pattern is the number of unique finalized bundle identities
containing the selected cohort in each displayed variant. Operation-member
multiplicity is a separate count. A pattern such as `1 / 3 / 1` describes
membership overlap; it does not prove a causal compiler split edge.

## Phase assignment rules

The Granite annotation is correlation-first:

1. Index **aiuLaunchControlBlocks** activities by numeric correlation ID.
2. For a kernel observation with one matching launch, use the launch timestamp.
3. Select the innermost supported range containing that timestamp.
4. Use kernel start only when no usable matching launch exists.
5. Keep duplicate launches or equally specific ranges ambiguous.
6. Keep timestamps outside every supported range unassigned.

The asynchronous kernel interval need not finish inside the host range. Event
order, nearest range, duration overlap, bundle identity, event name, and
JobPlan suffix are never used to infer a stage.

## Count definitions

- **Bundle identity** is a unique finalized provenance key.
- **Runtime observation** is one concrete resolved Kineto activity.
- **Bundle + JobPlan step** is one unique bundle identity and static backend
  command-index pair.
- **Operation member** is one preserved leaf-handle occurrence reached from an
  ordered direct SpecPath binding.

These values are not token latency or hardware utilization. A JobPlan suffix is
a static command position for the complete finalized bundle; it does not
identify an OpSpec subset.

## Retained Granite result

The preserved capture resolves all 83 observations. Unique runtime-launch
correlations annotate its three variants as Prefill, Decode 1, and Decode 2,
with global bundle counts of `1 / 81 / 1`.

The current exact-first cohort derivation finds 5,147 source cohorts. Forty have
a `1 / 2 / 1` bundle pattern, and none has `1 / 3 / 1`. This is a measured
result, not evidence that the reported behavior never occurred. It can differ
because the retained model revision, inputs, compiler behavior, or reported
cohort definition differs. The viewer must not broaden a cohort after reading
the counts merely to force the reported pattern.

Provenance identifies recorded membership and the earliest available divergence
boundary. It does not record the scheduler or proprietary backend decision that
caused a bundle boundary. Missing causal evidence must remain unavailable rather
than being inferred from final membership.
