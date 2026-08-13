# Provenance

```{toctree}
:hidden:
:maxdepth: 2

viewer
cross_stage_comparison
```

Torch-Spyre provenance connects a compiled Spyre profiler activity to the
recorded Python source locations, ATen identities, FX nodes, lower-IR lineage,
and direct OpSpec bindings that contributed to its finalized bundle.

The rich evidence lives in `spyre_provenance.json`. A profiler event carries
only a compact key that joins the trace to that sidecar. Use the
[offline provenance viewer](viewer.md) to inspect the relationship as a
self-contained six-panel HTML file.

The [compile-variant comparison guide](cross_stage_comparison.md) documents the
source-cohort workflow, correlation-first workload annotations, count
semantics, and evidence limits used by the Phase 4b Granite analysis.

The viewer is designed for saved-artifact analysis. It does not need a Spyre
device, compiler process, backend service, or `tlparse` installation after the
HTML has been generated.
