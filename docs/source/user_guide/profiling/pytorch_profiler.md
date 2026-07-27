# PyTorch Profiler on Spyre

**Stack:** torch-spyre (new, Inductor-based).

`torch.profiler.profile` is the entry point for per-op timing on Spyre.
Two modes are available:

1. **CPU-only** — no extra install; measures host-side Python and
   `torch.compile` activity.
2. **CPU + PrivateUse1** — measures CPU *and* Spyre-side kernel activity;
   requires the [`kineto-spyre`][kineto-spyre] PyTorch wheel.

## CPU-only (no extra install)

```python
import torch
from torch.profiler import profile, ProfilerActivity

compiled = torch.compile(model, backend="spyre")

with profile(activities=[ProfilerActivity.CPU]) as prof:
    output = compiled(x_spyre)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

This captures CPU wall-clock for every ATen call and every Dynamo /
Inductor stage.

## CPU + PrivateUse1

Install a matching [`kineto-spyre`][kineto-spyre] wheel for your
PyTorch version (check the [releases page][kineto-spyre-releases] for
the current combination). Example URL for PyTorch 2.10.0:

```bash
uv pip install --no-deps --force-reinstall \
  https://github.com/IBM/kineto-spyre/releases/download/torch-2.10.0.aiu.kineto.1.1.1/torch-2.10.0+aiu.kineto.1.1.1-cp312-cp312-linux_x86_64.whl
```

Then profile with `ProfilerActivity.PrivateUse1`:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    record_shapes=True,
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/mymodel"),
) as prof:
    compiled_result = compiled(x_device).cpu()
```

### Print aggregates

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10).replace("CUDA", "AIU"))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10).replace("CUDA", "AIU"))
```

The `.replace("CUDA", "AIU")` is a cosmetic workaround — the profiler's
internal column category is still named after CUDA; native renaming is
on the roadmap.

### Export a trace for viewers

```python
prof.export_chrome_trace("spyre_trace.json")
```

See [Trace analysis](trace_analysis.md) for viewing.

### Compiled-kernel provenance names

Compiled Spyre compute events use a versioned name that carries a stable
bundle identity:

```text
spyre_kernel_v1_<fused-aten-summary>_<52-character-key>#<step>
```

The key is the complete lowercase base32 encoding of a SHA-256 fingerprint
over the finalized `OpSpec` and `LoopSpec` bundle. Torch-Spyre shortens the
display-only ATen summary before the key, never the key itself, to fit the
AIUPTI event-name limit. Source paths and line numbers are not written in
plaintext, avoiding disclosure of private paths. The fingerprint does include
direct `debug_handle` IDs, which derive from source metadata; moving the same
model to a different path can therefore change the opaque key.

The name describes bundle-level attribution:

- Every `ComputeOnDevice` step produced from the bundle receives the same key.
  The `#<step>` suffix distinguishes commands; it does not claim that the
  proprietary backend assigned a particular subset of operations to that step.
- A compiler-generated provenance name deliberately replaces an existing
  SpyreCode compute label so every compute event retains the stable join key.
  Plans without a provenance name keep their previous labels and fallbacks.
- The associated descriptor lists only `debug_handle` IDs attached directly
  to finalized `OpSpec` records. Recursive `fused_from` records provide the
  constituent source and ATen lineage; the readable summary may use those
  constituents without adding their IDs to the direct list.
- A valid bundle with no handles still receives a key and uses
  `fused_unknown` as its display summary.

Phase 3a places only the join key in the trace. The trace alone cannot map that
key back to handles or source. Phase 3b will persist that mapping in the
`spyre_provenance.json` sidecar; consumers must pair the trace with that
artifact for durable source attribution. Native Kineto
`args.debug_handles` metadata is planned as an additive PyTorch 2.12 path, with
the v1 name retained for compatibility.

## Advanced features

Full reference lives in the upstream
[PyTorch profiler documentation][torch-profiler-docs]:

- `record_function` — annotate named spans
- `schedule` — skip warmup, sample a bounded window
- `on_trace_ready` — stream to TensorBoard-compatible JSON
- `with_stack` — include file and line for Python ops

## Known issues (from torch-spyre-docs)

- **Multi-AIU communication profiling is not supported yet.**

## See also

- [Trace analysis](trace_analysis.md) — viewers for the traces
- [Device monitoring](device_monitoring.md) — `aiu-smi` telemetry
  alongside `torch.profiler`

[kineto-spyre]: https://github.com/IBM/kineto-spyre
[kineto-spyre-releases]: https://github.com/IBM/kineto-spyre/releases
[torch-profiler-docs]: https://pytorch.org/docs/stable/profiler.html
