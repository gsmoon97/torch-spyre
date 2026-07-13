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

"""Device integration test: provenance survives a real Spyre compile end-to-end."""

import os
import warnings

import pytest
import regex  # noqa: F401  (repo convention: never import re)
import torch


def _spyre_available() -> bool:
    try:
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME

        torch.zeros(1, dtype=torch.float16, device=DEVICE_NAME)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _spyre_available(), reason="requires an available Spyre device"
)


class _MLP(torch.nn.Module):
    # Stick-aligned dims (multiples of the 64-element fp16 stick): compiles and
    # runs end-to-end without padding. Mirrors the provenance example
    # reference_mlp (SimpleMLP(128, 256, 128)).
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_handles_survive_real_compile(monkeypatch):
    import torch_spyre  # noqa: F401
    from torch_spyre.constants import DEVICE_NAME
    import torch_spyre._inductor.provenance as prov
    import torch_spyre._inductor.spyre_kernel as sk

    collected = []
    _orig = prov.build_debug_handle

    def _collect(buffer):
        h = _orig(buffer)
        collected.append(h)
        return h

    # spyre_kernel imported build_debug_handle by name, so patch it there too.
    monkeypatch.setattr(prov, "build_debug_handle", _collect)
    monkeypatch.setattr(sk, "build_debug_handle", _collect)

    # Defeat Inductor's on-disk FX graph cache so codegen (and therefore
    # build_debug_handle) actually runs this process. Same pattern as the
    # provenance audit tooling (audit.py): a cache hit silently skips
    # create_op_spec/define_kernel, which would make this test flaky across
    # repeated runs with identical dims rather than a genuine provenance signal.
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    torch._dynamo.reset()

    model = _MLP().half().to(DEVICE_NAME).eval()
    x = torch.randn(2, 128, dtype=torch.float16, device=DEVICE_NAME)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            torch.compile(model)(x)

    # (a) No pass dropped provenance (observer emitted no drop warnings).
    drops = [w for w in caught if "spyre-provenance" in str(w.message)]
    assert not drops, (
        f"observer reported provenance drops: {[str(w.message) for w in drops]}"
    )

    # (b) At least one handle resolved to a real source line (the matmul traces
    #     back to the model via the linear's weight-transpose origin).
    resolved = [
        h
        for h in collected
        if h is not None and h.source is not None and h.aten_op is not None
    ]
    assert resolved, (
        "no debug_handle resolved to a source; provenance did not reach the kernel"
    )
    # The resolved source should point at this test module (the model's forward).
    this_file = os.path.basename(__file__)
    assert any(h.source.file.endswith(this_file) for h in resolved)
