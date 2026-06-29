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

"""Source-to-kernel provenance construction for the Spyre Inductor backend.

The provenance *types* (``SourceLoc``, ``DebugHandle``) live in ``op_spec.py``
alongside the other IR-op schema dataclasses. This module holds the *logic* that
builds them from Inductor IR: stable-id hashing here, and ``build_debug_handle``
(reading ``ComputedBuffer.origins``) in a later task.
"""

from __future__ import annotations

import hashlib

from torch_spyre._inductor.op_spec import SourceLoc


def _stable_id(
    source: SourceLoc | None,
    aten_op: str | None,
    ir_chain: tuple[str, ...],
) -> int:
    """Deterministic content hash. Reproducible across processes (unlike hash())."""
    canonical = "|".join(
        [
            source.to_str() if source is not None else "",
            aten_op or "",
            ",".join(ir_chain),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1  # 63-bit non-negative int
