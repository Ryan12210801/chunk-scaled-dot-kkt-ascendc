# Provenance and attribution notes

This repository contains the author's review-approved implementation for the July 2026 `ChunkScaledDotKkt` Ascend C challenge.

The referenced implementation is the **July 2026 champion submission** published in the public `cann/cann-ops-competitions` repository, under:

```text
01_official/cann-ops-ladder-2026/July/chunk_scaled_dot_kkt/submissions/dhltat/
```

Upstream repository: https://gitcode.com/cann/cann-ops-competitions

The borrowed scope is limited to task assignment continuity: selecting active MIX groups / `blockDim` for the relevant workloads and assigning adjacent chunks of one KV group contiguously to an execution group. It does not extend to the champion submission's complete kernel implementation.

All other major parts were developed independently in this project, including the mathematical-semantics analysis, `(chunk, kvHead)` Gram reuse, AIC/AIV 1:2 producer-consumer pipeline, READY/FREE workspace protocol, K128/K256 direct-MMAD implementation, Vector epilogue, exponential factorization, output buffering, local tests, and profiling utilities.

This repository is released under the MIT License. The attribution above records the exact conceptual boundary of the referenced scheduling strategy.

The files under `docs/slides/` are the author's original presentation artifacts. Historical profiling CSVs/logs under `results/` are included only as evidence of the optimization process and do not certify the final release baseline.
