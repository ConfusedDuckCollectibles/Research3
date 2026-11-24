# Alignment numerical execution record

## Scope and frozen inputs

The design was published in commit `e7c467f2b097666dc474a3dbb6e273bf253e785f` before the numerical implementation or any study outcome. Its SHA256 is `c785e3fef3af9c8fac3bb7a04ccd7980ddfa51c121f3f11a9068eda3b7a70a5e`. The separately reviewed mathematical note has SHA256 `06fa7747aa96db8719ce7ae930789daee6291c65cab6752fcadb1364f0f6f2ff`.

This is a NumPy float64 mathematical study. It does not load a language model, execute the author's PyTorch implementation, establish bitwise equality with that implementation, or measure language-model quality. The referenced implementation is YanwenPneg/StateBridge at commit `3f6bf5442c6e8848555a6132516e6d36f35444fb`; the inspected `methods/state_bridge.py` has SHA256 `415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d`.

## Reviewed implementation

- `latent_bridge/alignment_core.py` implements the dense reference, efficient thin-SVD reference, guarded Gram formula, separately defined canonical completion, and matched postprocessing.
- `latent_bridge/alignment_packet.py` implements the specified 56-byte frame, rounded-Gram sender preflight, raw fallback, direct-prefix baseline, token-only baseline, and immutable session/table fingerprint.
- `latent_bridge/alignment_numerical.py` implements the fixed suites, timing order, required-case checks, and retained arrays.
- `tests/test_alignment_packet.py` contains 28 independently authored contract tests. They include manually specified wire bytes, a closed-form example, an actual float32 Gram-rounding boundary, malformed packets, and direct decoding without a vocabulary table.
- `tests/test_alignment_core.py` contains four focused checks for dewhitening, cross-rank geometry, required-support/error accounting, and exact timing coverage/order.
- `tests/test_alignment_execution.py` contains four operational tests for completion, whole-process-group timeout, preservation of existing gzip bytes, and rejection of contradictory compressed evidence.
- `scripts/run_alignment_numerical.py` supervises one child with a 300-second wall cap and inventories all completed and partial evidence.

Before execution, the matrix routines, codec, independent tests and supervisor received static review. Static review is not test execution; the actual CI gate and numerical results must be read from the original workflow.

## Execution and retention

The workflow `.github/workflows/alignment-numerical-v1.yml` checks out its exact source commit, installs Python 3.11.18 and NumPy 2.4.6, checks source syntax, runs the 36 focused tests, and then starts the numerical child. BLAS/OpenMP thread limits are set to one before numerical imports. The job cap is ten minutes. The supervisor permits only the first original workflow run and attempt; a correction requires a separately identified version with the original failure retained.

The child appends and flushes case and timing records. It checkpoints algebraic arrays before timing and preserves the largest timing fixture separately in the same retained array set. The supervisor keeps exact existing gzip bytes or creates an equivalent gzip only when absent, verifies decompression, inventories the directory, and exposes bounded recovery blocks in the job log. The complete output directory is uploaded even on failure, with 90-day retention. Permanent publication is a separate read-only audit/packaging step.

The source identity is the workflow's `GITHUB_SHA`, recorded in `launch_source_manifest.json` and `execution.json`. Those files also bind design hashes, source hashes, workflow/run identity, the cap, completion or failure, and the original retained-file inventory. The environment had no usable local executor at implementation time, so no local numerical results or local test passes are asserted.

## Interpretation rules

A successful process exit is insufficient for numerical support. Every required well-conditioned comparison must pass, unexpected unsupported cases must remain visible, and canonical nonunique outputs require feasibility and objective checks rather than an inappropriate equality claim. Timing is reported against both dense and thin-SVD references, after correctness checks. Packet results include receiver reconstruction, sender preflight, fallback, metadata, precision effects, and cold table requirements.

Established Procrustes, Bures and Gram geometry underlies the derivation. Any successful result is initially a computational/transport specialization for this particular interface. It does not by itself establish a new semantic language, a faster complete agent system, a novel general matrix theorem, or a breakthrough.
