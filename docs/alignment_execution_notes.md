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

## Original execution recovered

The original source commit is `7aee330f7799751a6a9f4b2a8decb612abe9c266`. Workflow [45369667584](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45369667584), job `138360349493`, completed its first run/attempt successfully. All 36 focused tests passed before the numerical child. The independent repository CI, run `45369667483`, job `138360348426`, passed 126 tests in 17.523 seconds with no skips, errors or failures.

The numerical child completed in 41.611000184 seconds including imports, with 15,759 required checks and zero failed required checks. All 242 algebraic, 2,304 applicability and 12 timing-validation records completed. All 2,052 timed calls, including warmups, were retained with exact frozen coverage and order. This is the original runner's result; the subsequent retained-array audit is documented below.

The recovered `execution.json` is 3,383 bytes with SHA256 `92212ae75c46bbd8c0f866b963801be39dd212d0efa737d19cd3adb4280a262d`. The recovered `results.json` is 141,386 bytes with SHA256 `8118b4994b5de1434e44dabf5692337c19112e469567be7c1c48d59a0ab51675`. Both were independently rehashed from their emitted log blocks. The original artifact is `14907778274`, named `alignment-numerical-45369667584-attempt-1`, with GitHub-reported ZIP digest `3032e4558d77c708f94aefd702cae803e6cb0d6b0dcecafcdb9ab656f2e1ffde` and size 11,479,164 bytes. The ZIP itself has not been independently rehashed here; permanent publication verifies every original file.

Runtime: Python 3.11.18, NumPy 2.4.6, OpenBLAS 0.3.31.188.0, Linux x86_64, with one BLAS/OpenMP thread. The manifest reports four available logical CPUs but does not identify the exact processor model. Hardware generalization is untested.

The [validation receipt](../results/alignment_validation.json) preserves the observed gate counts and retrieved-log hashes. Packaging must preserve the original 15-file inventory, replacing only duplicate raw case/timing JSONL in Git with their independently verified exact gzip representations. No fit, fixture generation, model inference or timing remeasurement is authorized by the packaging step.

## Independent audit and permanent publication completed

Publication run [45372752216](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45372752216), job `138370927774`, completed without rerunning the study. Commit `9bd33818226839934d04c8275d6e205672f83f6e` permanently retains the original evidence plus audit/provenance records. The [audit receipt](../results/alignment_numerical/audit_receipt.json) is 51,403 bytes with SHA256 `7c3cade741f70f108590342b0e924bfe3d304d36cb82e23f685b86ef8ef3a88d`. Its integrity, report and scientific issue arrays are empty.

The audit covers 861 retained arrays mapped to 243 cases, with independent matrix feasibility, proper-objective and appropriately scoped unique-product checks, plus recorded timing order, coverage and summaries. It does not independently rerun packet decoding, postprocessing, float32 diagnostics or the numerical measurements. The [human report](alignment_results.md) states these boundaries.

A subsequent [read-only record review](../results/alignment_postpublication_review.json) verified the exact decompressed case/timing hashes, recomputed all 144 composed phase totals from recorded medians without mismatch, and surfaced the 35.89% descriptive float32 error in an accepted nonunique canonical packet. This evidence is not a new empirical run. The original required-check result remains unchanged, and must not be interpreted as a float32 fidelity guarantee.
