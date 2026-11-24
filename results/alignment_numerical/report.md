# Alignment numerical study

Status: **verified**. Stage: complete.

This is a float64 numerical study of an alignment interface, not a language-model experiment.
The canonical method is a separate optimal completion when the aligned product is nonunique.

## Coverage and mathematical checks

Required checks: 15759; failed required checks: 0.
An unexpected unsupported required method counts as failed coverage. Stress inputs remain descriptive.

| Suite | Expected | Completed |
|---|---:|---:|
| A | 120 | 120 |
| B | 27 | 27 |
| C | 8 | 8 |
| D | 31 | 31 |
| E | 56 | 56 |
| F | 2304 | 2304 |
| T | 12 | 12 |

## Alignment-core timing

Table entries are medians of eight within-repetition ratios on one CPU host. A ratio above one favors the named Gram method.
Both the dense and standard thin-SVD baselines are retained; a gain over dense alone is not new fast Procrustes mathematics.

| Fixture | Dual vs dense | Dual vs thin | Canonical vs dense | Canonical vs thin |
|---|---:|---:|---:|---:|
| T-k8-d128 | 13.710 | 1.166 | 10.550 | 0.908 |
| T-k8-d384 | 163.604 | 1.430 | 108.351 | 0.970 |
| T-k8-d768 | 1310.963 | 1.778 | 711.679 | 1.081 |
| T-k8-d1024 | 3129.528 | 2.139 | 1699.292 | 1.178 |
| T-k32-d128 | 6.753 | 1.486 | 4.629 | 1.033 |
| T-k32-d384 | 66.143 | 1.968 | 35.082 | 1.052 |
| T-k32-d768 | 432.723 | 2.729 | 183.203 | 1.171 |
| T-k32-d1024 | 971.776 | 2.961 | 375.916 | 1.171 |
| T-k64-d128 | 2.912 | 1.506 | 2.010 | 1.065 |
| T-k64-d384 | 25.773 | 2.044 | 13.982 | 1.100 |
| T-k64-d768 | 169.529 | 2.643 | 73.748 | 1.159 |
| T-k64-d1024 | 393.576 | 3.016 | 156.781 | 1.179 |

Complete raw timings, warmups, size ledgers, and failures are in timings.jsonl.
The JSON report separately sums direct core + common postprocessing + direct encode/decode,
and Gram encoding (already including sender preflight/postprocessing) + receiver decode.
It does not add the core a second time to the Gram route. These arithmetic sums exclude network transfer and LLM inference.

## Packet accounting and precision

The frame header is 56 bytes. A Gram packet additionally sends K uint32 token IDs and
K(K-1)/2 floating-point values. Direct prefixes send Kd values; token-only packets send K IDs.
Each case records exact bytes, float32 Gram error, matched direct-prefix rounding error,
snap/nearest-ID changes, and sender-side fallback costs.
The Gram and token-lookup decoders require the shared vocabulary; direct-prefix decoding
is also exercised with a context containing no vocabulary. Cold vocabulary/setup cost is additional.

## Rank, limits, and interpretation

Full-rank equality, canonical Gram/objective validity, and close-margin vocabulary behavior
are reported separately. A canonical-versus-dense difference in a nonunique case is expected.
The ideal weak-regularization limits and the fixed 1e-6 covariance floor are separate tracks.
Synthetic duplicate-token rates do not estimate natural-language applicability.
No result establishes a new semantic language, model quality, end-to-end model speed, or lower bandwidth than text.

## Evidence

cases.jsonl retains every algebraic, token-applicability, and timing-validation row.
arrays.npz and arrays_manifest.json retain all algebraic inputs and available pre-calibration
outputs plus the largest timing fixture. Smaller timing fixtures have fixed seeds and input hashes.
The completed algebraic archive is checkpointed before timing begins.
Frozen design: e7c467f2b097666dc474a3dbb6e273bf253e785f; SHA256 c785e3fef3af9c8fac3bb7a04ccd7980ddfa51c121f3f11a9068eda3b7a70a5e.
