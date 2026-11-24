# Alignment results: a faster core, with transport and precision limits

The guarded Gram formulation was **1.17–3.02× faster than a standard thin-SVD implementation** for the alignment core across the twelve fixed CPU timing fixtures. All **15,759 required checks passed**. This supports a bounded implementation result: the chosen alignment calculation can be expressed and evaluated efficiently through small sample-space matrices. It does not establish new Procrustes mathematics, better model answers, or faster end-to-end language-model inference. [Original results](../results/alignment_numerical/results.json), [frozen design](alignment_numerical_design.md).

The complete packet comparison is less favorable. Sending a Gram statistic and reconstructing the prefix required **1.86–2.29× the arithmetic time** of computing the prefix once with the same fast dual core and sending that prefix directly. The smaller payload may still matter on a constrained link, but this study did not measure a network transaction. A substantial precision limit also appeared: one accepted float32 canonical packet changed its final vector by **35.89% relative Frobenius norm**, despite leaving its nearest-vocabulary IDs unchanged. Float32 reconstruction quality was a descriptive measurement, not one of the required equivalence gates. [Timing records](../results/alignment_numerical/timings.jsonl.gz), [case records](../results/alignment_numerical/cases.jsonl.gz).

## What was evaluated

The study isolates the centered whitening, orthogonal alignment, dewhitening, normalization and vocabulary anchoring interface used by the pinned [StateBridge implementation](https://github.com/YanwenPneg/StateBridge/blob/3f6bf5442c6e8848555a6132516e6d36f35444fb/methods/state_bridge.py). Inputs are controlled synthetic matrices with \(K\) rows and \(d\) features. No pretrained model was loaded or evaluated by the numerical study. Its float64 reference preserves the specified covariance eigenvalue floor and processing order; it does not claim bitwise identity with the upstream float32 implementation. [Design](alignment_numerical_design.md), [runtime manifest](../results/alignment_numerical/runtime_manifest.json).

Four alignment implementations were compared:

| Implementation | Purpose and scope |
|---|---|
| Dense reference | Direct feature-space covariance decompositions and proper orthogonal Procrustes. |
| Thin SVD | A standard \(O(K^2d+K^3)\) baseline in the supported full centered-rank regime; both input decompositions are charged to each invocation. |
| Guarded dual | A Gram-based \(O(K^2d+K^3)\) calculation of the unique aligned product, requiring full centered row rank and \(2\leq K<d\). |
| Canonical variant | A deterministic optimal completion for supported deficient-rank cases. Where the aligned product is nonunique, this is a specified separate output rather than a promise to reproduce the reference SVD's arbitrary completion. |

The dual result uses a \((K-1)\)-dimensional basis for centered rows and an SPD matrix \(M\) determined by the whitened source and reference Grams. The pre-calibration output can then be reconstructed as \(U M(U^\top R)+\mathbf{1}\mu_R^\top\), without a \(d\times d\) alignment matrix. The derivation and rank assumptions are in [Alignment geometry](alignment_geometry.md). The thin-SVD baseline has the same asymptotic complexity, so a large advantage over the deliberately dense reference is not evidence of a newly discovered fast Procrustes algorithm.

## Correctness and coverage

The complete run retained **2,558 case records**: **242 algebraic fixtures**, **2,304 synthetic token-applicability samples**, and **12 timing-fixture validations**. All frozen required checks completed with zero failures; an unexpected unsupported required method would have counted as a failure. Near-rank stress and float32 output errors were explicitly kept descriptive. [Original results](../results/alignment_numerical/results.json).

| Suite | Cases | What it tests |
|---|---:|---|
| A | 120 | Well-conditioned full-rank agreement over fixed sizes, seeds, spectra and regularization values. |
| B | 27 | Exact deficient-rank geometries, including zero and duplicate directions. |
| C | 8 | Near-rank numerical stress outside the primary equivalence claim. |
| D | 31 | Invariances, information collisions, structural boundaries, bypasses and postprocessing controls. |
| E | 56 | Weak-regularization behavior, keeping the ideal and fixed-floor cases separate. |
| F | 2,304 | Reference-side rank eligibility under synthetic token sampling. |
| T | 12 | Correctness and actual serialized-packet checks on every timed fixture. |

In suite A, the largest observed pre-calibration Frobenius difference from the dense reference was approximately **\(2.85\times10^{-12}\)** for the dual and thin implementations; the canonical variant was approximately **\(2.86\times10^{-12}\)**. The predeclared acceptance rule was \(\|E\|_F\leq10^{-10}+10^{-8}\|Y_{\mathrm{reference}}\|_F\). The canonical variant passed all 27 exact deficient-rank fixtures, using Gram and optimal-objective checks where output equality is not uniquely defined. The dual and thin implementations rejected those 27 fixtures as required by their narrower contract. [Original results](../results/alignment_numerical/results.json), [case records](../results/alignment_numerical/cases.jsonl.gz).

The timing claim is limited to the validated well-conditioned fixtures. Forming a Gram squares singular-value conditioning: for full-row-rank \(A\), \(\kappa_2(AA^\top)=\kappa_2(A)^2\). This can make small directions harder to resolve numerically. In the eight near-rank stress cases, the fixed guards admitted 2 dual cases, 1 thin-SVD case and 1 canonical case; the dense reference ran on all 8. Those are observed support decisions under the declared guards, not evidence that the dual method is universally more stable. Stress outputs remain outside the primary exact-equivalence claim. [Stress records and support counts](../results/alignment_numerical/results.json).

The structural and limit fixtures are also scope checks. Preserving a Gram statistic does not preserve every feature of a sender representation; distinct source states can produce the same aligned output. Likewise, an ideal zero-regularization limit must not be substituted for the behavior of code with a fixed \(10^{-6}\) eigenvalue floor. These results concern the specified transformation, not semantic information preservation by an arbitrary agent communication channel. [Mathematical derivation](alignment_geometry.md), [case records](../results/alignment_numerical/cases.jsonl.gz).

## Alignment-core timing against the strong baseline

Each size used one fixed validated input, one warmup and eight measured repetitions. The order of the four core implementations rotated so that each appeared first twice. Execution used one BLAS thread on one CPU host. The speed column below is the median of eight paired thin/dual ratios; it is not the ratio of the two displayed median times. [Frozen timing protocol](alignment_numerical_design.md), [raw timing records](../results/alignment_numerical/timings.jsonl.gz).

| \(K\) | \(d\) | Thin SVD median, ms | Dual median, ms | Paired thin/dual speed ratio |
|---:|---:|---:|---:|---:|
| 8 | 128 | 0.242 | 0.203 | 1.17× |
| 8 | 384 | 0.336 | 0.220 | 1.43× |
| 8 | 768 | 0.490 | 0.235 | 1.78× |
| 8 | 1,024 | 0.624 | 0.265 | 2.14× |
| 32 | 128 | 0.728 | 0.479 | 1.49× |
| 32 | 384 | 1.136 | 0.566 | 1.97× |
| 32 | 768 | 2.112 | 0.766 | 2.73× |
| 32 | 1,024 | 2.508 | 0.836 | 2.96× |
| 64 | 128 | 1.944 | 1.269 | 1.51× |
| 64 | 384 | 3.131 | 1.546 | 2.04× |
| 64 | 768 | 4.961 | 1.861 | 2.64× |
| 64 | 1,024 | 6.189 | 2.102 | 3.02× |

The canonical variant's paired ratio against thin SVD ranged from **0.91× to 1.18×** on these full-rank timing inputs: it was sometimes slower and offered at most a modest gain. Its reason for inclusion is its explicit deficient-rank completion, not a general timing advantage. Repetitions on one machine quantify this implementation on these fixtures; they are not confidence intervals for other CPUs, model workloads or vocabulary sizes. [Original timing summaries](../results/alignment_numerical/results.json).

## Smaller packets cost more computation in this implementation

The direct route charges the selected alignment core, one common postprocessing call, and direct encoding and decoding. The Gram route charges sender encoding, which already includes the rounded-packet preflight and postprocessing, followed by receiver reconstruction and postprocessing. No extra core is added a second time to the Gram total. The totals are sums of recorded component medians, not measured complete network latencies. [Accounting implementation](../latent_bridge/alignment_numerical.py), [timing records](../results/alignment_numerical/timings.jsonl.gz).

For the largest fixture, \(K=64,d=1024\), with 320 synthetic vocabulary rows:

| Route or phase | Arithmetic time, ms |
|---|---:|
| Thin SVD + direct float64 prefix | 8.363 |
| Dual + direct float64 prefix | **4.276** |
| Canonical + direct float64 prefix | 7.441 |
| Dual Gram encoding, including sender preflight | 4.456 |
| Dual Gram receiver decoding | 3.974 |
| Dual Gram total | **8.430** |
| Canonical Gram total | 14.722 |

Thus the dual Gram route used about **1.97×** the arithmetic time of dual plus direct transmission at this size. Across all twelve sizes the factor was **1.86–2.29×**. Sender preflight and receiver decoding both reconstruct and postprocess the prefix; this duplication is included in the ledger. The strong comparison is with the efficient dual-direct route, rather than with sending a prefix produced by an unnecessarily expensive dense implementation. No network delay, transfer bandwidth or LLM inference time was measured here. [Original results](../results/alignment_numerical/results.json).

## Wire size, setup and token-only comparison

The tested format has a 56-byte header. A Gram packet adds \(K\) uint32 token IDs and the lower triangle of the reduced source Gram, containing \(K(K-1)/2\) scalars. A direct packet contains \(Kd\) final-prefix scalars. A token-only packet contains just the IDs after its header. All stated sizes are actual serialized packet sizes, including metadata; lower-layer network framing was not measured. [Packet specification and implementation](../latent_bridge/alignment_packet.py), [recorded size ledgers](../results/alignment_numerical/timings.jsonl.gz).

For \(K=64,d=1024\):

| Wire precision | Direct final prefix | Gram packet | Token-only packet |
|---|---:|---:|---:|
| float64 | 524,344 bytes | 16,440 bytes | 312 bytes |
| float32 | 262,200 bytes | 8,376 bytes | 312 bytes |

The float64 Gram packet was **31.89× smaller** than the direct float64 prefix, but **52.69× larger** than the token-only packet. Token-only transmission does not reconstruct the same continuous prefix, so it is a useful size reference rather than an equivalent-output baseline. These measurements do not establish a bandwidth advantage over English or any particular tokenizer.

The warm Gram and token-lookup paths assume the receiver already has the exact shared vocabulary and configuration. In this fixture, a cold session requires **2,621,440 bytes of vocabulary data plus a 479-byte manifest**, totaling **2,621,919 bytes**. A direct-prefix recipient can decode with a lightweight identity-and-dimension context and no vocabulary table. That extra decoder requirement must not automatically be charged to both sides of the comparison. A deployment that already has the table may have no incremental table-transfer cost, but that deployment condition was not tested. [Case-level session accounting](../results/alignment_numerical/cases.jsonl.gz), [vocabulary-free direct decoding](../latent_bridge/alignment_packet.py).

## Precision is a material limitation

Float64 packet checks passed for the method-specific reconstruction and supported vocabulary-margin conditions. This is same-runtime numerical reconstruction of the chosen algorithm, not a general lossless codec for arbitrary original prefixes, nor a claim of cross-platform bit reproducibility.

Float32 quality was not equivalence-certified. In **B-geometry7-lambda0.1**, the source had centered rank 7 and the reference rank 3, so the aligned product was nonunique. The accepted canonical float32 Gram packet did **not** fall back: it used 200 bytes instead of the 1,080-byte direct float32 prefix. Its final output differed from the unrounded canonical output by **4.243878 absolute Frobenius norm**, or **0.358925 relative error (35.89%)**. On the same reference, direct float32 rounding produced only **\(2.665\times10^{-8}\)** relative error. Both had zero nearest-vocabulary-ID changes. This shows why unchanged nearest tokens are insufficient evidence that a continuous prefix was faithfully reconstructed. [Exact case record](../results/alignment_numerical/cases.jsonl.gz).

Across the 27 exact deficient-rank fixtures, the canonical float32 path sent 18 Gram packets and used 9 dense fallbacks; its float64 path sent all 27 as Gram packets. All 120 primary full-rank cases and all 12 timing cases used Gram packets without fallback at either precision. The maximum descriptive float32 relative error was approximately **\(1.185\times10^{-5}\)** in the primary full-rank suite and **\(3.443\times10^{-9}\)** in the timing suite. These are properties of the fixed synthetic fixtures, not reliability estimates for model activations. [Case records](../results/alignment_numerical/cases.jsonl.gz).

Fallback is explicitly tagged and sends the dense-reference final output. In a nonunique case that output can differ from the canonical completion the sender initially attempted. The records preserve that method-switch difference separately from wire rounding. A fallback therefore preserves its declared dense output; it does not silently establish equality between all optimal completions.

## Rank applicability is a deployment constraint

Repeated IDs reduce the rank of the gathered reference embeddings. Suite F measured only this reference-side condition using synthetic Gaussian vocabulary tables and sampling with replacement. Each cell contains 192 token lists over three fixed seeds. The table shows how many satisfied the reference-side full centered-rank requirement. [Applicability records and seed summaries](../results/alignment_numerical/results.json).

| \(K\) | Vocabulary rows | Uniform sampling | Zipf exponent 1.2 |
|---:|---:|---:|---:|
| 2 | 32 | 185/192 (96.35%) | 165/192 (85.94%) |
| 2 | 256 | 190/192 (98.96%) | 184/192 (95.83%) |
| 8 | 32 | 80/192 (41.67%) | 4/192 (2.08%) |
| 8 | 256 | 172/192 (89.58%) | 30/192 (15.63%) |
| 16 | 32 | 5/192 (2.60%) | 0/192 (0.00%) |
| 16 | 256 | 124/192 (64.58%) | 0/192 (0.00%) |

The sender must separately satisfy its own rank condition; suite F did not measure that. These rates are not estimates for natural-language sequences or any model tokenizer. They demonstrate why repeated tokens and deficient rank cannot be hidden behind a full-rank theorem. The canonical extension and dense fallback have distinct numerical and output-identity contracts.

## Independent audit and retained evidence

The subsequent [audit and publication job](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45372752216/job/138370927774) completed successfully. Its [receipt](../results/alignment_numerical/audit_receipt.json) reports consistent data integrity and report summaries, no scientific-validation issues, and no detected required matrix failure. The archive contains **861 arrays mapped to 243 cases**: all algebraic fixtures plus the largest timing fixture. Original case and timing records are permanently retained as exact compressed files.

The independent auditor checked 791 alignment-method outputs from retained inputs, recomputing dense covariance and proper-rotation Procrustes properties, Gram and optimal-objective conditions, and output equality only in the justified unique-product scope. The standalone postprocessing fixture is retained but excluded from those algebra checks. It also checked recorded timing coverage, component summaries, paired ratios and order. It did not remeasure performance, run a model, regenerate fixtures, reproduce the canonical basis choice in nonunique cases, or independently execute packet decoding, postprocessing or float32 quality checks. The other eleven timing fixtures intentionally have seeds and input hashes rather than retained matrices. The composed direct/Gram total fields are also outside that auditor's independent recalculation scope; they remain explicit arithmetic sums of recorded component medians. A separate [read-only review of the original records](../results/alignment_postpublication_review.json) recomputed all 144 such totals without a mismatch and aggregated the descriptive packet errors; it did not repeat timing or packet decoding. These boundaries matter when interpreting “audit passed.” [Audit scope and limits](../results/alignment_numerical/audit_receipt.json).

| Provenance item | Immutable identifier |
|---|---|
| Frozen design commit | `e7c467f2b097666dc474a3dbb6e273bf253e785f` |
| Executed source commit | `7aee330f7799751a6a9f4b2a8decb612abe9c266` |
| Original study | [Run 45369667584, job 138360349493](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45369667584/job/138360349493) |
| Permanent evidence commit | `9bd33818226839934d04c8275d6e205672f83f6e` |
| Original results SHA256 | `8118b4994b5de1434e44dabf5692337c19112e469567be7c1c48d59a0ab51675` |
| Original execution SHA256 | `92212ae75c46bbd8c0f866b963801be39dd212d0efa737d19cd3adb4280a262d` |
| Audit receipt SHA256 | `7c3cade741f70f108590342b0e924bfe3d304d36cb82e23f685b86ef8ef3a88d` |

The study process completed in **41.611 seconds** including imports, within its frozen 300-second cap. Its focused gate passed **36 tests**; the separate full repository CI passed **126 tests** on the same source commit. The runtime was Python 3.11.18, NumPy 2.4.6 and single-threaded OpenBLAS on one Linux CPU host. There was no empirical rerun for the audit or packaging. [Execution manifest](../results/alignment_numerical/execution.json), [runtime manifest](../results/alignment_numerical/runtime_manifest.json), [full CI](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45369667483/job/138360348426).

## What this supports next

The useful result is a reproducible implementation and transport boundary: a faster guarded alignment core, a rigorously scoped canonical extension, and an explicit packet ledger that exposes computation, setup, rank and precision costs. Standard Gram invariance, polar/Bures relations and Procrustes methods already provide the underlying mathematics; broad claims of a new communication language or a novel general Gram protocol are not justified. [Derivation and primary prior art](alignment_geometry.md).

A next communication experiment should compare complete sender-and-receiver transactions against the efficient dual-direct route, preserve both full-vector fidelity and task-relevant behavior, and charge every preflight, shared-table and fallback cost. Actual model use would require a separate task evaluation. The present results provide no measured improvement in answer quality, no complete LLM latency result, and no evidence that a smaller float32 Gram packet is generally safe to substitute for its intended continuous prefix.
