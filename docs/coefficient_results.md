# Coefficient packet v2 results

The coefficient packet completed all 720 planned TCP exchanges and passed both predeclared correctness and fidelity criteria. It failed both performance criteria, so the frozen study's overall engineering-success result is **false**.

Under the imposed bandwidth and reply-delay conditions, the coefficient route used 72.73% less completion time than the efficient dual-plus-direct route in the specified aggregate. Against the stronger optimized Gram packet, however, its aggregate improvement was only **1.97%**, below the required 3%. On unshaped warm loopback it was **20.98% slower** than the best direct comparator. Its cold first-message time was **6.23 times** the direct comparator when the receiver had to acquire the shared table. These are measurements of a synthetic matrix codec on one CPU host with application pacing.

**Independent audit status: completed.** The [published receipt](../results/coefficient_packet/audit_receipt.json) reports zero data-integrity issues, zero report-consistency issues and no failed scientific checks within its stated scope. It reproduces the two passed and two failed criteria, including **engineering success: false**. The receipt's exact 144,937 bytes were independently fetched from the permanent commit and SHA-256 verified. The audit's successful validation of the evidence does not convert the measured negative performance result into a success.

Sources: the [frozen design](https://github.com/ConfusedDuckCollectibles/Research3/blob/49cfba7261326d58e174e51c8ddf5abba6f4a966/docs/coefficient_packet_design.md), [executed source](https://github.com/ConfusedDuckCollectibles/Research3/tree/5e158fb8ecd9e4f70f532aeb93cc253bbea7741f), and [original execution job](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380601801/job/138398320543) define the experiment. The original worker report and machine-readable results are retained separately from this interpretation.

## 1. What the experiment tested

The preceding [alignment study](alignment_results.md) found that a small Gram packet could reproduce a defined aligned prefix in supported cases, but the complete codec could cost more CPU time than computing and sending the prefix directly. The present study tested a narrower change: compute the symmetric coefficient matrix at the sender and transmit that coefficient instead of asking the receiver to solve for it again.

For centered row basis $H$, gathered receiver rows $R$, their mean $\mu_R$, and $B=H^T R$, the supported receiver computes

$$
Z=(HM)B+\mathbf{1}\mu_R^T
$$

before the same normalization and vocabulary anchoring used by the reference pipeline. The coefficient $M$ comes from the established Gram/Procrustes construction described in [the geometry note](alignment_geometry.md). The experiment concerns how to transport and reconstruct that specified output. It does not establish a new alignment theorem, a learned communication language, model accuracy, or general-purpose compression of arbitrary hidden states.

The packet includes token IDs and the lower triangle of a symmetric matrix. Its configuration binds the basis, multiplication order, precision convention and table identity. The primary scope is the supported full-centered-row-rank, unique-product case. Unsupported cases use the declared dense final-prefix fallback; this study does not add a canonical rank-deficient codec.

### Comparators

| Arm | Sender work and transmitted information | Receiver work |
|---|---|---|
| Coefficient | Compute and guard the actual rounded coefficient; send its lower triangle and IDs | Check the received coefficient, expand it, then postprocess once |
| Dual direct | Use the efficient dual solver and shared postprocessing cache; send the final dense prefix | Decode the dense bytes; no vocabulary table is required |
| Thin direct | Use the thin-SVD solver and the same direct transport contract | Decode the dense bytes; no vocabulary table is required |
| Optimized Gram | Guard the actual rounded Gram matrix with two eigensolves and a conservative spectral bound; send the Gram triangle and IDs | Solve for the coefficient, expand, and postprocess |
| Original Gram | Retain the earlier, more expensive complete sender preflight | Decode the earlier Gram route |
| Token lookup | Send only the IDs | Return the corresponding table rows |

The optimized Gram arm is the important small-packet comparator. Its sender does not compute $M$, expand a prefix, or search the vocabulary, so a coefficient advantage cannot be attributed solely to removing the old expensive preflight. Both efficient direct arms are also retained.

Token lookup is a byte/time reference. It returns a different output and is excluded from the performance success criteria. Its low cost is not evidence that it preserves the continuous prefix.

## 2. Frozen coverage and acceptance criteria

The nine primary inputs combined shapes $K\times d$ of 12 × 192, 36 × 512 and 60 × 960 with seeds 17011, 17027 and 17041. The fixed settings were $\lambda=10^{-3}$, covariance floor $10^{-6}$, and anchor weight 0.3. Additional diagnostics covered 16 rank and conditioning stress inputs at 6 × 24.

There were 648 warm trials: three shapes, three seeds, three service profiles, six arms and four repetitions. The 72 cold trials used the three shapes, seed 17011, the constrained profile, all six arms and four repetitions. Eighteen separate warmups were excluded from the timing analysis. The frozen schedule balanced arm positions across seeds or shapes and shuffled the six-arm blocks deterministically.

The 300 diagnostic rows cover both precision labels for all six arms on nine primary and 16 stress inputs. Token packets retain integer IDs under both labels. All measured network trials used float64; float32 output fidelity was descriptive.

| Predeclared criterion | Observed evidence | Result |
|---|---|---|
| Complete correct transport and required diagnostics | 720/720 completed trials; 300/300 diagnostic rows; zero failed required diagnostic checks; 54/54 primary routes eligible; all 18 warmups completed | Pass |
| Shaped warm coefficient/comparator ratio at most 0.97 against both direct arms, optimized Gram and the hindsight reference | Worst aggregate comparator was optimized Gram/hindsight: 0.980299 | Fail |
| Unshaped loopback/hindsight ratio at most 1.10, and worst warm case at most 1.25 | Aggregate 1.209805; worst case 1.485981 | Fail |
| All-row coefficient-versus-dual fidelity in all nine primary cases | 9/9 passed, including final vectors and internal anchor choices | Pass |

The coefficient route's largest primary float64 final relative Frobenius error was $2.6981\times10^{-16}$. Required numerical comparisons used the fixed absolute-plus-relative tolerance $10^{-10}+10^{-8}\lVert Z_{\mathrm{reference}}\rVert_F$. Equality of stable-margin anchor choices was treated separately from vector error; the additional practical coefficient/dual gate checked all rows empirically.

A failed performance criterion is a valid completed outcome. No retry, altered threshold, replacement fixture or post-outcome retuning was used.

## 3. Measured completion time

Every ratio below is **coefficient wall time divided by comparator wall time**. Smaller values favor the coefficient.

For each fixed case, the analysis first divides the two measured times within each repetition and then takes the median of those four ratios. The population summary is the equal-case geometric mean of those case medians. It is neither a ratio of population mean times nor a ratio of the two arms' medians.

The hindsight reference selects the comparator with the smallest four-run raw-time median for each case, among dual direct, thin direct and optimized Gram. It is a retrospective benchmark, not an implemented selection policy.

| Fixed population | Dual direct | Thin direct | Optimized Gram | Original Gram | Hindsight reference |
|---|---:|---:|---:|---:|---:|
| Shaped warm: 18 cases | 0.272667 | 0.266467 | 0.980299 | 0.896141 | 0.980299 |
| Unshaped loopback warm: 9 cases | 1.209805 | 0.813874 | 0.883942 | 0.611237 | 1.209805 |
| Cold constrained: 3 cases | 6.229016 | 6.200734 | 0.999818 | 0.999273 | 6.229016 |

Optimized Gram was the hindsight choice in all 18 shaped warm cases. The coefficient had a lower paired median in every one, but the aggregate reduction was only 1.9701%, so the 3% target was not met. The individual coefficient/optimized-Gram case medians ranged from 0.946112 to 0.996596.

Dual direct was the hindsight choice in all nine unshaped warm cases and all three cold cases. The worst warm coefficient/hindsight ratio, 1.485981, occurred at 12 × 192, seed 17041, on unshaped loopback. The aggregate and worst-case overhead limits both failed.

The faster results against direct transmission under shaping are consistent with the much smaller message. Coefficient and optimized Gram have equal wire sizes, so their smaller difference reflects the measured placement and amount of computation under those service conditions. The older Gram implementation is a useful historical comparison, but it is not the strongest small-packet baseline.

### Cold first-message costs

Cold trials include the actual shared-table or direct-context setup, its acknowledgment, encoding, transfer, reconstruction and the final receipt. These are measured first-message costs; this report does not amortize them over a hypothetical number of later requests.

Each entry below is the median of four original times in milliseconds. All three cases used seed 17011 and the constrained profile.

| Shape | Coefficient | Dual direct | Thin direct | Optimized Gram | Original Gram | Token lookup |
|---|---:|---:|---:|---:|---:|---:|
| 12 × 192 | 461.193 | 61.184 | 61.312 | 461.291 | 461.652 | 459.403 |
| 36 × 512 | 1260.432 | 193.247 | 194.120 | 1260.559 | 1261.122 | 1252.378 |
| 60 × 960 | 2516.728 | 512.218 | 516.052 | 2517.853 | 2519.342 | 2497.798 |

The coefficient and Gram receivers need the table before reconstruction. The direct receivers only need the lightweight dimension/session context. The extra table is decisive here: the coefficient/direct cold ratios aggregate to 6.229016, even though the coefficient's subsequent warm messages are smaller.

## 4. Actual serialized and framed bytes

These are bytes observed by the application at both endpoints. They include the message's 56-byte packet header, the four-byte TCP application frame prefix, and the 40-byte acknowledgment plus its own frame prefix. They exclude TCP/IP headers and other lower-layer overhead.

Warm message totals were constant across repetitions and service profiles:

| Shape | Coefficient / either Gram | Either direct arm | Token lookup |
|---|---:|---:|---:|
| 12 × 192 | 680 | 18,536 | 152 |
| 36 × 512 | 5,288 | 147,560 | 248 |
| 60 × 960 | 14,504 | 460,904 | 344 |

A float64 coefficient or Gram triangle contains $K(K-1)/2$ scalars, plus $K$ 32-bit IDs. The final dense prefix contains $Kd$ float64 scalars. The coefficient changes the computation location; it does not reduce bytes relative to the optimized Gram packet in these supported cases.

Cold totals include both setup and message exchanges:

| Shape | Coefficient | Dual direct | Thin direct | Optimized Gram | Original Gram | Token lookup |
|---|---:|---:|---:|---:|---:|---:|
| 12 × 192 | 413,591 | 18,745 | 18,745 | 413,590 | 413,593 | 413,064 |
| 36 × 512 | 1,202,584 | 147,769 | 147,769 | 1,202,583 | 1,202,586 | 1,197,545 |
| 60 × 960 | 2,442,648 | 461,113 | 461,113 | 2,442,647 | 2,442,650 | 2,428,489 |

The underlying float64 table payloads were 411,648, 1,196,032 and 2,426,880 bytes respectively. Minor cold differences among the table arms come from their actual setup metadata, not omitted table costs. Token lookup also needs the shared table in this cold protocol.

## 5. Float32 and rank-boundary limitations

Float32 was not used to pass the primary timing or fidelity criteria. The retained diagnostics contain the actual rounded bytes, actual decoded vectors, route decisions, dense-fallback targets and same-precision direct controls.

On the nine well-conditioned primary cases, the coefficient's maximum float32 relative final error was $1.80953\times10^{-8}$, with no internal anchor changes. For context, the corresponding maximum across dual-direct cases was $2.55726\times10^{-8}$, and across optimized-Gram cases it was $9.75304\times10^{-9}$. These are fixture-wise maxima, not matched-case rankings of general numerical stability.

Stress inputs show the guard boundary:

| Arm | Float64 native / fallback, out of 16 | Float32 native / fallback, out of 16 |
|---|---:|---:|
| Coefficient | 2 / 14 | 2 / 14 |
| Dual direct | 2 / 14 | 2 / 14 |
| Thin direct | 2 / 14 | 2 / 14 |
| Optimized Gram | 0 / 16 | 2 / 14 |
| Original Gram | 2 / 14 | 4 / 12 |

The two native coefficient cases were the deliberately conservative-bound false-rejection controls. The optimized Gram sender's sufficient bound rejected all 16 unrounded float64 stress cases, including those two cases that the fuller coefficient calculation supported. This is the cost of a cheaper conservative preflight; these counts do not establish a universally better rank detector.

A material float32 case was `stress-k6-d24-seed19001-g1`. Both Gram routes accepted its rounded packet natively, even though the preferred unrounded fast reference was unsupported. The explicitly labeled descriptive comparison used the retained dense reference. Its final relative Frobenius error was **0.0008871187** (approximately **0.0887%**), despite **zero internal anchor-ID changes**. The route remains reported as native; it was not relabeled as a fallback.

That example illustrates why an unchanged token decision is an inadequate substitute for full-vector fidelity. It also shows that rounding can move an input across a numerical support boundary. Forming a Gram matrix squares singular-value conditioning relative to the underlying factor. These results therefore retain near-rank failures and fallback behavior as material limits. They do not certify arbitrary float32 packets, other dimensions, or other tables.

## 6. What the clock and protocol included

The experiment used real loopback TCP. A receiver worker reconstructed the output from received bytes and legitimate session state. No source array, expected prefix, reference decision or precomputed message was available to the receiver. The sender computed each measured packet anew.

The final `ACK2` receipt contains a hash of the receiver's actual canonical float64 output. Its construction and transmission are included in completion time. In cold mode, `SET2` acknowledges the exact received setup bytes. Expected-output comparisons, full trace reconciliation and retention hashing occur after the timed exchange.

Warm timing starts immediately before the complete encoder and ends after the sender validates the final acknowledgment. Endpoint readiness, connection setup, input copies and session preparation are recorded outside the warm timer. The receiver-ready event prevents thread acceptance or socket setup from leaking into that interval.

Cold timing begins before client connection and session preparation and continues through both setup and output receipts. Listener/thread startup is outside that interval. The receiver's table setup is charged only where the protocol requires a table.

The imposed service profiles were:

| Profile | Per-direction application rate | Holdoff before each receiver reply |
|---|---:|---:|
| loopback | Unlimited | 0 |
| moderate | 10,000,000 bytes/s | 5 ms |
| constrained | 1,000,000 bytes/s | 20 ms |

Every framed chunk, including a short final chunk and an acknowledgment, is charged before sending. These are application pacing and receiver reply-service delays. They are not measurements of WAN RTT, congestion, packet loss, production throughput or a deployed model system.

The original run used Python 3.11.18, NumPy 2.4.6, OpenBLAS 0.3.31.188.0 and one configured BLAS/OpenMP thread on Linux. The runner exposed four CPUs; its observed CPU-model string was “AMD EPYC 9V74 80-Core Processor.” That string does not imply 80 available cores or 80-core use. The child completed in 107.249 seconds within its fixed 300-second budget.

Four repetitions on one host support this bounded engineering comparison. The acceptance thresholds are predeclared practical targets, not confidence intervals or tests of statistical significance. Shape and dimension were changed together, so the grid does not separately estimate their scaling effects.

## 7. Evidence, audit boundary and interpretation

Before the measurement workflow was activated, the exact Python sources passed the full repository suite: 162 tests, no failures, errors or skips, in 12.513 seconds. The measurement job then passed its original supervision and focused contract gates. The workflow ran once; the original execution receipt records zero benchmark retries and zero model imports or inference.

The artifact retains the fixed plan, all inputs, exact packet/setup/ACK and framed-stream bytes, receiver arrays, reference outputs, raw timing records, phase and wait ledgers, diagnostic errors, and completion accounting. The original report and result data remain unchanged. They are published in [results/coefficient_packet](../results/coefficient_packet/), including [results.json](../results/coefficient_packet/results.json), [execution.json](../results/coefficient_packet/execution.json), [trials.jsonl.gz](../results/coefficient_packet/trials.jsonl.gz) and [diagnostics.jsonl.gz](../results/coefficient_packet/diagnostics.jsonl.gz).

The separate [audit/publication run](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45384590430/job/138412128542) completed and published the original evidence at commit [6172944a3d90180e1e95f521eed79a9ba02c6c25](https://github.com/ConfusedDuckCollectibles/Research3/commit/6172944a3d90180e1e95f521eed79a9ba02c6c25). Its read-only receipt records:

| Audit category | Checks | Reported issues or failures |
|---|---:|---:|
| Data integrity | 233,297 | 0 |
| Report consistency | 106,829 | 0 |
| Scientific validation within the stated scope | 17,025 | 0 |

The audit authenticated 734 indexed original files plus the separately anchored execution receipt: **735 original files, 122,683,484 bytes**. It parsed actual packets, setup messages, acknowledgment hashes and framed streams; checked the fixed schedule, observed wait and phase arithmetic; and recomputed the descriptive vector errors, paired statistics and all four acceptance criteria. The resulting engineering-success field remains false.

Numerical reconstruction has a precise scope. The audit independently reconstructed **25 distinct coefficient wire objects** and **18 primary Gram wire objects**. It called the SHA-pinned earlier dense solver on **nine retained primary inputs**. Reusing that solver is not an independently rewritten dense reference. It explicitly skipped **14 nonprimary Gram decompositions**, while checking their retained pre/post arrays and descriptive errors. Arbitrary rank-deficient null completions are not claimed to reproduce bit for bit across hosts.

For near-tied replay anchor choices, the audit checks the retained decision against a near-maximum score and verifies the anchored vector; it does not infer exact cross-runtime argmax identity from a tiny margin. Not every sender guard predicate is independently re-executed or certified. The observed clock and pacing records are checked arithmetically, without remeasuring time.

The receipt records **zero new fixture generations, zero model imports or inference, zero optimization invocations and zero timing measurements**. The random input arrays were read, hashed and spectrum-checked, not regenerated. The independent audit therefore strengthens confidence in the recorded result within these boundaries; it does not establish general float32 fidelity, a new workload result or a production latency claim.

| Original evidence or separately labeled audit | SHA-256 |
|---|---|
| `results.json` | `6c390b5f46f2f6c91863373da25b7df12a023f166e1f543e3ed98284eb9d3263` |
| `execution.json` | `35dd69474b036160dc34cff07eb2bfc85fb4d5223f7f143456933e56a03a4f95` |
| `runtime_manifest.json` | `fd5326b0421949c033ec45795f1b18c7fc56fe1b931b5f86ba27b001c27f4e6d` |
| Separate `audit_receipt.json` | `5aa9b2be66268efb780c7b37a941e52a7e9f336d021bd0acb9f740081e5d3627` |
| `trials.jsonl.gz` | `fd71a00c9eafb18f5cc9fa9adbc340ab4b3a7660c6f090d9ff4d1387e2e2ce8b` |
| `diagnostics.jsonl.gz` | `b4c5392db05d2823b236481ff6dadac0726b1c22bb69e2d1ade4b155c86dc92f` |

Original run: [45380601801](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380601801). Original job: [138398320543](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380601801/job/138398320543). Original artifact ID: **14914125531**. Executed source: [5e158fb8ecd9e4f70f532aeb93cc253bbea7741f](https://github.com/ConfusedDuckCollectibles/Research3/commit/5e158fb8ecd9e4f70f532aeb93cc253bbea7741f).

The supported result is a numerically faithful coefficient representation on this fixed primary cohort, together with a measured cost tradeoff. Shared-table small packets gave large savings against sending dense prefixes under imposed bandwidth constraints. The cheaper Gram baseline captured almost all of that benefit, and direct transmission had lower aggregate times for warm unshaped loopback and the measured cold starts. The frozen criteria therefore reject an overall engineering-success claim for this version.

## Appendix: every warm case

Entries are medians of four original completion times in milliseconds. The aggregate decisions above use within-repetition ratios, so they cannot be reconstructed by simply dividing these rounded median columns. Token lookup remains non-equivalent.

| Shape | Seed | Profile | Coefficient | Dual direct | Thin direct | Optimized Gram | Original Gram | Token lookup |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 12 × 192 | 17011 | loopback | 0.931 | 0.689 | 0.758 | 1.052 | 1.462 | 0.217 |
| 12 × 192 | 17011 | moderate | 6.310 | 7.774 | 7.914 | 6.401 | 6.718 | 5.423 |
| 12 × 192 | 17011 | constrained | 22.022 | 39.673 | 39.824 | 22.120 | 22.450 | 20.615 |
| 12 × 192 | 17027 | loopback | 0.971 | 0.671 | 0.799 | 1.118 | 1.431 | 0.233 |
| 12 × 192 | 17027 | moderate | 6.276 | 7.800 | 7.884 | 6.299 | 6.688 | 5.418 |
| 12 × 192 | 17027 | constrained | 21.976 | 39.571 | 39.696 | 22.052 | 22.424 | 20.675 |
| 12 × 192 | 17041 | loopback | 0.989 | 0.650 | 0.787 | 1.075 | 1.432 | 0.234 |
| 12 × 192 | 17041 | moderate | 6.277 | 7.762 | 7.920 | 6.396 | 6.783 | 5.478 |
| 12 × 192 | 17041 | constrained | 21.959 | 39.685 | 39.722 | 22.070 | 22.474 | 20.610 |
| 36 × 512 | 17011 | loopback | 2.144 | 1.838 | 2.896 | 2.443 | 3.621 | 0.341 |
| 36 × 512 | 17011 | moderate | 7.933 | 22.487 | 23.648 | 8.197 | 9.531 | 5.551 |
| 36 × 512 | 17011 | constrained | 27.817 | 170.974 | 172.098 | 28.151 | 29.384 | 20.767 |
| 36 × 512 | 17027 | loopback | 2.168 | 1.828 | 2.990 | 2.421 | 3.663 | 0.333 |
| 36 × 512 | 17027 | moderate | 8.027 | 22.343 | 23.515 | 8.126 | 9.361 | 5.536 |
| 36 × 512 | 17027 | constrained | 27.800 | 170.979 | 171.976 | 28.122 | 29.291 | 20.849 |
| 36 × 512 | 17041 | loopback | 2.139 | 1.875 | 2.875 | 2.498 | 3.624 | 0.311 |
| 36 × 512 | 17041 | moderate | 7.933 | 22.443 | 23.516 | 8.087 | 9.320 | 5.534 |
| 36 × 512 | 17041 | constrained | 27.805 | 171.454 | 172.097 | 28.197 | 29.463 | 20.840 |
| 60 × 960 | 17011 | loopback | 4.869 | 4.457 | 8.161 | 5.413 | 8.260 | 0.643 |
| 60 × 960 | 17011 | moderate | 11.324 | 57.517 | 60.997 | 11.968 | 14.840 | 5.797 |
| 60 × 960 | 17011 | constrained | 39.614 | 489.306 | 492.843 | 40.270 | 43.104 | 21.187 |
| 60 × 960 | 17027 | loopback | 4.662 | 4.479 | 7.787 | 5.335 | 8.020 | 0.597 |
| 60 × 960 | 17027 | moderate | 11.456 | 57.615 | 61.243 | 11.984 | 14.890 | 5.817 |
| 60 × 960 | 17027 | constrained | 39.495 | 489.121 | 492.536 | 40.089 | 42.716 | 21.196 |
| 60 × 960 | 17041 | loopback | 4.679 | 4.282 | 7.975 | 5.239 | 8.050 | 0.584 |
| 60 × 960 | 17041 | moderate | 11.318 | 57.514 | 61.153 | 11.969 | 14.870 | 5.828 |
| 60 × 960 | 17041 | constrained | 39.525 | 489.398 | 492.804 | 40.237 | 43.016 | 21.203 |
