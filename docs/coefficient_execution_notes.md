# Coefficient packet v2: execution and evidence contract

The [fixed design](coefficient_packet_design.md) was published in commit `49cfba7261326d58e174e51c8ddf5abba6f4a966` before v2 implementation or outcomes. Its UTF-8 SHA256 is `b8307677f5320c3413b5a1007bdfe26230a844c57503d53b4443e5fc542647ba`.

This follow-up preserves the original alignment implementation and every original result. The original retained-array audit passed; its [report](alignment_results.md) distinguishes the faster matrix core from the slower original complete-packet arithmetic and the descriptive float32 failure. The new experiment tests complete transactions on fresh fixed fixtures, including a strong optimized Gram control. It does not evaluate an LLM or make a novelty claim.

## Implementation and review boundaries

The new codec lives in `latent_bridge/coefficient_packet.py`. It binds the coefficient mode, Helmert basis, grouping and precision conventions into the session identity; shares the dual solve with the direct baseline; guards the actually rounded coefficient; and gives the competing Gram encoder a two-eigensolve sufficient-condition preflight. The original numerical modules are imported unchanged and their SHA256 digests are enforced by the launcher.

The new transport uses actual loopback TCP. Every frame byte is paced under the declared shaped profiles, including the final acknowledgment. The receiver gets actual packet bytes and legitimate context; no source matrix, expected prefix, cached per-message matrix or oracle controls its decoder. The ordinary ACK hashes its actual decoded output. The distinct setup ACK hashes the actual setup payload. Expected-output comparisons occur afterward.

Warm timing starts only after the receiver has accepted and configured its socket, and then includes the complete encoder through final ACK receipt. Cold timing includes client connection, both required session constructions, table/setup transfer, setup acknowledgment and the complete first message. A declared shared table is never silently charged to a direct-prefix recipient.

All actual sent/received streams and decoded arrays are retained or deduplicated after measurement by exact content identity. Instrumentation hashes streams after the timer; the decoded-output hash required by the protocol is inside it. If a receiver thread cannot be stopped, later timed trials must stop rather than run alongside that thread.

## Gates before one measurement run

The complete new source and focused contract tests are published before execution. The existing repository CI runs first. Only after that gate succeeds is the fresh measurement workflow `.github/workflows/coefficient-packet-v2.yml` activated. Its own focused tests run before the worker. A unit-test correction before empirical execution is recorded as such; there is no performance-based tuning or hidden rerun.

The focused checks cover actual numerical and framing boundaries, fallback identity, sender/receiver separation, setup and ACK verification, final-byte pacing, warm endpoint readiness, and complete/partial grid accounting. The launcher reuses the original immutable and already-tested whole-process-group deadline and byte-preserving gzip functions. Local execution is unavailable in this session; static review is not represented as a local test pass.

## Fixed budget and retention

The scientific child runs once, with Python 3.11.18, NumPy 2.4.6 and one BLAS/OpenMP thread. Its 300-second wall cap includes imports, fixture construction, diagnostics, warmups, all 648 warm and 72 cold planned network slots, scientific checkpointing and finalization. The launcher permits only the first original workflow run and attempt and refuses to replace an existing output directory or permanent study results. There are no model downloads, model calls, paid API calls or accelerator jobs.

The worker writes the full 720-slot plan atomically before diagnostics. Trial records are appended and flushed; failures and ineligible slots retain their fixed identities. After a hard timeout, the parent derives missing IDs by reading that plan and the complete retained JSONL records. It never resumes scientific work, repairs a truncated raw record, or fills a missing duration with a synthetic measurement.

The parent records byte-preserving compression, file hashes and missing-slot accounting after the child exits. This packaging time is reported separately from the scientific child budget. The original output directory is uploaded even when a gate, worker or measurement fails; the artifact retention is 90 days. Permanent publication requires a separate read-only verification and packaging step, preserving partial and negative outcomes.

The worker source commit is the exact workflow `GITHUB_SHA`. The launch manifest inventories the design, immutable dependencies, new modules, tests, launcher, workflow and project configuration. The execution record binds source/run identity, cap, child status, file hashes and all-slot accounting. Bounded textual recovery blocks expose exact file hashes in the completed job log. No audit or publication step authorizes another experiment.

## Interpretation

Actual complete transaction wall time is the primary endpoint. Phase medians are explanatory; their sums do not replace a measured transaction. The strongest primary comparison includes an explicitly retrospective per-case envelope over dual-direct, thin-direct and the optimized Gram codec. It is not a deployable free selector.

All-row primary float64 output fidelity, complete fixed-grid accounting, the declared shaped-link gain, and the unshaped/worst-case overhead limits must all pass for the narrow engineering criterion. Cold first-message cost and descriptive float32 distortions remain separate and visible. A successful synthetic loopback result would still require independent replication and a task-quality evaluation before supporting a broader agent-efficiency claim.

## Pre-execution CI observed

The complete reviewed implementation was published at `367d81a6306e670ff39397c0747848cfd5a1de13`. [Tests run 45380063999, job 138396467220](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380063999/job/138396467220) passed all **162 tests in 12.513 seconds**, with no skips, errors or failures. This includes 36 new coefficient tests and the existing 126-test suite. The [validation receipt](../results/coefficient_validation.json) preserves the retrieved-log SHA256 and observed counts. No coefficient study grid was run by that CI gate.

The separately reviewed measurement workflow is now activated once. Its own runtime and focused contracts must pass before the 300-second scientific child. The exact executed source identity will be retained as the workflow `GITHUB_SHA`; this activation does not alter the validated Python implementation.

## Original execution recovered; independent audit queued

The one original [measurement run 45380601801, job 138398320543](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380601801/job/138398320543) completed at source `5e158fb8ecd9e4f70f532aeb93cc253bbea7741f`. Its scientific child took **107.248639553 seconds**, below the 300-second cap. All **720 network trials**, **300 numerical diagnostics**, and **9 primary all-row float64 fidelity gates** completed successfully. The two correctness criteria passed, but both prespecified performance criteria failed. The exact-source repository CI also passed all **162 tests in 14.769 seconds** in [run 45380602011](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45380602011).

The shaped-warm coefficient/optimized-Gram aggregate was **0.9802991311030891**, which misses the required 0.97 maximum. The loopback coefficient/hindsight aggregate was **1.2098053452463209** and the worst warm case was **1.4859810966717295**, exceeding the declared 1.10 and 1.25 limits. The cold aggregate was **6.229016066505879** relative to the retrospective best comparator. Each aggregate uses equal-case geometric means of four-repetition paired-ratio medians. These are negative engineering findings, not evidence of an agent-efficiency breakthrough.

The recovered artifact is `coefficient-packet-45380601801-attempt-1`, ID `14914125531`. Its GitHub ZIP metadata digest is `e0187f8b6db22705221857b002486fd4c146c3e7a1baa662c6806d297dccb199`; the archive itself has not been independently rehashed in this session. The exact recovered `execution.json` digest is `35dd69474b036160dc34cff07eb2bfc85fb4d5223f7f143456933e56a03a4f95`, and `results.json` is `6c390b5f46f2f6c91863373da25b7df12a023f166e1f543e3ed98284eb9d3263`. Two reviewers independently verified the bounded original log blocks. The immutable execution inventory contains **735 files and 122,683,484 bytes**, including `execution.json`.

The separate publication job will rehash every original file and check exact raw/gzip roundtrips before a read-only audit of actual packet/setup/ACK bytes, retained numerical arrays, schedule coverage, phase arithmetic and paired gates. Its numerical replay does not import the new codec, transport or study modules. It reuses the pinned original dense solver for nine primary references; nonprimary Gram decompositions, arbitrary null completions, every guard predicate and general float32 fidelity are outside its independent certification scope. No trial is repeated and no model is invoked. Only five redundant raw JSONL copies will be omitted from Git after their exact gzip replacements are verified; the original artifact retains both. Every original outcome and the audit receipt will be preserved, including negative or inconsistent findings. The audit is pending at this checkpoint.

## Independent audit and permanent publication completed

[Publication run 45384590430](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45384590430), job `138412128542`, verified all **735 original files / 122,683,484 bytes**, ran the independent auditor's elementary controls, and completed its read-only audit. The [receipt](../results/coefficient_packet/audit_receipt.json) records **233,297 integrity checks**, **106,829 report-consistency checks**, and **17,025 scientific validation checks**, with no issues or failed required checks. It independently reproduced the original four criteria, including both performance failures and `engineering_success: false`.

The receipt is **144,937 bytes**, SHA256 `5aa9b2be66268efb780c7b37a941e52a7e9f336d021bd0acb9f740081e5d3627`. Its auditor source SHA256 is `ca44d9d545664745c31f3b0aea9ff55bd16a991dd3f7a7a201fb482b46faea6c`. The audit ran on Python 3.11.18 and NumPy 2.4.6 on a separate Linux runner, without importing the new codec, transport or study modules. It independently reconstructed 25 distinct coefficient wire objects and 18 primary Gram objects, reused the fixed original dense solver for nine primary reference inputs, and explicitly did not replay 14 nonprimary Gram decompositions. The complete receipt documents the additional numerical and scope qualifications.

Commit [`6172944a3d90180e1e95f521eed79a9ba02c6c25`](https://github.com/ConfusedDuckCollectibles/Research3/commit/6172944a3d90180e1e95f521eed79a9ba02c6c25) publishes **733 files / 110,187,599 bytes**: 730 exact original files after replacing five redundant raw JSONL copies with their exact gzip forms, plus the audit receipt, fixed publication manifest and provenance. The published receipt's hash was checked again against the original audit job block. No benchmark, fit, model inference, fixture generation or timing remeasurement occurred during publication. Read the [complete results report](coefficient_results.md) for interpretation of the negative engineering outcome.
