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
