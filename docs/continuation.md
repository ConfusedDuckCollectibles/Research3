# Research continuation

The owner has authorized continued investigation and publication in this repository. The objective is a defensible advance in useful learned communication. Experimental progress, causal evidence and scientific novelty must be assessed separately.

## Resume protocol

Read `research_state.json`, the latest README, frozen designs, and completed results before choosing work. Check current GitHub Actions and any available active-worker state. Respect a fresh worker lease; do not duplicate an active experiment or overwrite another worker's files. An expired lease is a recovery signal, not evidence that the previous experiment failed. Recover any completed workflow artifacts first.

Claim a new lease with an expected-head Git update before starting a new experiment. Record a unique worker ID, current UTC update time, a lease expiration no more than two hours ahead, and the exact next action. Renew during active work. At a normal stopping point, publish all reproducible work and set the status to `ready` or `blocked`, remove the active lease, and describe what remains. UTC timestamps are coordination metadata; user-facing schedules use America/Chicago.

Publish the hypothesis, primary comparison, controls, held-out split, success criteria, resource budget, and failure reporting before inspecting new held-out outcomes. Then publish the implementation before execution. Keep all seeds, failures, negative results, raw predictions and provenance. Separate tuning from evaluation. Use new evaluation content after an outcome-driven design change.

The frozen-message readout diagnostic and binding study are complete and permanently published. Read `docs/readout_results.md` and `docs/binding_results.md`. The binding run completed 18 fits and all 86,016 fresh predictions; its independent audit passed. Both aware latent arms scored zero target pairs correctly, while full text scored 99.54%. Do not repeat fitting or fresh scoring, relabel the negative result, or tune on that completed cohort.

The original alignment numerical study is complete and independently audited. Read `docs/alignment_results.md`, `docs/alignment_execution_notes.md` and `results/alignment_numerical/`. All 15,759 required checks passed; the dual core was 1.17–3.02x faster than the thin-SVD baseline on twelve CPU fixtures, but the original Gram route cost 1.86–2.29x the arithmetic time of dual-direct. A nonunique canonical float32 case had 35.89% relative vector error despite unchanged nearest IDs. Float32 quality was descriptive, not equivalence-certified. Preserve all original source and outcomes; do not repeat the completed run or broaden its claims.

The new investigation is preregistered in `docs/coefficient_packet_design.md` at commit `49cfba7261326d58e174e51c8ddf5abba6f4a966`. Implement and review new v2 files before a single bounded CPU run. It compares a coefficient packet with dual-direct, thin-direct, an optimized cheap-guard Gram control, the original Gram codec and a non-equivalent token lookup. The fresh grid contains 648 warm and 72 cold actual TCP transactions, all inside a 300-second child budget. Preserve all numerical checks, guards, setup bytes, ACKs, actual received arrays, missing slots and failures. No precomputed prefix, packet or oracle can enter measured encoder decisions. Keep cold setup separate from warm results. The precise full-vector and strongest-comparator success conjunction is frozen in the design.

The basis in `docs/alignment_geometry.md` concerns established Procrustes/Bures/Gram mathematics specialized to an alignment interface. Unique-product equivalence, canonical nonunique completion and dense fallback have different contracts. No matrix or loopback result establishes model quality, production speed, scientific novelty or a breakthrough.

Recover any active workflow and its artifacts before starting new work. Keep main unchanged while a publication workflow is committing an already verified artifact. Packaging and read-only audits do not authorize rerunning empirical outcomes.

The separate feasibility note `docs/lossless_predictor_feasibility.md` rejects starting another model study for vocabulary-byte XOR prediction with the current artifacts. Reopen only with the evidence and strong baselines specified there.

Use the existing tools and bounded local or standard public CPU workflows. Preserve all previous studies. If access, runtime or usage prevents progress, record the concrete blocker instead of fabricating a result. No worker can infer the user's remaining ChatGPT quota from these files.

## Finding labels

`discoveries/` indexes measured findings, including negative findings; the name does not assert that a method is new to science. `breakthroughs/` is reserved for a candidate supported by a precise claim, strong comparisons, appropriate independent replication, a relevant prior-art review, and reproducible evidence. Publication alone does not meet that threshold.
