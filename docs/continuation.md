# Research continuation

The owner has authorized continued investigation and publication in this repository. The objective is a defensible advance in useful learned communication. Experimental progress, causal evidence and scientific novelty must be assessed separately.

## Resume protocol

Read `research_state.json`, the latest README, frozen designs, and completed results before choosing work. Check current GitHub Actions and any available active-worker state. Respect a fresh worker lease; do not duplicate an active experiment or overwrite another worker's files. An expired lease is a recovery signal, not evidence that the previous experiment failed. Recover any completed workflow artifacts first.

Claim a new lease with an expected-head Git update before starting a new experiment. Record a unique worker ID, current UTC update time, a lease expiration no more than two hours ahead, and the exact next action. Renew during active work. At a normal stopping point, publish all reproducible work and set the status to `ready` or `blocked`, remove the active lease, and describe what remains. UTC timestamps are coordination metadata; user-facing schedules use America/Chicago.

Publish the hypothesis, primary comparison, controls, held-out split, success criteria, resource budget, and failure reporting before inspecting new held-out outcomes. Then publish the implementation before execution. Keep all seeds, failures, negative results, raw predictions and provenance. Separate tuning from evaluation. Use new evaluation content after an outcome-driven design change.

The frozen-message readout diagnostic and binding study are complete and permanently published. Read `docs/readout_results.md` and `docs/binding_results.md`. The binding run completed 18 fits and all 86,016 fresh predictions; its independent audit passed. Both aware latent arms scored zero target pairs correctly, while full text scored 99.54%. Do not repeat fitting or fresh scoring, relabel the negative result, or tune on that completed cohort.

The original alignment numerical study is complete and independently audited. Read `docs/alignment_results.md`, `docs/alignment_execution_notes.md` and `results/alignment_numerical/`. All 15,759 required checks passed; the dual core was 1.17–3.02x faster than the thin-SVD baseline on twelve CPU fixtures, but the original Gram route cost 1.86–2.29x the arithmetic time of dual-direct. A nonunique canonical float32 case had 35.89% relative vector error despite unchanged nearest IDs. Float32 quality was descriptive, not equivalence-certified. Preserve all original source and outcomes; do not repeat the completed run or broaden its claims.

The coefficient-packet TCP study is complete, independently audited and permanently published. Read `docs/coefficient_results.md`, `docs/coefficient_execution_notes.md` and `results/coefficient_packet/`. Original run45380601801 completed720/720 actual TCP trials and300 diagnostics; all nine primary full-vector fidelity gates passed. Both frozen performance criteria failed: shaped coefficient/optimized Gram0.9802991311, warm loopback/hindsight1.2098053452, worst warm1.4859810967. Cold/hindsight was6.2290160665. Audit run45384590430 found no issues within its stated byte/array/statistical scope and preserved all findings in commit6172944a3d90180e1e95f521eed79a9ba02c6c25. Do not rerun this completed cohort, relax its thresholds or select a weaker primary baseline after outcomes.

The analytical rank-deficient note `docs/rank_deficient_feasibility.md` preserves the selected legacy completion through reference-space coefficients and residual coordinates. Its algebra has been reviewed, but its proposed representation is not an executed codec, an information bound, or a new compression principle. The next read-only feasibility question is exact information recovery from the full normalized and anchored alignment at fixed reference tokens, together with practical precision on real contextual model states. No new model or numerical assay is activated at this checkpoint; any such work needs its own frozen claim, controls, budget and evidence-retention rules.

The basis in `docs/alignment_geometry.md` concerns established Procrustes/Bures/Gram mathematics specialized to an alignment interface. Unique-product equivalence, canonical nonunique completion and dense fallback have different contracts. No matrix or loopback result establishes model quality, production speed, scientific novelty or a breakthrough.

Recover any active workflow and its artifacts before starting new work. Keep main unchanged while a publication workflow is committing an already verified artifact. Packaging and read-only audits do not authorize rerunning empirical outcomes.

The separate feasibility note `docs/lossless_predictor_feasibility.md` rejects starting another model study for vocabulary-byte XOR prediction with the current artifacts. Reopen only with the evidence and strong baselines specified there.

Use the existing tools and bounded local or standard public CPU workflows. Preserve all previous studies. If access, runtime or usage prevents progress, record the concrete blocker instead of fabricating a result. No worker can infer the user's remaining ChatGPT quota from these files.

## Finding labels

`discoveries/` indexes measured findings, including negative findings; the name does not assert that a method is new to science. `breakthroughs/` is reserved for a candidate supported by a precise claim, strong comparisons, appropriate independent replication, a relevant prior-art review, and reproducible evidence. Publication alone does not meet that threshold.

## Reviewed information characterization

Read `docs/statebridge_information_recovery.md`. At positive regularization, any exact orthogonal pre-calibration output determines the centered source Gram when the reference is known. Full row rank of the actual pre-prefix also permits inversion of calibration and anchoring for0<=alpha<1. In the centered unique-product regime with full uncentered reference rank, this is exact Gram equivalence. Nonunique completions may carry additional orientation. These statements are exact-arithmetic results, not bit-capacity or practical semantic claims. The upstream paper does not transmit the reference IDs, so the proposed inverse must not be represented as a decoder for its original prefix-only receiver.

The bounded assay is now frozen in `docs/contextual_information_design.md`: 256 authentic Pythia14M sequences from 64 paired families and two predetermined public probes, with 128 primary unique-probe records. Implement and review the pure inverse, exact dyadic Gram checks, cached prediction-step extraction, unchanged author-method execution, supervisor and independent offline tests before activating one 600-second scientific child. The fixed public-side-information boundary, 122/128 numerical target, byte-collision denominators, partial retention and no-rerun rule must remain intact. The owner's existing authorization permits this work; source and offline test gates are internal scientific requirements, not a new user-approval request.
