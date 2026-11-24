# Research continuation

The owner has authorized continued investigation and publication in this repository. The objective is a defensible advance in useful learned communication. Experimental progress, causal evidence and scientific novelty must be assessed separately.

## Resume protocol

Read `research_state.json`, the latest README, frozen designs, and completed results before choosing work. Check current GitHub Actions and any available active-worker state. Respect a fresh worker lease; do not duplicate an active experiment or overwrite another worker's files. An expired lease is a recovery signal, not evidence that the previous experiment failed. Recover any completed workflow artifacts first.

Claim a new lease with an expected-head Git update before starting a new experiment. Record a unique worker ID, current UTC update time, a lease expiration no more than two hours ahead, and the exact next action. Renew during active work. At a normal stopping point, publish all reproducible work and set the status to `ready` or `blocked`, remove the active lease, and describe what remains. UTC timestamps are coordination metadata; user-facing schedules use America/Chicago.

Publish the hypothesis, primary comparison, controls, held-out split, success criteria, resource budget, and failure reporting before inspecting new held-out outcomes. Then publish the implementation before execution. Keep all seeds, failures, negative results, raw predictions and provenance. Separate tuning from evaluation. Use new evaluation content after an outcome-driven design change.

The frozen-message readout diagnostic and binding study are complete and permanently published. Read `docs/readout_results.md` and `docs/binding_results.md`. The binding run completed 18 fits and all 86,016 fresh predictions; its independent audit passed. Both aware latent arms scored zero target pairs correctly, while full text scored 99.54%. Do not repeat fitting or fresh scoring, relabel the negative result, or tune on that completed cohort.

The current separate investigation is fixed in `docs/alignment_numerical_design.md`, with its analytical basis in `docs/alignment_geometry.md`. The design was published before implementation or numerical outcomes. Follow its four-arm comparisons, fixed fixtures, rank policy, tolerances, eight-repetition timing order and 300-second child budget. Reviewed source is published with the bounded original workflow `.github/workflows/alignment-numerical-v1.yml`; see `docs/alignment_execution_notes.md`. Recover that run and its original artifact before proposing another numerical study. It concerns standard Procrustes/Gram identities specialized to an alignment interface, plus a distinct canonical optimum for nonunique cases. Require direct output equivalence only where the product is unique. Rank-deficient cases require independent optimal-objective and Gram checks; their canonical output may differ from an author's SVD null completion. Keep paper equations separate from fixed eigenfloors, norm floors, dtype and early-return behavior in code. Preserve every fixture result, failure, timing and packet cost. No matrix benchmark or constructed information-loss example establishes model accuracy, production speed, bandwidth savings over text, or scientific novelty. The mathematical note identifies established prior art and the narrow possible contribution.

Recover any active workflow and its artifacts before starting new work. Keep main unchanged while a publication workflow is committing an already verified artifact. Packaging and read-only audits do not authorize rerunning empirical outcomes.

The separate feasibility note `docs/lossless_predictor_feasibility.md` rejects starting another model study for vocabulary-byte XOR prediction with the current artifacts. Reopen only with the evidence and strong baselines specified there.

Use the existing tools and bounded local or standard public CPU workflows. Preserve all previous studies. If access, runtime or usage prevents progress, record the concrete blocker instead of fabricating a result. No worker can infer the user's remaining ChatGPT quota from these files.

## Finding labels

`discoveries/` indexes measured findings, including negative findings; the name does not assert that a method is new to science. `breakthroughs/` is reserved for a candidate supported by a precise claim, strong comparisons, appropriate independent replication, a relevant prior-art review, and reproducible evidence. Publication alone does not meet that threshold.
