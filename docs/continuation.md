# Research continuation

The owner has authorized continued investigation and publication in this repository. The objective is a defensible advance in useful learned communication. Experimental progress, causal evidence and scientific novelty must be assessed separately.

## Resume protocol

Read `research_state.json`, the latest README, frozen designs, and completed results before choosing work. Check current GitHub Actions and any available active-worker state. Respect a fresh worker lease; do not duplicate an active experiment or overwrite another worker's files. An expired lease is a recovery signal, not evidence that the previous experiment failed. Recover any completed workflow artifacts first.

Claim a new lease with an expected-head Git update before starting a new experiment. Record a unique worker ID, current UTC update time, a lease expiration no more than two hours ahead, and the exact next action. Renew during active work. At a normal stopping point, publish all reproducible work and set the status to `ready` or `blocked`, remove the active lease, and describe what remains. UTC timestamps are coordination metadata; user-facing schedules use America/Chicago.

Publish the hypothesis, primary comparison, controls, held-out split, success criteria, resource budget, and failure reporting before inspecting new held-out outcomes. Then publish the implementation before execution. Keep all seeds, failures, negative results, raw predictions and provenance. Separate tuning from evaluation. Use new evaluation content after an outcome-driven design change.

The frozen-message readout diagnostic is complete; its selected states, features and predictions are permanently published, and its small readers did not establish useful transfer. The current follow-up is the frozen binding study in `docs/binding_design.md`. First complete the separate pinned-runtime offline validation workflow. Publish the empirical workflow only after that gate passes; its single run/attempt must retain all outcomes and failures. All three full-text validation controls must pass before any latent/symbolic fits or fresh model scoring. A failed competence gate is a valid stopped study: preserve its three text fits, do not run the other arms or score the fresh cohort, and require a separate design for any new model/task/budget. If the gate passes, all 18 selected checkpoints must be written and verified before fresh model access. Do not rerun or selectively replace an empirical result. Recover complete or partial workflow artifacts before starting new research. Read `docs/prior_art_continuation.md`; the controlled auxiliary objective and causal audits use established ideas and do not by themselves support novelty.

Use the existing tools and bounded local or standard public CPU workflows. Preserve all previous studies. If access, runtime or usage prevents progress, record the concrete blocker instead of fabricating a result. No worker can infer the user's remaining ChatGPT quota from these files.

## Finding labels

`discoveries/` indexes measured findings, including negative findings; the name does not assert that a method is new to science. `breakthroughs/` is reserved for a candidate supported by a precise claim, strong comparisons, appropriate independent replication, a relevant prior-art review, and reproducible evidence. Publication alone does not meet that threshold.
