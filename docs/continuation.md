# Research continuation

The owner has authorized continued investigation and publication in this repository. The objective is a defensible advance in useful learned communication. Experimental progress, causal evidence and scientific novelty must be assessed separately.

## Resume protocol

Read `research_state.json`, the latest README, frozen designs, and completed results before choosing work. Check current GitHub Actions and any available active-worker state. Respect a fresh worker lease; do not duplicate an active experiment or overwrite another worker's files. An expired lease is a recovery signal, not evidence that the previous experiment failed. Recover any completed workflow artifacts first.

Claim a new lease with an expected-head Git update before starting a new experiment. Record a unique worker ID, current UTC update time, a lease expiration no more than two hours ahead, and the exact next action. Renew during active work. At a normal stopping point, publish all reproducible work and set the status to `ready` or `blocked`, remove the active lease, and describe what remains. UTC timestamps are coordination metadata; user-facing schedules use America/Chicago.

Publish the hypothesis, primary comparison, controls, held-out split, success criteria, resource budget, and failure reporting before inspecting new held-out outcomes. Then publish the implementation before execution. Keep all seeds, failures, negative results, raw predictions and provenance. Separate tuning from evaluation. Use new evaluation content after an outcome-driven design change.

The next authorized task is the frozen-message readout diagnostic described in `docs/research_protocol.md`. It must distinguish information accessible in the compressed message from receiver-interface limitations, use matched training/control budgets, and retain wrong-source and nuisance interventions. A failed probe does not prove that no information is present. The earlier selected checkpoints were not published, so reconstruction and its verification must be explicit.

Use the existing tools and bounded local or standard public CPU workflows. Preserve all previous studies. If access, runtime or usage prevents progress, record the concrete blocker instead of fabricating a result. No worker can infer the user's remaining ChatGPT quota from these files.

## Finding labels

`discoveries/` indexes measured findings, including negative findings; the name does not assert that a method is new to science. `breakthroughs/` is reserved for a candidate supported by a precise claim, strong comparisons, appropriate independent replication, a relevant prior-art review, and reproducible evidence. Publication alone does not meet that threshold.
