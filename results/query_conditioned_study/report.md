# Public-query-conditioned private-fact study

Execution: **executed_pretrained_query_conditioned_study**. Interpretation: **competence_gates_passed_engineering_advancement_targets_not_met**.

The primary original-view query-aware minus query-blind paired-accuracy difference is 2.08%, with descriptive family-bootstrap interval [-0.005208333333333336, 0.04947916666666666].

Adapted text must reach 80% validation accuracy in each seed: {'101': 1.0, '202': 1.0, '303': 1.0}. Receiver-limited seeds: []. Every seed remains reported.

Predeclared advancement targets: paired accuracy at least 80%, nuisance stability at least 90%, and all text gates passing. Observed: 7.81% paired accuracy and 61.98% stability. Targets passed: **False**.

| View | Condition | Both target answers correct | Target record accuracy | Nuisance both correct | Nuisance stability |
|---|---|---:|---:|---:|---:|
| iid | aware_shuffled | 1.82% | 13.54% | 10.42% | 61.98% |
| iid | aware_wrong_question | 7.81% | 27.86% | 20.05% | 62.24% |
| iid | aware_zero | 0.00% | 11.72% | 11.46% | 100.00% |
| iid | frozen_no_channel | 0.00% | 12.50% | 12.50% | 100.00% |
| iid | private_constant | 0.00% | 12.24% | 12.50% | 100.00% |
| iid | query_aware | 7.81% | 27.86% | 20.05% | 61.98% |
| iid | query_blind | 5.73% | 28.65% | 20.57% | 63.54% |
| iid | symbolic | 23.44% | 39.84% | 36.46% | 75.00% |
| iid | text | 100.00% | 100.00% | 100.00% | 100.00% |
| joint | aware_shuffled | 0.00% | 12.24% | 10.68% | 95.83% |
| joint | aware_wrong_question | 0.00% | 12.11% | 10.68% | 95.57% |
| joint | aware_zero | 0.00% | 12.24% | 11.98% | 100.00% |
| joint | frozen_no_channel | 0.00% | 11.33% | 10.16% | 100.00% |
| joint | private_constant | 0.00% | 11.33% | 10.16% | 100.00% |
| joint | query_aware | 0.00% | 12.11% | 10.68% | 95.83% |
| joint | query_blind | 0.52% | 13.93% | 13.02% | 96.35% |
| joint | symbolic | 0.52% | 13.02% | 11.20% | 96.35% |
| joint | text | 0.00% | 13.93% | 13.54% | 97.92% |
| query | aware_shuffled | 0.26% | 13.67% | 10.94% | 80.47% |
| query | aware_wrong_question | 0.78% | 19.14% | 15.62% | 79.95% |
| query | aware_zero | 0.00% | 12.24% | 11.98% | 100.00% |
| query | frozen_no_channel | 0.00% | 11.33% | 10.16% | 100.00% |
| query | private_constant | 0.00% | 11.33% | 10.16% | 100.00% |
| query | query_aware | 0.78% | 19.01% | 15.62% | 80.47% |
| query | query_blind | 0.52% | 19.92% | 18.49% | 86.72% |
| query | symbolic | 0.52% | 13.02% | 11.20% | 96.35% |
| query | text | 0.00% | 12.63% | 13.02% | 97.92% |
| sender | aware_shuffled | 0.00% | 12.89% | 11.98% | 96.88% |
| sender | aware_wrong_question | 0.00% | 13.67% | 13.02% | 96.88% |
| sender | aware_zero | 0.00% | 11.72% | 11.46% | 100.00% |
| sender | frozen_no_channel | 0.00% | 12.50% | 12.50% | 100.00% |
| sender | private_constant | 0.00% | 12.24% | 12.50% | 100.00% |
| sender | query_aware | 0.00% | 13.67% | 13.02% | 96.88% |
| sender | query_blind | 1.04% | 13.28% | 10.42% | 92.97% |
| sender | symbolic | 23.44% | 39.84% | 36.46% | 75.00% |
| sender | text | 100.00% | 100.00% | 100.00% | 100.00% |

## Interpretation and costs

The primary comparison isolates sender access to the public question within the same trainable architecture, initialization and training budget. Inspect absolute accuracy, wrong-private-source and wrong-public-question interventions, nuisance invariance, and adapted full text before interpreting an improvement as useful fact transfer.

The private-constant control receives the real public question. Its input stays identical within a target pair while the correct answer changes, so both-correct is structurally zero. Beating that zero alone is weak evidence. Arbitrary prediction flips also do not satisfy paired correctness.

These are fixed eight-choice classifications, not open-ended generation. All four views reuse the same families. Descriptive intervals average queries and the three fixed seeds inside each family; they do not treat repeated views or seeds as independent worlds.

The public-question encoding is computed from question tokens alone. Actual and fixed query vectors use nonparametric LayerNorm-style normalization with epsilon 1e-5. Scalar query gates start at zero; all trained arms start with identical bridge weights. Source and question caches are charged explicitly.

The latent payload is 64 bytes. Original-view public questions average 24.00 UTF-8 bytes, charged separately on every request/reply. Question-conditioned messages cannot be reused for another question. Actual cached query encoder costs and logical uncached token work are reported separately; no hypothetical uncached wall time is presented as measured.

The study excludes all 1,042 unique worlds from the two earlier pretrained experiments before assigning its new families. Earlier outputs remain unchanged. This is a targeted follow-up on a small synthetic task with existing receiver-conditioning prior art, not a new-method or universal-language claim. Equal adaptation budgets do not imply equal FLOPs. A two-byte packed representation can convey these four categorical facts, so accuracy alone cannot establish an efficiency advantage.

Results below the practical targets are retained without relaxing thresholds or changing the architecture after seeing test outcomes.
