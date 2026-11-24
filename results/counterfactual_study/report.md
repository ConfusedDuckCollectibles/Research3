# Pretrained private-fact counterfactual study

Execution status: **executed_pretrained_counterfactual_study**. Interpretation: **all_seed_competence_gates_passed_comparative_evidence_still_required**.

The predeclared adapted-full-text validation gate is 80% in **each seed**. Seed accuracies: {'101': 1.0, '202': 1.0, '303': 1.0}. Receiver-limited seeds: []. No seed is excluded from the reported results.

These are eight-choice classifications, not open-ended generated-answer scores. They do not establish a universal language, real-world agent performance, or a speed/bandwidth advantage.

| View | Condition | Both target answers correct | Target record accuracy | Nuisance both correct | Nuisance prediction stability |
|---|---|---:|---:|---:|---:|
| iid | constant | 0.00% | 12.89% | 13.54% | 100.00% |
| iid | frozen_no_channel | 0.00% | 16.02% | 15.62% | 100.00% |
| iid | latent | 3.65% | 26.95% | 21.61% | 57.55% |
| iid | latent_shuffled | 0.78% | 14.06% | 8.85% | 57.55% |
| iid | latent_zero | 0.00% | 12.89% | 13.02% | 100.00% |
| iid | symbolic | 4.17% | 19.79% | 18.23% | 83.85% |
| iid | text | 100.00% | 100.00% | 100.00% | 100.00% |
| joint | constant | 0.00% | 12.11% | 8.59% | 100.00% |
| joint | frozen_no_channel | 0.00% | 12.11% | 8.59% | 100.00% |
| joint | latent | 0.00% | 15.49% | 10.68% | 91.67% |
| joint | latent_shuffled | 0.00% | 12.50% | 8.07% | 91.67% |
| joint | latent_zero | 0.00% | 11.46% | 9.64% | 100.00% |
| joint | symbolic | 0.00% | 11.33% | 8.33% | 94.79% |
| joint | text | 2.60% | 19.01% | 17.71% | 93.75% |
| query | constant | 0.00% | 12.11% | 8.59% | 100.00% |
| query | frozen_no_channel | 0.00% | 12.11% | 8.59% | 100.00% |
| query | latent | 0.52% | 18.10% | 14.32% | 71.09% |
| query | latent_shuffled | 0.00% | 13.02% | 7.81% | 71.09% |
| query | latent_zero | 0.00% | 11.46% | 9.64% | 100.00% |
| query | symbolic | 0.00% | 11.33% | 8.33% | 94.79% |
| query | text | 0.26% | 16.80% | 15.36% | 97.40% |
| sender | constant | 0.00% | 12.89% | 13.54% | 100.00% |
| sender | frozen_no_channel | 0.00% | 16.02% | 15.62% | 100.00% |
| sender | latent | 0.78% | 18.49% | 18.23% | 86.72% |
| sender | latent_shuffled | 0.00% | 12.63% | 11.20% | 86.72% |
| sender | latent_zero | 0.00% | 12.89% | 13.02% | 100.00% |
| sender | symbolic | 4.17% | 19.79% | 18.23% | 83.85% |
| sender | text | 94.01% | 96.09% | 95.31% | 98.18% |

## Reading the result

The primary outcome requires both answers in a single-fact edit pair to be correct. An arbitrary answer flip is insufficient. Constant/no-channel conditions have a structural zero on this primary measure: the query and model input do not change while the two correct answers differ. Beating that zero alone is weak evidence; inspect absolute accuracy, the matched-versus-shuffled drop, and the task-adapted text/symbolic controls.

A seed failing the full-text validation gate is labeled receiver-limited, regardless of the across-seed mean. A negative latent result in that setting is not evidence that useful latent communication is impossible. Passing the gate would still not itself establish superiority or novelty.

The JSON reports seed means/ranges and paired family-cluster intervals. Four views reuse the same families; queries, edits, and trained seeds are averaged inside each cluster. Intervals are descriptive and conditional on these three seeds.

The parser oracle is only a solvability check with complete facts. Four trained conditions have equal adaptation architecture, initial weights, training order, and update budget, but longer text/symbolic contexts consume more FLOPs. No runtime equality is implied.

All negative or positive results must be retained. No held-out outcomes were used to choose this study's fixed settings.
