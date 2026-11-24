# Frozen-message readout diagnostic

Reconstruction passed the predeclared behavioral gate. Historical message byte identity cannot be established because old final checkpoint hashes were absent.

The primary linear-aware probe minus original receiver difference is **+2.0833 percentage points**, with descriptive family interval **[-0.7812, +4.9479] points**.

Validation-only secondary MLP triggered: **True**. The linear primary was retained.

The probe receives extra supervised readout fitting and an explicit parser of public question text. Larger source/pool probes have more parameters. Results measure accessibility to these readers, not a superior agent, universal language or new method.

| View | Condition | Target pair both correct | Target record accuracy | Nuisance both correct | Nuisance stability |
|---|---|---:|---:|---:|---:|
| iid | linear_full_source | 82.03% | 88.67% | 86.98% | 90.10% |
| iid | linear_full_source_wrong_source | 6.25% | 7.03% | 6.25% | 90.10% |
| iid | linear_message_aware | 9.90% | 29.69% | 17.97% | 50.26% |
| iid | linear_message_aware_shuffled_labels | 1.30% | 13.02% | 6.77% | 63.54% |
| iid | linear_message_aware_shuffled_labels_wrong_source | 0.78% | 12.37% | 6.77% | 63.54% |
| iid | linear_message_aware_wrong_question | 9.90% | 29.82% | 18.23% | 50.52% |
| iid | linear_message_aware_wrong_source | 1.56% | 10.42% | 5.73% | 50.26% |
| iid | linear_message_blind | 9.90% | 32.16% | 19.27% | 48.44% |
| iid | linear_message_blind_wrong_source | 2.08% | 11.33% | 3.91% | 48.44% |
| iid | linear_message_private_constant | 0.00% | 10.81% | 9.64% | 100.00% |
| iid | linear_message_private_constant_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| iid | linear_pooled_aware | 52.86% | 66.02% | 53.91% | 64.06% |
| iid | linear_pooled_aware_wrong_source | 7.03% | 13.28% | 8.59% | 64.06% |
| iid | linear_pooled_blind | 57.55% | 71.61% | 61.20% | 66.67% |
| iid | linear_pooled_blind_wrong_source | 8.33% | 12.63% | 8.59% | 66.67% |
| iid | linear_query_only | 0.00% | 10.81% | 9.64% | 100.00% |
| iid | linear_query_only_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| iid | mlp_message_aware | 13.54% | 32.29% | 20.05% | 47.92% |
| iid | mlp_message_aware_shuffled_labels | 0.00% | 10.42% | 9.11% | 78.12% |
| iid | mlp_message_aware_shuffled_labels_wrong_source | 0.00% | 12.63% | 10.42% | 78.12% |
| iid | mlp_message_aware_wrong_question | 13.54% | 32.42% | 20.31% | 48.44% |
| iid | mlp_message_aware_wrong_source | 2.08% | 10.81% | 5.73% | 47.92% |
| iid | mlp_message_blind | 11.46% | 35.03% | 16.41% | 40.36% |
| iid | mlp_message_blind_wrong_source | 1.56% | 10.68% | 4.95% | 40.36% |
| iid | mlp_message_private_constant | 0.00% | 11.20% | 11.20% | 100.00% |
| iid | mlp_message_private_constant_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| iid | mlp_query_only | 0.00% | 11.20% | 11.20% | 100.00% |
| iid | mlp_query_only_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| iid | receiver_aware_shuffled | 1.30% | 11.98% | 5.99% | 58.85% |
| iid | receiver_aware_wrong_question | 7.81% | 25.78% | 16.93% | 58.59% |
| iid | receiver_aware_zero | 0.00% | 12.11% | 13.54% | 100.00% |
| iid | receiver_frozen_no_channel | 0.00% | 13.28% | 15.62% | 100.00% |
| iid | receiver_private_constant | 0.00% | 13.41% | 11.72% | 100.00% |
| iid | receiver_query_aware | 7.81% | 25.78% | 16.93% | 58.85% |
| iid | receiver_query_blind | 6.51% | 27.34% | 18.49% | 58.85% |
| iid | receiver_symbolic | 25.78% | 42.06% | 33.85% | 75.26% |
| iid | receiver_text | 100.00% | 100.00% | 100.00% | 100.00% |
| joint | linear_full_source | 0.00% | 13.67% | 10.16% | 100.00% |
| joint | linear_full_source_wrong_source | 0.00% | 13.67% | 10.16% | 100.00% |
| joint | linear_message_aware | 0.26% | 13.28% | 17.97% | 94.27% |
| joint | linear_message_aware_shuffled_labels | 0.00% | 12.24% | 13.54% | 89.32% |
| joint | linear_message_aware_shuffled_labels_wrong_source | 0.26% | 11.72% | 13.54% | 89.32% |
| joint | linear_message_aware_wrong_question | 0.26% | 13.41% | 17.97% | 94.53% |
| joint | linear_message_aware_wrong_source | 0.00% | 12.89% | 17.19% | 94.27% |
| joint | linear_message_blind | 0.52% | 13.41% | 14.58% | 90.89% |
| joint | linear_message_blind_wrong_source | 0.00% | 11.85% | 13.02% | 90.89% |
| joint | linear_message_private_constant | 0.00% | 10.81% | 9.64% | 100.00% |
| joint | linear_message_private_constant_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| joint | linear_pooled_aware | 0.78% | 19.01% | 20.31% | 93.49% |
| joint | linear_pooled_aware_wrong_source | 0.00% | 14.58% | 15.10% | 93.49% |
| joint | linear_pooled_blind | 2.08% | 20.44% | 19.01% | 90.36% |
| joint | linear_pooled_blind_wrong_source | 0.26% | 12.89% | 10.16% | 90.36% |
| joint | linear_query_only | 0.00% | 10.81% | 9.64% | 100.00% |
| joint | linear_query_only_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| joint | mlp_message_aware | 0.52% | 14.58% | 16.41% | 92.71% |
| joint | mlp_message_aware_shuffled_labels | 0.00% | 12.24% | 10.68% | 99.22% |
| joint | mlp_message_aware_shuffled_labels_wrong_source | 0.00% | 12.89% | 10.68% | 99.22% |
| joint | mlp_message_aware_wrong_question | 0.52% | 14.58% | 16.41% | 92.97% |
| joint | mlp_message_aware_wrong_source | 0.52% | 13.80% | 15.62% | 92.71% |
| joint | mlp_message_blind | 0.26% | 14.71% | 16.41% | 92.19% |
| joint | mlp_message_blind_wrong_source | 0.00% | 12.37% | 14.32% | 92.19% |
| joint | mlp_message_private_constant | 0.00% | 11.20% | 11.20% | 100.00% |
| joint | mlp_message_private_constant_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| joint | mlp_query_only | 0.00% | 11.20% | 11.20% | 100.00% |
| joint | mlp_query_only_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| joint | receiver_aware_shuffled | 0.00% | 13.28% | 10.42% | 96.35% |
| joint | receiver_aware_wrong_question | 0.00% | 13.41% | 10.68% | 96.35% |
| joint | receiver_aware_zero | 0.00% | 12.11% | 9.38% | 100.00% |
| joint | receiver_frozen_no_channel | 0.00% | 12.50% | 9.38% | 100.00% |
| joint | receiver_private_constant | 0.00% | 12.50% | 9.38% | 100.00% |
| joint | receiver_query_aware | 0.00% | 13.41% | 10.68% | 96.35% |
| joint | receiver_query_blind | 0.26% | 13.54% | 11.72% | 94.01% |
| joint | receiver_symbolic | 0.26% | 13.80% | 10.16% | 94.27% |
| joint | receiver_text | 0.00% | 14.32% | 15.62% | 98.96% |
| query | linear_full_source | 82.03% | 88.67% | 86.98% | 90.10% |
| query | linear_full_source_wrong_source | 6.25% | 7.03% | 6.25% | 90.10% |
| query | linear_message_aware | 9.90% | 29.56% | 17.97% | 50.00% |
| query | linear_message_aware_shuffled_labels | 1.30% | 13.02% | 6.51% | 62.76% |
| query | linear_message_aware_shuffled_labels_wrong_source | 0.78% | 12.24% | 6.77% | 62.76% |
| query | linear_message_aware_wrong_question | 10.16% | 29.43% | 17.97% | 50.00% |
| query | linear_message_aware_wrong_source | 1.56% | 10.55% | 5.73% | 50.00% |
| query | linear_message_blind | 9.90% | 32.16% | 19.27% | 48.44% |
| query | linear_message_blind_wrong_source | 2.08% | 11.33% | 3.91% | 48.44% |
| query | linear_message_private_constant | 0.00% | 10.81% | 9.64% | 100.00% |
| query | linear_message_private_constant_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| query | linear_pooled_aware | 52.34% | 65.76% | 52.86% | 63.02% |
| query | linear_pooled_aware_wrong_source | 7.03% | 13.15% | 8.59% | 63.02% |
| query | linear_pooled_blind | 57.55% | 71.61% | 61.20% | 66.67% |
| query | linear_pooled_blind_wrong_source | 8.33% | 12.63% | 8.59% | 66.67% |
| query | linear_query_only | 0.00% | 10.81% | 9.64% | 100.00% |
| query | linear_query_only_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| query | mlp_message_aware | 13.28% | 32.16% | 20.31% | 49.22% |
| query | mlp_message_aware_shuffled_labels | 0.00% | 10.29% | 9.11% | 77.86% |
| query | mlp_message_aware_shuffled_labels_wrong_source | 0.00% | 12.50% | 10.42% | 77.86% |
| query | mlp_message_aware_wrong_question | 13.28% | 32.42% | 20.31% | 49.22% |
| query | mlp_message_aware_wrong_source | 2.08% | 10.94% | 5.47% | 49.22% |
| query | mlp_message_blind | 11.46% | 35.03% | 16.41% | 40.36% |
| query | mlp_message_blind_wrong_source | 1.56% | 10.68% | 4.95% | 40.36% |
| query | mlp_message_private_constant | 0.00% | 11.20% | 11.20% | 100.00% |
| query | mlp_message_private_constant_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| query | mlp_query_only | 0.00% | 11.20% | 11.20% | 100.00% |
| query | mlp_query_only_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| query | receiver_aware_shuffled | 0.26% | 13.67% | 9.38% | 84.11% |
| query | receiver_aware_wrong_question | 1.04% | 17.45% | 13.54% | 84.38% |
| query | receiver_aware_zero | 0.00% | 12.11% | 9.38% | 100.00% |
| query | receiver_frozen_no_channel | 0.00% | 12.50% | 9.38% | 100.00% |
| query | receiver_private_constant | 0.00% | 12.50% | 9.38% | 100.00% |
| query | receiver_query_aware | 1.04% | 17.45% | 13.54% | 84.11% |
| query | receiver_query_blind | 1.04% | 18.62% | 13.02% | 83.59% |
| query | receiver_symbolic | 0.26% | 13.80% | 10.16% | 94.27% |
| query | receiver_text | 0.00% | 12.76% | 12.24% | 98.44% |
| sender | linear_full_source | 0.00% | 13.67% | 10.16% | 100.00% |
| sender | linear_full_source_wrong_source | 0.00% | 13.67% | 10.16% | 100.00% |
| sender | linear_message_aware | 0.00% | 13.28% | 17.97% | 94.27% |
| sender | linear_message_aware_shuffled_labels | 0.00% | 12.11% | 13.80% | 89.06% |
| sender | linear_message_aware_shuffled_labels_wrong_source | 0.26% | 11.72% | 13.54% | 89.06% |
| sender | linear_message_aware_wrong_question | 0.00% | 13.28% | 17.97% | 94.27% |
| sender | linear_message_aware_wrong_source | 0.00% | 12.89% | 17.19% | 94.27% |
| sender | linear_message_blind | 0.52% | 13.41% | 14.58% | 90.89% |
| sender | linear_message_blind_wrong_source | 0.00% | 11.85% | 13.02% | 90.89% |
| sender | linear_message_private_constant | 0.00% | 10.81% | 9.64% | 100.00% |
| sender | linear_message_private_constant_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| sender | linear_pooled_aware | 0.52% | 18.88% | 20.31% | 92.97% |
| sender | linear_pooled_aware_wrong_source | 0.00% | 14.58% | 15.10% | 92.97% |
| sender | linear_pooled_blind | 2.08% | 20.44% | 19.01% | 90.36% |
| sender | linear_pooled_blind_wrong_source | 0.26% | 12.89% | 10.16% | 90.36% |
| sender | linear_query_only | 0.00% | 10.81% | 9.64% | 100.00% |
| sender | linear_query_only_wrong_source | 0.00% | 10.81% | 9.64% | 100.00% |
| sender | mlp_message_aware | 0.52% | 14.71% | 16.41% | 93.49% |
| sender | mlp_message_aware_shuffled_labels | 0.00% | 12.24% | 10.68% | 99.22% |
| sender | mlp_message_aware_shuffled_labels_wrong_source | 0.00% | 12.89% | 10.68% | 99.22% |
| sender | mlp_message_aware_wrong_question | 0.52% | 14.71% | 16.41% | 93.75% |
| sender | mlp_message_aware_wrong_source | 0.52% | 13.80% | 15.89% | 93.49% |
| sender | mlp_message_blind | 0.26% | 14.71% | 16.41% | 92.19% |
| sender | mlp_message_blind_wrong_source | 0.00% | 12.37% | 14.32% | 92.19% |
| sender | mlp_message_private_constant | 0.00% | 11.20% | 11.20% | 100.00% |
| sender | mlp_message_private_constant_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| sender | mlp_query_only | 0.00% | 11.20% | 11.20% | 100.00% |
| sender | mlp_query_only_wrong_source | 0.00% | 11.20% | 11.20% | 100.00% |
| sender | receiver_aware_shuffled | 0.00% | 12.63% | 12.24% | 95.31% |
| sender | receiver_aware_wrong_question | 0.00% | 13.28% | 12.24% | 95.57% |
| sender | receiver_aware_zero | 0.00% | 12.11% | 13.54% | 100.00% |
| sender | receiver_frozen_no_channel | 0.00% | 13.28% | 15.62% | 100.00% |
| sender | receiver_private_constant | 0.00% | 13.41% | 11.72% | 100.00% |
| sender | receiver_query_aware | 0.00% | 13.28% | 12.24% | 95.31% |
| sender | receiver_query_blind | 0.78% | 14.06% | 12.50% | 92.97% |
| sender | receiver_symbolic | 25.78% | 42.06% | 33.85% | 75.26% |
| sender | receiver_text | 100.00% | 100.00% | 100.00% | 100.00% |

## Limits and retention

The 32 fresh families exclude all 2,002 prior pretrained-study worlds. Probe training/validation explicitly reuse the preceding study's fitting families. Four views and three paired seeds are averaged within family clusters, not counted as independent worlds.

Full-source linear validation competence (80% record accuracy per seed): {'threshold': 0.8, 'per_seed_accuracy': {'101': 0.88671875, '202': 0.8828125, '303': 0.87890625}, 'limited_seeds': [], 'passed': True}. Failed seeds are retained.

Reliable message-probe recovery with poor original-receiver performance shows information accessible to that additional reader. A failed linear or small MLP probe does not prove information is absent. Larger raw/pool controls cannot uniquely attribute a gap to irreversible compression loss because their capacity and positional access differ.

Use the prespecified 80% fresh paired accuracy and 90% nuisance stability targets together with wrong-source controls when assessing practical utility. These targets do not define scientific novelty. The shuffled-training-label control and all negative outcomes remain reported.

Compressed bridge tensors, 16-coordinate feature caches and small-probe weights are exported for durable repository publication. Larger checkpoints and normalization statistics are retained in the workflow artifact; its 90-day retention is finite. Full-source caches are represented by hashes and a regeneration recipe rather than a large repository upload.

No paid API, production-latency measurement or bandwidth claim is part of this diagnostic.
