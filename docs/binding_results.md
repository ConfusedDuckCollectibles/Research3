# Binding-sensitive training: complete audited negative result

The fixed auxiliary loss did not establish useful transfer in this model pair. Both the matched CE-only and auxiliary-trained latent channels achieved **0 of 1,536 target pairs correct** on the fresh evaluation cohort. The adapted full-text control achieved **1,529 of 1,536 (99.54%)**. All 18 fits and all 86,016 fresh prediction records completed; the result is neither a timeout nor a receiver-competence stop.

The independent NumPy/standard-library auditor passed **1,056,048 recorded consistency checks**, with zero errors and no model execution. Its [complete receipt](../results/binding_study/audit_receipt.json), the [machine-readable results](../results/binding_study/results.json), and the [original retained evidence](../results/binding_study/) are permanent repository files. This audit validates the reported evidence within its stated scope; it is not an independent model-training replication.

## What was tested

The [frozen design](binding_design.md) specified the question before outcomes: can a fixed invariance/separation loss improve a learned message over the same architecture trained on the same edited examples using answer cross-entropy alone?

Every private world assigns the same eight colors to the same eight objects, each color exactly once. Consequently, color presence cannot reveal which object has which color. All fact orders are randomized. The exact sender and receiver token-ID bags were checked before fitting. The three partition sizes are 64 training families, 32 validation families and 64 fresh test families. Each family reserves its base world and all 28 single-swap neighbors; these 4,640 reserved worlds are disjoint between families.

For each of eight public questions, the data contain four views:

- Original world.
- A swap that changes the queried object's answer.
- A swap between two other objects, preserving that answer.
- The original world with a different fact order.

Target-pair correctness requires answering both the original and changed-answer questions correctly. Nuisance and reorder stability measure whether the answer stays the same; the corresponding both-correct metrics are reported separately.

The sender is frozen DistilGPT2 and the receiver is frozen Pythia-14m, at the immutable model revisions in the [model manifest](../results/binding_study/model_manifest.json). Each trained arm has 10,378 trainable bridge parameters and starts from the same per-seed initial state. The message contains two slots of eight float32 values: 64 payload bytes. Each fit uses eight epochs, 1,024 updates, and earliest minimum validation CE for checkpoint selection. Seeds are 101, 202 and 303.

The auxiliary arm adds an equally weighted loss that draws original/nuisance/reordered messages together and separates the original from the changed-answer message using a fixed squared-distance margin of 0.25. This is task-dependent relational supervision built from established invariance/contrastive ideas, not a claim to have invented those ideas. See the [prior-art review](prior_art_continuation.md).

## The full-information control passed first

The three text fits were trained and selected before any other arm. Their validation target-pair scores were 100%, 100% and 99.61%; nuisance/reorder stability also exceeded the fixed 90% thresholds. Thus the study proceeded under the prespecified rule. The receiver could solve this task when it received the full English facts through its ordinary input interface with the matched trained prefix.

All 18 selected checkpoints were persisted and verified before fresh model features were loaded. The independent auditor checked that selection evidence, reconstructed validation metrics and the complete fit/order records. File evidence alone is not an independent proof of real-time execution chronology.

## Fresh results

Each condition contains 2,048 records per seed, or 6,144 records and 1,536 target pairs across the three seeds.

| Condition | Target pair both correct | Record accuracy | Nuisance stability | Reorder stability |
|---|---:|---:|---:|---:|
| aware_ce | 0.00% | 12.53% | 99.48% | 97.59% |
| aware_aux | 0.00% | 12.13% | 99.61% | 97.27% |
| blind_ce | 0.00% | 12.27% | 99.61% | 98.05% |
| private_constant | 0.00% | 12.94% | 100.00% | 100.00% |
| text | 99.54% | 99.74% | 99.67% | 99.61% |
| symbolic | 0.46% | 13.23% | 91.21% | 65.17% |
| aware_ce_wrong_source | 0.00% | 12.37% | 99.48% | 97.59% |
| aware_ce_wrong_question | 0.07% | 12.55% | 99.35% | 97.92% |
| aware_ce_zero | 0.00% | 12.84% | 100.00% | 100.00% |
| aware_aux_wrong_source | 0.00% | 12.00% | 99.61% | 97.27% |
| aware_aux_wrong_question | 0.00% | 12.13% | 99.61% | 97.27% |
| aware_aux_zero | 0.00% | 12.84% | 100.00% | 100.00% |
| frozen_no_channel | 0.00% | 12.50% | 100.00% | 100.00% |
| rank_codec_oracle | 100.00% | 100.00% | 100.00% | 100.00% |

The primary auxiliary-minus-CE paired difference is **0.00 percentage points** for every seed and every observed family. The predeclared 2,000-resample family bootstrap therefore returns **[0.00, 0.00] points**. This degenerate interval reflects the observed all-zero paired scores. It does not establish that the unknown population effect is exactly zero. Intervals are descriptive over 64 whole families, conditional on the three fixed training seeds.

The auxiliary arm passes the nuisance- and reorder-stability thresholds but fails the 80% paired-accuracy requirement and the positive correct-source versus wrong-source paired-effect requirement. The practical-advance and supported-comparative-improvement decisions are both false.

## Why stability and ablation size would have misled us

Auxiliary-trained messages achieve 99.61% nuisance stability, but only **12.43% nuisance both-correct**. Their 97.27% reorder stability accompanies only **12.11% reorder both-correct**. Stable incorrect predictions are not useful invariance.

Zeroing the latent message changes **5,632 of 6,144 choices** for each of the two aware arms. Despite that large intervention effect, both normal arms still answer zero target pairs correctly. A learned prefix can substantially alter the receiver's biases without carrying useful information about the private answer. A large zero-message ablation by itself would not establish successful communication.

Replacing the private source changes 122 CE-arm choices and 153 auxiliary-arm choices out of 6,144; giving the sender the wrong question changes 29 and 1 respectively. The wrong-source intervention replaces the entire family source, including its fact order. It is not an isolated binding-only edit; the original-to-target pairs provide the controlled same-order answer-change test. The auditor verifies that wrong-source outputs match the corresponding normal-arm computation on the donor family.

The recorded selected-validation target distances for the auxiliary messages are approximately 0.00000201, 0.00000105 and 0.0000000584 across the three seeds. Their separation hinges remain almost the full 0.25 margin. These retained summaries are consistent with a failure to learn separation under this training recipe. Raw message activations were not retained, so the auditor checks bounds and loss algebra rather than independently reproducing those activation distances. These observations do not isolate whether optimization, architecture, features or supervision is the underlying bottleneck.

## Compression and cost comparison

The exact hand-written permutation-rank codec solves every pair and encodes a complete eight-object world in two bytes. It parses this task's strict grammar and is a task-specific oracle, separate from the trained symbolic-text language-model arm. The 64-byte latent payload is **32 times larger** than that complete-world code.

The English facts occupy 152 UTF-8 bytes; the symbolic rendering occupies 96. Public questions occupy 23–25 bytes, averaging 24, and the study charges the full question bytes for every query-aware request. The complete-world rank packet is reusable across questions. These payload figures exclude a common transport envelope and do not measure a production protocol.

The trained symbolic-input arm achieves only 0.46% paired correctness despite supplying all facts. Compact syntax is not automatically interpretable to this frozen receiver under the fixed prefix-training budget.

The entire empirical worker completed in **515.51 seconds**, including model downloads and retained study work, within its fixed 1,800-second deadline. Across all fits it recorded 357.33 seconds of training and 79.76 seconds of validation. Fresh evaluation recorded 29.56 seconds. Equal trainable parameters and updates did not equalize sequence lengths or compute: total text training took 117.44 seconds and symbolic training 110.54 seconds, versus 32.42 and 32.93 seconds for the two aware latent arms. Their accuracy is much worse, so these timings do not establish a useful speed/quality tradeoff.

Source and question features were cached. The retained accounting distinguishes logical requests from the 2,340 actual distinct source sequences and eight actual questions encoded. It reports the cache storage, checkpoint bytes and token/UTF-8 lengths. No network latency, energy usage or end-to-end production speedup was measured.

## Execution and evidence chain

| Stage | Immutable evidence |
|---|---|
| Design fixed before implementation/outcomes | [dfce06b](https://github.com/ConfusedDuckCollectibles/Research3/commit/dfce06b5dfb2860414af166ef4441eefd9548afc) |
| Pre-outcome fixes and all 90 offline tests passing | [Validation receipt](../results/binding_validation.json), [execution notes](binding_execution_notes.md) |
| Single empirical source and run | [3cfbf2b](https://github.com/ConfusedDuckCollectibles/Research3/commit/3cfbf2b02f70235f089d86d429da33ee818818d8), [run 45355476489](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45355476489) |
| Independent audit and publication | [run 45359713696](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45359713696), [configuration 167b2be](https://github.com/ConfusedDuckCollectibles/Research3/commit/167b2beafade9cd2c2ef7339d62840ba138d1665) |
| Permanent original outputs and audit | [ae43625](https://github.com/ConfusedDuckCollectibles/Research3/commit/ae436255f412b827d5d6c19350616f021516c982) |

The original artifact is **14902482655**, named `binding-study-45355476489-attempt-1`, with GitHub-reported archive SHA256 `04b07caadf59babdafa900f54193931cc7a637b64c36181817459507738c20cc`. It contains 69 study files plus the outer test log and is retained for 90 days.

Publication independently rehashed all 69 study files against the exact execution manifest recovered from the original job log. It permanently copied 67 original files, totaling **8,221,774 bytes**, plus three publication/audit records. All 40 NPZ files are retained: initial states, fixed controls, selected models and final in-progress states. Only the two uncompressed prediction duplicates were omitted from Git; their published gzip files reproduce the original bytes exactly. The outer test log remains in the original archive, with no independent file-hash claim.

The audit reconstructed data bytes and labels, all selected-validation and fresh metrics, whole-family bootstrap draws, epoch orders, initialization/selected-state hashes, accounting and intervention consistency. It imported no model library and performed no inference or fitting. It does not independently reproduce pretrained logits, per-record CE, tokenizer outputs, random PyTorch tensors or optimizer execution. The complete limitations are in its receipt.

Key permanent hashes:

- `results.json`: `18d2482626d039e4030f2cf8e2e27010b7084025a7fbd5675bce5d99dc9f6fee`
- `predictions.jsonl.gz`: `f26b186e3c308e81abf681e9b684d31cb7a327e6fcc248ab6a2ca09be6c81c83`
- `execution.json`: `5d3c5e45edf8ab147881493f5274bcfc06ab80eea74220b9e2731fb849a97dc3`
- `audit_receipt.json`: `a7106870dda3a576d41a1d842e19c9e0c28144fa94bbf42314d6ee42b4fd4a00`

## What this changes

The controlled auxiliary comparison failed for this fixed architecture, training schedule, model pair and data distribution. It does not establish that all latent communication fails. Scores from earlier studies are not controlled comparisons with this harder, differently constructed cohort.

A useful follow-up must change a specific limitation and define fresh success criteria before new outcomes. Repeating or retuning this held-out run would not provide independent evidence. The parallel mathematical investigation of per-message alignment is a separate line of work; its proofs or speed measurements, if validated, will not retroactively change this negative model result.

**No breakthrough, practical communication advance, or novel training method is claimed.**
