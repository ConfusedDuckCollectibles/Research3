# Frozen-message readout results

**The diagnostic did not establish reliable information recovery from the learned 16-coordinate messages.** The primary query-aware linear probe improved paired accuracy over the original receiver by **2.08 percentage points**, with a descriptive 95% family interval of **[−0.78, +4.95] points**. That interval includes zero. Neither the linear probe nor the prespecified small MLP met the practical targets of 80% paired accuracy and 90% stability under an irrelevant fact edit.

This is a diagnostic result, not a breakthrough, a new communication method, or evidence of an efficiency advantage. The frozen [design](readout_design.md), original [generated report](../results/readout_study/report.md), and complete [machine-readable results](../results/readout_study/results.json) remain separate from this editorial interpretation.

## What was compared

The study reconstructed the previously trained DistilGPT2-to-Pythia-14M bridges, froze them, and trained additional readers on their outputs. The original receiver and the probes received identical cached messages. A successful probe could show that information was accessible to that additional supervised reader even when the original receiver failed. A failed probe cannot establish that the information was absent.

The 32 fresh test families excluded all 2,002 worlds reserved in earlier pretrained studies. Probe training and validation deliberately reused the preceding study's fitting families. Each family contains a target fact change and an irrelevant fact change, and is evaluated in four wording views. The same three seeds and family clusters are paired across conditions; wording views are not independent samples.

All linear checkpoints were selected by validation loss before any fresh scoring. The prespecified validation rule triggered the secondary small MLP for every message arm and seed; these checkpoints were also fixed before fresh scoring. The linear comparison remained primary.

## Original-template results

“Both correct” requires correct answers before and after the target fact changes. “Nuisance stability” means that the answer stays unchanged when an irrelevant fact changes; stability alone does not establish correctness.

| Reader or condition | Target pair both correct | Nuisance stability |
|---|---:|---:|
| Original query-aware receiver | 7.81% | 58.85% |
| Linear query-aware message probe | 9.90% | 50.26% |
| Linear query-blind message probe | 9.90% | 48.44% |
| Small MLP query-aware message probe | 13.54% | 47.92% |
| Linear probe before message compression, query-aware | 52.86% | 64.06% |
| Linear full-source-state probe | 82.03% | 90.10% |
| Adapted full-text receiver | 100.00% | 100.00% |

The query-aware and query-blind linear message probes had identical aggregate paired accuracy. Replacing the sender's public-question encoding changed only 3 of 1,152 original-view linear-probe choices and 4 of 1,152 MLP choices; their paired-correct counts did not change. These are descriptive intervention counts, and provide little support for useful question routing in this trained bridge.

Private source content does influence the message readers. Substituting another family's source lowered linear query-aware paired accuracy from 9.90% to 1.56%, and MLP paired accuracy from 13.54% to 2.08%. This dependence does not make either reader reliable.

## The major limitation: wording changes

**The full-source-state probe fell from 82.03% paired accuracy to 0% when the source wording and fact order changed while the question stayed unchanged, despite 100% nuisance stability.** Its joint source-and-question shift also scored 0%. The full-text receiver scored 100% on the source-only shift. High stability can therefore accompany consistently wrong answers, and the large probe's original-template success does not establish robust semantic access.

The full-source reader also has substantial extra capacity and positional access: **1,572,896 trainable parameters**, compared with **544** for a message linear probe and **3,232** for a message MLP. The precompression pooled probe has **49,184** parameters. These are diagnostic positive controls, not equal-capacity competitors. The probe head is selected by a deterministic parser of public question text, an explicit advantage over the original language-model receiver.

The gap between source-state and message performance is consistent with a binding or compression problem, but it does not uniquely locate the cause. Probe capacity, source positions, representation geometry, supervision, and template dependence differ. Failure of the fixed linear and small MLP readers rules out reliable recovery by those readers under this training budget; it does not prove irreversible information loss.

## Reconstruction and provenance

The behavioral reconstruction gate passed: all 41,472 historical prediction records, choices, and recorded losses reproduced exactly, with matching selected epochs, configuration, data, model, and historical source checks. Historical final-checkpoint hashes were not retained, so this does **not** establish byte identity with the original historical messages. Reconstructed checkpoints and feature hashes are retained for this cohort.

- Study workflow: [run 45335536056](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45335536056).
- Study job: [138246603169](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45335536056/job/138246603169).
- Study source: [9eb4478fae1ad373234eb2a725345b66c8c63c57](https://github.com/ConfusedDuckCollectibles/Research3/commit/9eb4478fae1ad373234eb2a725345b66c8c63c57).
- Reconstructed historical training source: [fb793ba31d7a10975a9e5c6032b44ecb5eb5774f](https://github.com/ConfusedDuckCollectibles/Research3/commit/fb793ba31d7a10975a9e5c6032b44ecb5eb5774f).
- Recorded execution: [execution.json](../results/readout_study/execution.json), [runtime manifest](../results/readout_study/runtime_manifest.json), and [reconstruction gate](../results/readout_study/reconstruction_gate.json).

The completed workflow passed 48 tests. Independent audits checked the retained-file hashes, 170,496 fresh prediction rows, all 740 condition metrics, 28 paired contrasts, 39 selected probe fits, and coherent source substitutions. The workflow took about 290 seconds including model downloads and reconstruction. This is experiment runtime, not a communication latency benchmark.

The complete artifact is named **readout-study-complete**, ID **14893873430**. Its ZIP SHA256 is **4e0f4a6499563ad72875f4e0a50e58f54659b57a1d77e93202e3823bf0044b56**. The publication helper copies only the fixed, checksum-verified outputs from this completed artifact. It performs no training, inference, outcome revision, or study rerun.

## Retained outputs

The repository retains [all compressed predictions](../results/readout_study/predictions.jsonl.gz), manifests, selection records, normalization statistics, and the original generated report. The redundant 42 MB uncompressed prediction file is retained only in the workflow artifact.

Three bounded exports are published permanently with the repository:

- [Reconstructed bridge weights](../results/readout_study/exports/bridge_weights.npz).
- [Message feature caches](../results/readout_study/exports/message_features.npz).
- [Small-probe weights](../results/readout_study/exports/small_probe_weights.npz).

All selected source/probe .pt checkpoints remain in the workflow artifact, with 90-day retention currently ending on **22 February 2027**. Larger source-state caches are reproducible from pinned models and retained manifests rather than uploaded in expanded form. Descriptive intervals are conditional on these three seed pairs and 32 families; they do not establish generalization across model families, tasks, or unrestricted language.
