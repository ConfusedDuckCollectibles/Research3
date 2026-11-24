# Paired fact-change study: a small source-specific effect, unreliable transfer

The corrected, three-seed experiment detects a small effect of supplying the
correct sender information. It also shows that this latent bridge is far less
reliable than an equally adapted full-text baseline. This is a controlled
finding within one small synthetic task, not a new communication method or a
breakthrough.

- [Executed workflow](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45325792109)
- [Exact executed source](https://github.com/ConfusedDuckCollectibles/Research3/tree/9df6e71a74ee2ffce4e52c51ee453e52e4adc74c)
- [Design published before outcomes](../../docs/counterfactual_design.md)
- [Complete measurements](results.json), [generated all-condition report](report.md),
  [compressed individual predictions](predictions.jsonl.gz),
  [data manifest](dataset_manifest.json), [model manifest](model_manifest.json),
  and [execution details](execution.json)

## What changed from the first pilot

The receiver now selects among eight fixed color tokens. All four trained arms
receive identical bridge architecture, initial parameters, example order,
optimizer and update count. Full text and compact symbolic facts receive task
adaptation too. No gold answer tokens or answer length enter the model input.

Each held-out question has an original world, a version changing only the
queried fact, and a nuisance version changing a different fact. Entire families
are isolated across training, validation and test. Four views separate sender
wording changes from question wording changes. The models remain frozen
DistilGPT2 and Pythia-14M, connected by a trained 64-byte float32 message.

All three adapted-text validation scores reached **100%**, passing the
predeclared 80% competence gate in every seed. The receiver can solve the
original-template task when given complete adapted text.

## Primary result

The primary measure requires both the original and targeted-change answers to
be correct. The prespecified comparison is correct latent messages versus
messages from another family, on the original templates.

| Condition | Both target-pair answers correct | Individual target-record accuracy |
|---|---:|---:|
| Correct latent message | **3.65%** | **26.95%** |
| Wrong-family latent message | **0.78%** | **14.06%** |
| Trained constant private input | 0% | 12.89% |
| Zeroed latent message | 0% | 12.89% |
| Frozen receiver, query only | 0% | 16.02% |
| Adapted compact symbolic facts | 4.17% | 19.79% |
| Adapted complete text | **100%** | **100%** |

The primary difference is **+2.86 percentage points**, with the prespecified
descriptive family-bootstrap interval **[+0.52, +5.47] percentage points**.
The difference is positive in each seed: +3.125, +3.125 and +2.34375 points.

Across the three seeds, the correct latent channel answers 14 of 384 pair
evaluations completely correctly, compared with three shuffled evaluations.
Those evaluations share only **32 independent held-out world families**. The
interval resamples families after averaging their questions and the three
fixed seeds; it does not establish uncertainty over a population of models,
training seeds or real-world tasks.

Constant-input and no-channel conditions have a structurally zero paired score:
their input stays the same while the required answers differ. Beating that
zero alone would be weak evidence. The source-shuffling comparison provides
the more informative intervention. Absolute latent performance remains poor,
and full text wins decisively.

## Relevant versus irrelevant edits

The latent prediction changes on **43.75%** of targeted edits and **42.45%** of
nuisance edits. Its nuisance prediction stability is only **57.55%**, and both
nuisance-pair answers are correct only **21.61%** of the time. The channel is
sensitive to input changes but poorly selective about which changes matter.

This motivates testing sender-side selection using the receiver's public
question. That follow-up must use new held-out worlds and a separate frozen
design; these results must not be retuned or relabeled. Receiver conditioning
already has prior art, reviewed in [the literature note](../../docs/prior_art_state_updates.md).

## Wording and order shifts

| View | Latent target-pair accuracy | Adapted-text target-pair accuracy |
|---|---:|---:|
| Original templates | 3.65% | 100% |
| Sender wording/order changed | 0.78% | 94.01% |
| Question wording changed | 0.52% | 0.26% |
| Both changed | 0% | 2.60% |

The source-only shift favors text strongly. A question change harms both
methods, so those failures also reveal brittleness in the receiver/prompt
interface. These are paired views of the same worlds and descriptive
secondary results, not four independent discoveries.

## Costs, verification and limits

The latent payload remains 64 bytes before framing; complete text is about
75 bytes and compact symbolic facts about 43 bytes on this task. Four
eight-valued facts can also be packed into two bytes with a deterministic
codec. No bandwidth advantage follows from this study. Equal update budgets
also do not imply equal FLOPs because input lengths differ.

The runner took **114.39 seconds**, or **121.83 seconds** including pinned model
downloads and wrapper preparation. Workflow provisioning and dependency
installation are excluded. These are whole experiment durations, not measured
communication speedups or production latency.

All **25 tests passed** in the execution environment before training, including
selected-output-logit and bridge-gradient equivalence for both GPT-2 and
GPT-NeoX. The optimized eight-row output calculation is a computation-saving
implementation detail, not a new scientific result.

All 32,256 condition/member predictions were preserved. Report and prediction
SHA-256 hashes were verified after recovery from workflow logs. Independent
recalculation reproduces the primary interval exactly; every shuffled
prediction matches the corresponding correct-message prediction under the
declared source-family mapping. Model and data revisions are recorded.

The study establishes a narrow source-specific effect under controlled
conditions. It does not establish reliable useful communication, efficiency,
universal transfer, a novel algorithm or a scientific breakthrough.
