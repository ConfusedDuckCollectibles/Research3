# Counterfactual communication study: fixed design

## Purpose and claim boundary

The first pretrained pilot did not demonstrate useful private-fact transfer.
Its strict generated-answer metric also confounded answer formatting with
retrieval. This study tests whether a learned message causes a receiver to
follow a specific change in a private fact, under comparable adaptation.

This is a causal-validity milestone. Learned communication, targeted fact
interventions, receiver conditioning, and trained information-free controls
already have prior art. Success here would not establish a new language or a
breakthrough. Four eight-valued facts contain only 12 bits: a packed symbolic
representation already fits in two bytes before framing. A 64-byte latent
message cannot claim a bandwidth advantage on this task.

This design, the implementation, and its workflow are committed before the
held-out outcomes are inspected. Any execution repair must be documented and
must not change hyperparameters in response to those outcomes.

## Frozen models and training budget

| Role | Model | Exact revision |
|---|---|---|
| Sender | `distilbert/distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` |
| Receiver | `EleutherAI/pythia-14m` | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |

Both pretrained models remain frozen. Only bridge/prefix parameters are
trained. Download safetensors and configuration/tokenizer files from the exact
public revisions; do not use remote model code or API credentials.

- Dataset seed: 20261124.
- Training seeds: 101, 202, 303.
- 128 training families, 32 validation families, 32 held-out test families.
- Eight epochs, batch size 16, AdamW learning rate 0.001.
- Two latent/prefix slots, eight bottleneck coordinates, float32.
- Maximum sequence length 192; reject overflow rather than truncate facts.
- CPU, two threads; workflow timeout 20 minutes.
- Each trained arm selects its own lowest-validation-loss checkpoint.
- Equal initial bridge parameters, example order, update count, optimizer,
  clipping and epoch budget across the four trained arms within each seed.

Equal update budgets do not mean equal FLOPs. Full-text and symbolic prefixes
are longer. Phase timings, message sizes, and the exact input sizes must be
reported separately, without a production efficiency claim.

## Private worlds and interventions

Each world assigns one of eight colors to each of four objects. A family
reserves a base world and four variants, each changing one object's color.
All reserved worlds are disjoint across families and train/validation/test
partitions. This includes variants that are not used for training.

For each queried object, the evaluation contains:

1. The original world.
2. A target intervention: only the queried object's color changes.
3. A nuisance intervention: only a different object's color changes, so the
   required answer stays the same.

The question remains identical across these members. Training uses the four
base-world questions from each training family: 512 examples. Validation uses
base and target members to select checkpoints. Each test view contains 128
query pairs across 32 families, plus their nuisance members. The independent
evaluation units are families, not the correlated individual questions.

The same held-out families are rendered in four views: original templates,
sender-only wording/order shift, query-only wording shift, and both shifts.
These are paired views of the same worlds, not four independent test datasets.
The generator records identities, answer frequencies, seeds and file hashes.
An exact structured-fact parser verifies labels and task solvability; it is a
software oracle, not a language-model communication result.

## Fixed-choice receiver interface

The receiver chooses among the same eight colors for every record. Each
leading-space color must tokenize to exactly one distinct receiver token; fail
clearly if the pinned tokenizer does not satisfy this assumption. Record IDs.

Compute the eight candidate logits at the final query position and train with
eight-way cross-entropy. No gold answer, gold length, or newline is supplied to
the receiver's input. This removes the completion-format confound in the first
pilot, while changing the task to constrained classification rather than open
generation. Preserve the earlier pilot unchanged.

The implementation may compute only the eight selected output-head rows after
the final query hidden state. A meaningful test must compare these logits and
bridge gradients against the full language-model output, including padding and
different input lengths, before using this optimization in the real study.

## Conditions

| Condition | Trainable adaptation and available facts |
|---|---|
| Latent | Learned bridge receives cached hidden states of the correct sender observation |
| Constant | Same bridge architecture receives one fixed seeded nonzero tensor, independent of every world and label |
| Adapted text | Same fixed-input bridge architecture supplies a learned prefix; complete text facts follow it |
| Adapted symbolic | Same fixed-input bridge architecture supplies a learned prefix; compact key/value facts follow it |
| Frozen no channel | Receiver sees only the query; no task adaptation |
| Latent zero | Trained latent message is replaced with zero before receiver-side expansion |
| Latent shuffled | Trained latent message comes from a different held-out family |

Within each seed, all four trained arms start with identical bridge weights.
The constant observation is nonzero and position-varying, preserving the
bridge's attention and slot capacity without communicating facts. Informative
text and symbolic baselines receive complete facts and the same number of
learned prefix positions. Their complete inputs are intentionally longer.

Source-family shuffling must be coherent across members and use a different
family. The constant arm cannot answer both members of a target pair correctly
when it receives the same query and the targets differ. Its pair score is
therefore structurally zero; merely beating that zero is weak evidence.

## Primary outcomes and interpretation gates

**Receiver competence gate:** the adapted full-text arm must reach at least
80% validation closed-choice accuracy in each seed. If a seed fails this gate,
report it and mark that seed's communication comparison receiver-limited. Do
not silently exclude it or change training to make it pass. Report all seeds
and all held-out conditions regardless of this gate.

**Primary test outcome:** target-pair both-correct accuracy: the original and
changed-fact answers must both be correct. This distinguishes correct tracking
from arbitrary prediction changes.

**Primary contrast and view:** on the original-template test view, compare
latent target-pair both-correct accuracy against latent-shuffled accuracy,
averaging across the three fixed seeds. Its paired family-bootstrap interval
is the primary interval. Other views, arms and metrics are descriptive
secondary analyses, with no unadjusted multiple-comparison success claim.
Use the same source-family shuffle mapping across the four paired views.

Secondary outcomes: original/target per-record accuracy; nuisance-pair both-correct accuracy;
nuisance prediction invariance; target change sensitivity; matched-versus-
shuffled paired differences; original-versus-shifted view differences; and
validation cross-entropy history. Report all conditions and views.

Evidence of useful transfer requires reliable target-pair accuracy, a loss
when source information is shuffled, and preserved answers under nuisance
changes, with the competent text control as context. A score above a constant
control alone, or a reduction in loss alone, is insufficient. Even a positive
result remains specific to this model pair, task, training budget and dataset.

For descriptive 95% percentile intervals, first average each family's score
over its four queried objects and the three fixed seeds, then bootstrap the
32 family clusters with 2,000 resamples using seed 66173. Paired comparisons use
the same sampled families. These intervals are conditional on the three
chosen training seeds; separately report seed means and ranges. Correlated
questions, views and seeds must not inflate the independent family count.

## Records and reproducibility

Retain results, compact individual predictions, a readable report, model and
dataset manifests, selected checkpoints' epoch numbers, source revisions,
package versions, and runtime phases. Publish compressed predictions with
checksums; do not publish downloaded model weights. No paid model API is used.

The runner is `latent_bridge.counterfactual_runner`. The fixed download/data/
execution wrapper is `scripts/run_counterfactual_study.py`, invoked by
`.github/workflows/counterfactual-study.yml`. The full unit suite runs with
PyTorch and Transformers available before the experiment.

## Following this study

Only after causal transfer is demonstrated should the project test efficiency
on richer evolving private evidence. That comparison must include compact
symbolic updates, caching, sender encoding, receiver work, setup and training
amortization, revision loss, recovery and factual correctness. A potentially
interesting question is reliable recipient-specific updates under uncertain
cache state; its novelty remains unestablished and requires further review.
