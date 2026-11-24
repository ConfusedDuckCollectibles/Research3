# Public-query-conditioned communication: fixed follow-up design

## Motivation and boundary

The completed counterfactual study found a small correct-source effect, but
only 3.65% target-pair accuracy and 57.55% nuisance stability. Adapted full text
reached 100% on the original templates. The next question is whether giving
the sender's pooling mechanism the receiver's public question improves factual
selection under the same private observations and adaptation budget.

Receiver conditioning already has prior art. This is a hypothesis-driven
follow-up to a measured failure, not a claim to invent query-aware messaging.
The old datasets, predictions, source revisions and results remain unchanged.
This design and implementation must be committed before the new held-out
outcomes are inspected. No outcome-driven hyperparameter search is allowed.

## Models, task and fresh data

Use the same frozen pretrained models:

- Sender: `distilbert/distilgpt2`, revision
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
- Receiver: `EleutherAI/pythia-14m`, revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`.

Keep the four-object/eight-color task, targeted one-fact edits, nuisance edits,
eight fixed single-token receiver choices, and four paired rendering views.
The sender receives only its source text and the public receiver question.
No structured fact values, answer labels, expected-answer tokens or answer
length are supplied as encoder inputs or features.

Use dataset seed **20261125**. Exclude the union of all **960 reserved worlds**
from the previous counterfactual experiment and the **112 worlds** from the
first pretrained pilot, including its wording-shift partition. Their union
contains **1,042 distinct worlds**. Reserve each new family's base world and
four single-key variants before accepting it. No new family may intersect the
exclusion set or another new family. Record the canonical exclusion identities,
their hash and partition hashes. Preflight enumeration found enough families.

Assign **128 training, 32 validation and 32 test families**, using the same
original-only training and original/target validation protocol as the previous
study. All test views share the same new families. Training has 512 original
questions; validation has 256 original/target records; each test view has 128
target pairs plus nuisance members. Families remain the independent units.

## Sender-side question input

Encode the public question alone with the frozen sender model. Mean-pool its
last hidden states over actual tokens. Normalize the pooled vector with
nonparametric, zero-mean/unit-variance normalization, epsilon **1e-5**. This is
LayerNorm-style normalization without learned affine parameters, not unit-L2
normalization. Apply the same rule to the seeded nonzero fixed-query control.
Record vector norms, the fixed vector seed/hash and encoding calls.

For latent slot k, use:

`attention_query[k] = learned_base_query[k] + learned_scalar_gate[k] * query_vector`

Initialize every scalar gate to zero. At initialization, real and fixed public
query vectors therefore give identical message outputs for the same private
states. The two learned gates add the same parameter count to every trained
arm. The sender still pools the complete private source; it does not parse the
answer or select a fact using its ground-truth label.

## Trained arms and interventions

| Arm | Private source input | Sender question input | Receiver information |
|---|---|---|---|
| `query_aware` | Actual frozen source states | Actual public question vector | Learned message and actual question |
| `query_blind` | Actual frozen source states | One fixed seeded normalized vector | Learned message and actual question |
| `private_constant` | One fixed seeded nonzero sequence | Actual public question vector | Learned message and actual question |
| `text` | Same fixed private sequence | Actual public question vector | Learned prefix, complete text facts, actual question |
| `symbolic` | Same fixed private sequence | Actual public question vector | Learned prefix, complete symbolic facts, actual question |

All trained arms have the same architecture, initial weights, data order,
optimizer, clipping, epoch count and update count within each seed. The
no-private-information control must retain the real public question path, so
extra adaptation from the question alone cannot masquerade as fact transfer.

Evaluate additional interventions on the trained query-aware arm:

- **Wrong private source:** coherent successor-family substitution across
  original, target and nuisance members, keeping the public question correct.
- **Wrong public question:** encode the cyclic next object's question using
  the same template while leaving the private source and receiver's actual
  question unchanged.
- **Zero message:** replace the compressed message before receiver expansion.
- **Frozen no-channel receiver:** actual receiver question only.

Keep the same family substitution mapping across the four paired views. The
wrong-question rule is fixed and does not consult answers. Constant private
inputs stay identical across a target pair, so their paired both-correct score
is structurally zero; beating that zero alone is not a useful success claim.

## Fixed budget and selection

- Training seeds: **101, 202, 303**.
- Eight epochs, batch size 16, AdamW learning rate **0.001**, weight decay
  **0.01**, gradient clipping norm **1.0**.
- Two message slots, eight bottleneck coordinates, float32: **64 payload bytes**.
- Maximum sequence length **192**, with overflow rejection and no truncation.
- CPU with two threads; workflow timeout **20 minutes**.
- Each arm selects its lowest-validation restricted-choice cross-entropy
  checkpoint independently. No test-based selection.
- The adapted-text validation accuracy must be at least **80% in each seed**.
  Report failed seeds as receiver-limited and retain all their results.

The fixed architecture uses two scalar gates rather than searching a family
of query projections. Subsequent architecture changes require a separate
design and fresh evaluation data.

## Primary comparison and interpretation

The **primary outcome** is original-template target-pair both-correct accuracy.
The **primary contrast** is `query_aware - query_blind`, averaged across the
three fixed seeds. This isolates the value of the public question to the
sender within the equal-architecture experiment.

Use the same descriptive 95% family-bootstrap procedure: average each family's
four query scores across the three fixed seeds, resample the 32 families
2,000 times with seed **66173**, and reuse sampled indices for all paired
contrasts. Report seed means and ranges separately. These intervals are
conditional on the chosen seeds and do not justify population-level model
claims. Other arms, interventions and wording views are secondary analyses.

Interpret a positive primary effect alongside absolute accuracy, correct-
versus-wrong-source performance, the wrong-question intervention, nuisance
stability and the adapted text baseline. Public-query dependence alone does
not prove useful private-fact communication.

Before proceeding to efficiency claims, the engineering advancement target
is at least **80% original-template target-pair accuracy** and **90% nuisance
prediction stability**, with the text competence gates passing. These are
predeclared practical targets, not a definition of scientific novelty. A
smaller positive effect is retained as a limited result and does not trigger
posthoc relaxation of the targets.

## Feedback and computation accounting

Report the question's UTF-8 bytes separately from the 64-byte latent payload.
For a conservative request/reply accounting, charge the full public question
on each query-specific message. Record sender question-token counts, all
unique question encoding calls, query-vector cache storage and preparation
time. Cache hits are a stated execution optimization, not free initial setup.

The query-aware message depends on the question and cannot be reused across
different questions in the way a query-blind world message can. State that
limitation explicitly. Equal adaptation budgets do not imply equal FLOPs or
equal input lengths. This small task has a two-byte packed-fact representation;
success would still not demonstrate a bandwidth breakthrough.

## Validation and retained records

Before training, tests must verify initial zero-gate equivalence, meaningful
gate/query gradients, no access to answer features, fixed-input invariance
within target pairs, source-family and wrong-question intervention identity,
fresh-world exclusion, and the inherited exact selected-logit/gradient checks.
The full suite must run with PyTorch and Transformers installed.

Preserve results, all compressed individual predictions, the readable report,
model and data manifests, exclusion hashes, exact code revision, training
history, selected epochs, parameter/input sizes and phase timings. Publish
every condition and seed, including failures. No paid model API or unpinned
model download is part of this study.
