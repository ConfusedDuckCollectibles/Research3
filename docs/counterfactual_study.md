# Running the paired private-fact study

This guide describes implementation and reproduction. The separately committed
[fixed design](counterfactual_design.md) determines the frozen hypotheses,
models, data, budgets, gates, and uncertainty calculations. The earlier
negative pretrained pilot and its reports remain unchanged.

## What the study asks

Does the receiver track a specific change in a private fact? For each question,
it sees three possible sender worlds in separate evaluations:

- Original facts.
- Only the queried object's color changes: the required answer must change.
- A different object's color changes: the required answer must remain stable.

The question is identical across these three members. The main score requires
both the original and targeted-edit answers to be correct. A model does not
earn that score by making arbitrary prediction changes.

Four views reuse the same held-out world families: original templates,
source-only changes in wording/order, query-only rewording, and both changes.
They are paired presentations, not four independent datasets.

## Generate and validate the data

```bash
python -m latent_bridge.private_counterfactual \
  --output local_data/counterfactual \
  --seed 20261124 \
  --train-families 128 \
  --validation-families 32 \
  --test-families 32
```

The generator enumerates all four-object, eight-color worlds, shuffles them
once with the fixed seed, and chooses disjoint families. A family reserves its
original world plus all four one-object edits. Every reserved world is unique
across distinct families and train/validation/test partitions, including edits
not used in training.

Output files:

| File | Role |
|---|---|
| `train.jsonl` | 128 families × four questions; training uses original members only, 512 examples |
| `validation.jsonl` | 32 families × four pairs; original and targeted-edit members select checkpoints, 256 examples |
| `test_iid.jsonl` | 32 families × four target pairs and their nuisance members |
| `test_sender.jsonl` | Same held-out families, source wording/order changed |
| `test_query.jsonl` | Same held-out families, query wording changed |
| `test_joint.jsonl` | Same held-out families, both renderings changed |
| `manifest.json` | Exact identities, seed, rules, label frequencies, counts, and file hashes |

The loader verifies fact hashes, rendered source/query consistency, exact
target/nuisance edits, labels, four questions per family, cross-family world
separation, shared test families across views, and manifest hashes. A
deterministic parser checks that complete symbolic facts solve the task. This
100% parser oracle is only a solvability check, not an LLM or communication
performance result.

## Train and score accessible models

The runner loads existing local model/tokenizer directories, with downloads
and remote model code disabled. The outer reproduction wrapper handles pinned
public downloads. Use the exact model revisions in the fixed design.

```bash
python -m latent_bridge.counterfactual_runner \
  --sender-model local_models/sender \
  --receiver-model local_models/receiver \
  --data-dir local_data/counterfactual \
  --output checkpoints/counterfactual \
  --seeds 101 202 303 \
  --epochs 8 --batch-size 16 \
  --slots 2 --bottleneck 8 --learning-rate 0.001 \
  --max-length 192 --threads 2 --device cpu \
  --model-manifest local_models/manifest.json
```

`--model-manifest` is optional for manual use but the fixed reproduction wrapper
supplies it. Do not change the frozen settings after inspecting held-out scores.

The sender is frozen DistilGPT2; the receiver is frozen Pythia-14M. Eight
leading-space color strings must each tokenize to one distinct receiver token.
The runner rejects an incompatible tokenizer rather than changing its scoring
rule. Every example has the same eight candidate IDs, independent of the gold
answer.

All trained arms minimize the same eight-way cross entropy over the candidate
logits at the final query position. No answer token, answer length, newline,
or gold-dependent stopping rule enters the model input. This is constrained
classification, not free-form generation.

### Efficient scoring without a different objective

For the supported GPT-2/GPT-NeoX affine output heads, the runner obtains the
last query hidden vector from the frozen base model and projects it onto only
the eight selected output-head rows, including any output bias. This produces
the same eight logits as computing the entire vocabulary head and selecting
those rows. It avoids a large batch × sequence × vocabulary allocation.

Queries can have different lengths. Right-padding masks are validated, and
each example's last valid query position is selected separately. A test
compares both logits and every bridge-parameter gradient against the full LM
head on tiny GPT-2 and GPT-NeoX models, with different lengths and an explicit
head bias. The workflow must run that test before the real study.

## Comparable task adaptation

Four independently trained arms start from identical bridge parameters in
each seed. State hashes verify the equality. They receive the same batches in
the same order, update count, AdamW configuration, gradient clipping, and epoch
budget. Each chooses its own lowest-validation-cross-entropy checkpoint.

| Arm | Bridge input | Receiver context after the learned prefix |
|---|---|---|
| `latent` | Correct sender hidden states | Query |
| `constant` | One fixed seeded nonzero tensor for every example | Query |
| `text` | Same fixed tensor | Complete source text, then query |
| `symbolic` | Same fixed tensor | Complete compact key/value facts, then query |

The constant input has position-varying values so attention queries and slots
can adapt. It is independent of all worlds, labels, and source lengths. Text
and symbolic controls receive the same trainable prefix architecture and
training budget as the latent arm. They intentionally have longer complete
inputs, so equal update budgets are not equal FLOP budgets.

The frozen-no-channel condition has no learned prefix. The latent-zero
condition zeros the compressed message before receiver-side expansion. The
latent-shuffled condition uses the same trained bridge with a different
family's source. Shuffling uses sorted family identities, never labels, and
maps all original/target/nuisance members coherently. The same family mapping
is used across all rendering views. Matching colors can occur by chance even
between distinct families.

## Interpretation and uncertainty

The adapted full-text validation accuracy must reach **80% in each seed**.
Failing seeds are individually labeled receiver-limited; all their outcomes
remain in the report. An average above 80% cannot conceal a failed seed.

The primary contrast is original-template latent-minus-shuffled target-pair
both-correct accuracy. Other arms, views, and metrics are descriptive secondary
results. Constant/no-channel inputs are identical within each target pair,
while the two correct answers differ. Their target-pair score is structurally
zero; outperforming that zero alone is weak evidence. Examine absolute paired
correctness, the shuffled-source drop, nuisance selectivity, and the competent
adapted text/symbolic controls together.

For each family, average over its four queries and the three fixed training
seeds. Resample the 32 family clusters 2,000 times with seed 66173. The same
sampled family indices are used for every arm/view and paired contrast. The
95% percentile intervals are descriptive and conditional on these three
trained seeds. Seed means and ranges are reported separately; there is no
population-of-training-seeds or multiplicity-adjusted significance claim.

All views share families; all questions and interventions within a family are
correlated. None increases the independent world-cluster count.

## Results and provenance

The runner writes:

- `results.json`: configuration, candidate IDs, model and dataset hashes,
  training/validation histories, selected epochs, per-seed gates, all metrics,
  paired intervals, actual input lengths, message/storage accounting, and phase
  timing.
- `predictions.jsonl`: compact seed/view/condition/family/query/member rows with
  predicted candidate, gold candidate, and restricted cross entropy. Inputs
  are not duplicated per condition.
- `report.md`: readable interpretation and result tables.
- `checkpoints/<seed>/<arm>.pt`: learned bridge weights only. Do not publish
  downloaded model weights.

Sender states and static receiver embeddings are cached. Accounting reports
their actual storage and both logical per-record input lengths and unique
source extraction counts. Timings are for this reference implementation,
excluding external downloads, and do not establish a production speed gain.

Even success would be a controlled validity milestone. These four eight-valued
facts contain only 12 bits; a packed symbolic payload fits in two bytes, while
the configured latent payload is 64 bytes before framing. This experiment
cannot demonstrate a bandwidth advantage. A useful follow-on would require
richer tasks and complete sender/receiver/setup/recovery accounting, alongside
a separate prior-art and novelty assessment.
