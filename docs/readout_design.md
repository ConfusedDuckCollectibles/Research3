# Frozen-message readout diagnostic: fixed design

## Question and boundary

The completed query-conditioning study missed its practical targets, while
adapted full text solved the original-template task. This diagnostic asks
whether the selected frozen messages contain facts recoverable by a small
supervised readout that the original receiver fails to use.

The primary comparison is the original-view paired both-correct accuracy of
the **query-aware frozen-message linear probe minus the reconstructed original
query-aware receiver**, on exactly the same fresh worlds. This is an
accessibility diagnostic, not a fair claim that a new agent is superior: the
probe gets additional supervised decoder fitting and a deterministic parser
for the public question. No new communication method or novelty is claimed.

Only this design is prepared initially. Publish the design and implementation
before reconstructing the old run or inspecting any fresh held-out outcomes.
All settings below are fixed before those outcomes. Preserve all earlier
studies without changing their source, measurements or interpretation.

## Stage 1: reconstruct and verify the frozen checkpoint cohort

The prior selected bridge checkpoints were not retained with the published
results. Reconstruct them by executing the exact query-conditioned training
code from source revision
`fb793ba31d7a10975a9e5c6032b44ecb5eb5774f`, with the exact original data,
arguments, seeds, selection rule and pinned pretrained models. Run all five
original training arms, not a reduced or retuned reconstruction.

Reference run:

- Workflow `45329986439`, job `138228616914`.
- Published outputs: `results/query_conditioned_study/`.
- Original design SHA256:
  `25701024eaf35d112e2f4eae80f094370154abc041018e2e973233f9438f33c0`.
- Raw prediction JSONL SHA256:
  `324d8184f7730507da651eea0c49e16b646563d0746bc4d54f5747dfa798757c`.
- Sender: `distilbert/distilgpt2`, revision
  `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
- Receiver: `EleutherAI/pythia-14m`, revision
  `cf967c0a9a04383db6f7b1108d86b2962634b4ac`.
- Python `3.11.18`; PyTorch `2.6.0+cpu`, Transformers `4.48.2`, NumPy
  `2.4.6`, Tokenizers `0.21.4`, Safetensors `0.8.0`, Hugging Face Hub `0.36.2`.
- Original bridge seeds 101, 202 and 303; eight epochs; batch 16; AdamW 0.001,
  weight decay 0.01; clipping norm 1; two slots of eight float32 coordinates;
  CPU with two threads; maximum sequence length 192.

Record hashes for every old module used by the reconstruction and require
them to match the frozen source files. Keep reconstruction output in a new
directory; never overwrite the published reference results. The new wrapper
records its own revision separately from the old training implementation.

### Reconstruction acceptance gate

Before fitting probes or scoring fresh test worlds, require all of the
following:

1. Model file hashes, regenerated data hashes, candidate token identities,
   configuration, training-arm names, initial bridge hashes, selected epochs
   and epoch-order hashes match the reference exactly.
2. The complete 41,472-row old prediction grid has exactly the same row
   identities, gold labels and predicted choices. No condition or seed may be
   dropped to obtain agreement.
3. Every recorded per-record restricted cross-entropy satisfies
   `abs(new - old) <= 1e-5 + 1e-5 * abs(old)`.
4. Every recorded selected query gate satisfies
   `abs(new - old) <= 1e-6 + 1e-5 * abs(old)`.
5. Original validation accuracies and selected epochs match exactly; all
   original per-epoch validation cross-entropies satisfy the same CE
   tolerance above.

Also report whether the reconstructed raw prediction file's SHA256 matches
exactly. Timing fields are not expected to match. The tolerances are fixed
before reconstruction, not widened after seeing a difference.

If any gate fails, **stop before probe fitting and fresh-test scoring**, retain
the reconstruction and discrepancy report, and label it a failed
reconstruction. Do not silently retry, tune to old test scores, substitute a
different seed or treat that cohort as the accepted frozen run.

Even a passing gate cannot prove that historical messages were byte-identical:
the historical final checkpoint/message hashes do not exist. Call the cohort
a **verified behavioral reconstruction of the published run**, not the
identical historical messages. Save the reconstructed final checkpoints and
their tensor hashes before extracting probe features, then freeze them for
the entire diagnostic.

## Stage 2: fitting data and genuinely fresh test families

Reuse only the query-conditioned study's 128 training and 32 validation
families for probe fitting and selection:

- Training: 512 original-member records, four questions per world.
- Validation: 256 original/target-edit records, four question pairs per world.
- No old test world, target or nuisance member enters probe training or
  validation.

This reuses labeled data already used to train/select the bridge, which is
intentional and disclosed. It does not create additional fresh training data.
The new held-out evaluation is disjoint from every prior pretrained-study
world, including the reused fitting data and their reserved variants.

The exclusion set is the union of all previous pretrained experiments:

- 1,042 distinct worlds excluded before the query-conditioning study.
- All 960 base/variant worlds reserved by that study.
- Total: **2,002 distinct excluded worlds**.

Canonical exclusion hash:
`a34f2c6f1b1bf4602e9daea8c4feaacdd83272ebb98372b032a90cefa52499c1`.
Serialize sorted unique world-SHA256 strings as compact UTF-8 JSON, with
`separators=(",", ":")` and no final newline, before hashing.

Enumerate all 8^4 worlds, shuffle once with Python `random.Random(20261126)`,
and greedily accept a family only when its base world and four single-key
next-color variants intersect neither the exclusion set nor any accepted
family. Use the first **32 fresh families** for the new test set. A pure-data
preflight found that this fixed greedy generator accepts **53 families** with
seed 20261126, so it cannot produce a fresh 128/32/32 split under the declared
rule. This is not a proof of the maximum possible combinatorial packing. It
motivates the explicit fitting-data reuse without searching new packing rules.

Each fresh family has the same four public object questions, targeted edit
and nuisance edit as before. There are 128 target pairs and 384 member records
per view. Evaluate all four paired views: original templates, sender-only
wording/order shift, question-only shift and joint shift. These are the same
32 families rendered differently, not four independent test datasets.

Record the complete exclusion and new-reservation identities, hashes,
partition hashes, counts and label frequencies. Fresh test labels are used
only after every checkpoint and any secondary-reader decision are fixed.

## Stage 3: frozen representations and public-question interface

Both pretrained models and every reconstructed bridge remain frozen in
evaluation mode. No encoder, compressor, expander or original receiver
parameter receives a probe-training gradient. Representations are extracted
without gradients and cached before probe fitting.

The public-question parser is a declared exact-string mapping for the two
existing templates:

```text
The color of the <object> is
According to the record, the <object> has color
```

The four allowed objects are apple, book, cup and lamp. The selected probe
head comes **only from the actual public question text**. Do not read hidden
`query_key` metadata, fact dictionaries, world IDs, labels, expected-answer
positions or gold tokens as model features. Unexpected public wording fails
clearly; there is no fallback to metadata.

This parser supplies a known question identity and handles both declared
wordings. That is an explicit interface advantage over the original frozen
receiver's natural-language question processing, especially in the shifted
views. A probe advantage does not isolate that interface difference from all
other receiver-integration differences. The original-template result remains
primary; adapted complete text already established competence there.

### Primary probe representations

| Probe arm | Frozen input, excluding the public head selector | Input coordinates |
|---|---|---:|
| `message_aware` | Query-aware compressed message, two slots flattened in order | 16 |
| `message_blind` | Query-blind compressed message | 16 |
| `message_private_constant` | Reconstructed private-constant bridge message, retaining its public-query path | 16 |
| `query_only` | One seeded, fixed nonzero vector reused for every record | 16 |
| `message_aware_shuffled_labels` | Same query-aware message features, trained with the label control below | 16 |

The query-only raw vector is drawn once per bridge/probe pair with NumPy
`default_rng(bridge_seed + 56039).standard_normal(16)`, cast to float32. Its
input contains no private facts. After training-only centering it may be zero;
the four learned head biases still provide the complete available
question-only hypothesis class. This is a linear prior-control, not a
degenerate attention architecture.

For the training-label control, sort training family IDs and assign each
family the next family's original color labels for the **same public object
question**, wrapping at the end. Choose this derangement without consulting
label values. It preserves question identities and per-question label
frequencies while breaking the intended message/world correspondence.
Validation and fresh-test labels remain the true labels for every arm.
Report this control's training accuracy against both the labels it was given
and the actual world labels. It is a memorization/selectivity safeguard, not
a substitute for held-out source interventions.

### Larger linear diagnostic controls

| Probe arm | Frozen input | Input coordinates | Linear parameters |
|---|---|---:|---:|
| `pooled_aware` | Two query-aware attention-pooled sender states, immediately before compressor LayerNorm/projection/tanh | 1,536 | 49,184 |
| `pooled_blind` | Corresponding two query-blind attention-pooled states | 1,536 | 49,184 |
| `full_source` | Complete frozen sender last-hidden sequence, right-padded with zeros to 64 positions, then flattened | 49,152 | 1,572,896 |

The full-source reader retains every actual source position in original
order. Reject a source longer than 64 positions; do not truncate, select an
answer position or parse fact values into features. Padding uses only the
actual source length. The earlier pinned templates used 20 or 40 positions;
the new run verifies and records its actual lengths.

These controls use the same conditional-linear topology, label budget,
optimization rule and update count, but they have many more trainable input
weights and, for full source, explicit positional access. They are larger
diagnostic positive controls, not capacity-matched competitors. A successful
larger control and failed small message probe cannot prove compression
destroyed all information. No lossy projection is introduced merely to hide
the dimensional difference.

### Feature preprocessing

For each representation and bridge seed, calculate coordinate means and
population standard deviations from its 512 **training features only**. Use
the standard deviation as the scale when it is at least `1e-4`; otherwise use
scale 1. Apply

`z[j] = (feature[j] - training_mean[j]) / training_scale[j] / sqrt(input_dim)`.

All scales are positive, so this is a fixed reversible affine transformation
once its recorded statistics are supplied. It drops no feature. Inactive
padded coordinates use scale 1, avoiding amplification of previously unseen
positions. Reuse the same statistics on validation, fresh tests and
interventions. The shuffled-label arm shares the aware arm's statistics.
Save every mean, scale, input shape and representation hash.

## Probe architecture, budgets and conditional secondary reader

The primary readout has **four separate affine eight-class heads**, one for
each public object question. Equivalently it is a linear readout with an
explicit question-by-message interaction. A linear classifier on a simple
concatenation of message and question would only add a question-specific
bias and would not provide this selection mechanism.

Each 16-coordinate linear probe has **544 trainable parameters**. All five
message/control arms have identical shapes and initial parameter values
within each seed, and use identical shuffled batches. The larger controls
report their additional parameter counts rather than implying equal capacity.

Use three fixed bridge/probe pairs:

| Reconstructed bridge seed | Probe initialization/order seed |
|---:|---:|
| 101 | 4343 |
| 202 | 4444 |
| 303 | 4545 |

The probe seed is the bridge seed plus 4242. There is one probe initialization
per bridge seed, not nine independent combinations. Uncertainty is conditional
on these three paired seeds.

For every probe:

- 80 epochs, batch size 64: **640 optimizer updates** per arm and seed.
- AdamW learning rate 0.001, weight decay 0.01, gradient clipping norm 1.
- Eight-class cross-entropy over the fixed color ordering used before.
- At each epoch, evaluate true-label validation cross-entropy. Restore that
  arm's lowest-validation-CE checkpoint; ties retain the earlier checkpoint.
- No early stopping, test-based checkpoint selection, hyperparameter sweep,
  hidden-unit search or additional training after fresh results are inspected.
- Record each seed's shared initial-state hash, epoch-order hashes, training
  history, selected epoch, train/validation accuracy and all selected weights.

### Validation-only secondary-MLP trigger

After selecting **all primary linear checkpoints**, but before scoring any
fresh test record, evaluate their reused-validation paired both-correct
accuracy. If `message_aware` is below **80% in any seed**, train one fixed
secondary reader for **all five 16-coordinate arms and all three seeds**.
Otherwise omit the secondary reader and record why it was not triggered.

The secondary reader has four separate `16 -> 32 -> 8` heads with GELU between
the affine layers: **3,232 parameters**. It uses the same preprocessing,
training examples, seed pairing, optimizer, 80-epoch/640-update budget and
selection rule. Initial weights/order match across its five arms. Larger
pooled/full-source MLPs are not part of this study.

This is one prespecified capacity check, not a model search. Its trigger uses
validation only, runs all arms/seeds together, and cannot replace the linear
primary result. If linear validation passes but fresh performance fails,
there is no post-test MLP retry. A nonlinear probe result is always secondary.

## Frozen original receiver and interventions

Evaluate all nine previously declared original receiver conditions on the
fresh test families using the accepted reconstructed checkpoints, without any
new receiver/bridge adaptation. In particular retain query-aware, query-blind,
private-constant and adapted complete-text reference scores. These are fresh
outcomes of the reconstructed cohort, distinct from the published old test.

For every trained probe, evaluate its correct frozen representation and a
wrong-family representation. Map sorted test family IDs to their cyclic
successor, preserving public question, original/target/nuisance member and
view. The same family mapping is used across the four views and all probes.
It uses no answer values. Constant representations naturally remain constant.

For query-aware message probes, additionally substitute the cyclic next
object's sender-side public-question encoding in the frozen message generator
while keeping the probe head selected by the actual public question. This
tests sensitivity to the message generator's public-question path separately
from the probe's own question selector. Do not relabel the expected answer.

All conditions retain targeted single-fact and unqueried nuisance edits.
Report original/target both-correct, individual target-record accuracy,
targeted prediction-change rate, nuisance both-correct and nuisance stability.
Arbitrary answer changes do not count as correct communication.

## Analysis and interpretation fixed before outcomes

### Primary result

The primary contrast is:

`linear message_aware paired correctness - original query_aware receiver paired correctness`

on the **fresh original-template view**. Average each family's four query-pair
scores across the three paired seeds, then bootstrap the 32 family clusters
2,000 times with seed 66173. Use the same family resampling indices for all
contrasts and views. Report the difference and descriptive 95% percentile
interval in **percentage points**, with per-seed means and ranges separately.
Repeated queries, views and seeds do not increase the independent family count.

The aware-minus-blind probe difference, source-shuffle drop, pooled/full-source
controls, wrong sender-question results, MLP results and shifted views are
prespecified **secondary** analyses. They do not replace an unfavorable primary
contrast. Preserve all outcomes and failed controls. No universal or
multiplicity-adjusted significance claim is made.

### Competence and localization boundaries

Report the full-source linear probe's original/target validation record
accuracy separately for every seed, using 80% as its predeclared competence
threshold. A failure labels the corresponding probe diagnosis limited by its
positive-control readout; it does not suppress that seed or trigger a different
full-source architecture.

For any probe described as practically useful, retain the existing 80% fresh
paired-accuracy and 90% nuisance-stability targets, together with a drop under
wrong private source and the controls. Scores below these targets are still
reported; the thresholds are not relaxed after outcomes.

Interpret results as follows:

- A message probe recovering facts reliably while the original receiver fails
  establishes information accessible to that supervised readout. It directs
  attention to receiver integration, but does not show the original receiver
  uses the message or that no extra decoding training is required.
- A pooled-state probe outperforming a message probe narrows the limitation to
  the intervening transformation **or accessibility to these unequal-size
  readouts**. It does not prove irreversible information destruction.
- A full-source success with failed pooled/message probes shows that the larger
  positional readout can recover source facts. Pooling, compression and reader
  capacity remain possible causes; the controls do not uniquely identify one.
- Failure of the fixed linear and, if triggered, small MLP probes only limits
  those probe families, label budgets and optimization settings. It does not
  prove that the message contains no relevant information.
- High training performance with poor held-out or shuffled-label-control
  behavior indicates memorization or insufficient selectivity, not useful
  factual transfer.
- Public-question parser robustness is declared design knowledge. Superior
  shifted-question performance cannot be credited entirely to the latent
  representation.

## Compute, accounting and retained artifacts

Use a standard CPU GitHub Actions runner with two threads and a **20-minute
workflow limit**, including reconstruction, feature extraction, probes and
scoring. No paid API, unpinned model download or GPU is part of the design.
Record reconstruction, source extraction, feature preprocessing, each probe's
training/validation, fresh evaluation and overall wall times separately.
If execution fails or times out, retain partial outputs and report incomplete
coverage; do not reduce epochs/arms or silently retry after inspecting scores.

Account for the additional supervised decoder fitting, parameter counts and
input dimensions. Message probes read the same 64-byte float32 message; full
source and pooled-state readouts require much larger representations. Public
question feedback, query-specific message reuse limits, model/setup storage
and feature/probe cache sizes remain separate. This diagnostic makes no
communication-bandwidth or speedup claim.

Retain:

- Exact new design/implementation revision and dependency/model manifests.
- Original reference and separate reconstruction outputs, gate comparison
  report, original-input hashes and reconstructed checkpoint tensor hashes.
- All reconstructed selected bridge checkpoints, saved before probing.
- Frozen message caches, pooled/full-source cache hashes, feature metadata,
  public-parser definition and training-only normalization statistics.
- Fresh exclusion/reservation/partition manifests and all individual
  predictions, compressed without dropping failed arms or seeds.
- Every selected probe checkpoint, complete training histories, trigger
  decision, metrics, paired intervals and a readable qualified report.

Upload the complete checkpoint cohort and available caches as a GitHub Actions
artifact with **90-day retention**. Publish a bounded compressed repository
export of the reconstructed bridge checkpoints, 16-coordinate message caches
and small-probe weights, plus exact hashes and a reconstruction recipe for
larger full-source caches/weights. Full-source probe checkpoints belong in the
artifact; do not commit a large flattened full-source-state cache to the
repository or print it as encoded log output. Do not claim indefinite
retention for an expiring artifact. Report exactly which learned states are
durably retained in the repository versus retained for 90 days in the artifact.

## Implementation gates before execution

Tests must verify reconstruction comparison tolerances and failure handling;
fresh-family exclusions; parser-derived head selection without metadata;
frozen/no-gradient encoder boundaries; equal message-probe initialization and
budgets; training-only preprocessing; exact pre-compression feature location;
full-source padding/overflow handling; label-free coherent interventions;
constant/query-only invariance; validation-only MLP triggering; paired
metrics and family-cluster uncertainty. Fit no fresh test labels in a test or
warmup. Tests can use separate artificial tensors and tiny random local models.

Root publication order: freeze this design, implement and review the new
modules/tests, publish their revision, then run the fixed reconstruction and
diagnostic once. No new outcomes have been executed while drafting this design.
