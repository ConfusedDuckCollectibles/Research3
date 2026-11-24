# Running the public-query-conditioning follow-up

This implementation tests one fixed hypothesis after the completed negative
counterfactual study: giving the sender the public receiver question may help
its pooling mechanism preserve the relevant private fact. The frozen design
is in [query_conditioned_design.md](query_conditioned_design.md). Previous
studies and their results are retained unchanged. Receiver conditioning has
prior art; this is a controlled follow-up, not a new-method claim.

## What changes

The frozen sender encodes the complete private source and the public question
in **separate calls**. The question vector is the mean of the question's final
hidden states, normalized across features without learned affine parameters:

```text
q = (mean_token_state - mean_feature_value)
    / sqrt(population_feature_variance + 1e-5)
attention_query[k] = learned_base_query[k] + learned_scalar_gate[k] * q
```

This is LayerNorm-style normalization, not unit-L2 normalization. Each gate
starts at zero. The real-question and fixed-question arms therefore produce
identical messages at initialization for the same private source and bridge
weights. The only new trainable parameters are one scalar gate per slot.

The attention queries pool the private source states. The existing bottleneck
compresses the pooled states into two slots of eight float32 coordinates. The
receiver-side linear expansion turns these into two soft-prefix embeddings.
The frozen receiver scores eight fixed single-token color choices at the last
actual query position. The exact selected-head scorer handles different input
lengths and output-head bias; unit tests compare its logits and all bridge
gradients with the full vocabulary head.

| Trained arm | Private states | Sender question vector | Extra receiver context |
|---|---|---|---|
| `query_aware` | Actual source | Actual public question | None |
| `query_blind` | Actual source | Fixed normalized random vector | None |
| `private_constant` | Fixed position-varying random sequence | Actual public question | None |
| `text` | Same fixed sequence | Actual public question | Complete source text |
| `symbolic` | Same fixed sequence | Actual public question | Complete symbolic facts |

All five bridges have identical parameter shapes, initial weights, training
order and update counts within a seed. A fixed **nonzero, position-varying**
private sequence retains the attention mechanism's adaptation capacity. The
private-constant control also receives the actual public question, so extra
question-based adaptation is controlled for. Equal updates do not imply equal
FLOPs: text and symbolic receiver sequences have different lengths.

The trained query-aware bridge is additionally evaluated with a coherent
wrong-family private source, a wrong public-question vector, and a zero
compressed message. Wrong questions advance to the cyclic next object in the
same template; the receiver's actual question and the private source stay
unchanged. A frozen no-channel receiver is also reported.

## Fresh worlds and data isolation

`latent_bridge.query_conditioned_data` regenerates both earlier pretrained
datasets and excludes their union before creating new families:

- Previous counterfactual study: 960 base and single-edit worlds.
- First pretrained pilot: 112 worlds, including its wording-shift partition.
- Overlap: 30 worlds; total excluded: **1,042**.

The exclusion hash is
`4ef5acf31e0db8cc8fe507b7dcbbc16fcd93623f6072c66379c8323236eb366c`.
Its input is the UTF-8 encoding of a sorted unique list of world SHA256
identities, serialized as compact JSON with no final newline. The manifest
stores the complete identity list, exact prior generator parameters and all
new reserved identities by partition.

Seed 20261125 produces enough disjoint families for 128 training, 32 validation
and 32 test families. Every accepted family reserves its original world and
four single-key variants. All 960 newly reserved worlds are disjoint from the
1,042 old worlds and from every other new family. Four test renderings reuse
the same 32 test families; they are paired views, not independent datasets.

Training uses the original member only, with all four object questions.
Validation uses original and target-edited members. Testing also includes a
nuisance edit to an unqueried object. Labels are used only for supervision and
evaluation; source/question encoders receive strings alone. IDs, fact
dictionaries, `query_key` metadata and gold answers are not bridge features.
The structured parser is used only as a separate task-solvability check.

## Commands

Install this repository's optional transformer dependencies, then provide
existing local sender and receiver model directories. The runner never
downloads models and always uses `local_files_only=True` and
`trust_remote_code=False`. The root workflow handles pinned public model
downloads separately and runs all unit tests before training.

```bash
python -m latent_bridge.query_conditioned_data \
  --output local_data/query_conditioned \
  --seed 20261125 --train-families 128 \
  --validation-families 32 --test-families 32

python -m latent_bridge.query_conditioned \
  --sender-model local_models/sender \
  --receiver-model local_models/receiver \
  --data-dir local_data/query_conditioned \
  --output checkpoints/query-conditioned-study \
  --seeds 101 202 303 --epochs 8 --batch-size 16 \
  --slots 2 --bottleneck 8 --learning-rate 0.001 \
  --weight-decay 0.01 --max-length 192 --threads 2
```

The sender is pinned DistilGPT2 and the receiver pinned Pythia-14M, as specified
in the design. Both remain frozen and in evaluation mode. Gradients flow
through the receiver operations into the bridge during training. Sender
encodings run without gradients. Each bridge independently selects the epoch
with lowest validation restricted-choice cross-entropy. Held-out outcomes are
never used for model selection.

Outputs are `results.json`, `report.md`, `predictions.jsonl`, and selected
bridge checkpoints at `checkpoints/<seed>/<arm>.pt` under the requested output
directory. The wrapper retains compressed raw predictions, model/download
manifests, exact source revision, dataset manifest and execution metadata.
Checkpoints, model weights and data caches are not repository deliverables.

## Reading the measurements

The primary metric requires **both** the original and target-edited answers
to be correct. The primary contrast is original-view
`query_aware - query_blind`. An answer flip alone is insufficient. The
private-constant and zero-message controls have a structural zero for paired
correctness because their input is unchanged while the required answer
changes; beating that zero alone is weak evidence.

The adapted full-text validation accuracy must reach **80% in every seed**.
Failed seeds are labeled receiver-limited and retained. The additional
predeclared engineering targets are at least **80% original-view paired
accuracy** and **90% nuisance prediction stability**, averaged over the three
fixed seeds. Passing these targets would justify considering further work;
it would not establish novelty or communication efficiency. Smaller effects
are reported without weakening the targets.

The summary first averages a family's four query-pair scores across the three
fixed seeds. It then bootstraps the 32 families 2,000 times with seed 66173,
using the same sampled indices for every arm, view and paired contrast. The
95% percentile intervals are descriptive and conditional on those seeds.
Per-seed means and ranges are reported separately. Repeated views and seeds
do not increase the number of independent world clusters.

## Feedback and compute costs

The latent reply is 64 float32 payload bytes. The full public question's UTF-8
bytes are charged separately for a conservative request/reply accounting.
A question-conditioned message cannot be reused for a different question,
whereas a query-blind world message can. Framing, network latency and energy
are not measured. A two-byte packed representation can convey all four
categorical facts in this task; no bandwidth advantage follows from a small
latent alone.

Actual execution caches each distinct frozen source and each of the eight
public-question strings once across all splits, arms and seeds. No trained
bridge output is cached. Results record question vector means, variances,
norms and hashes, fixed-query seeds/hashes, actual encoder calls and token
counts, cache bytes, source/question preparation time, training/validation/
test time, receiver input lengths and public-feedback bytes by partition and
view. Logical uncached encoder calls/tokens are reported separately. Their
wall time is not estimated and is never presented as a measured result.

## Verification

Run the full suite with optional dependencies installed:

```bash
python -m unittest discover -s tests -v
```

New tests verify freshness even when someone recomputes a modified data file's
hash, immutable exclusion provenance, exact wrong-question mapping, the new
primary contrast and advancement thresholds, zero-gate equivalence, nonzero
initial gate gradients, public-query sensitivity after opening the gates,
private-constant pair invariance, query-blind invariance, intervention input
isolation, and full-head versus selected-head gradient agreement. At zero
initial gates the derivative with respect to the query vector is correctly
zero; the test checks query gradients only after setting nonzero gates.

As with the preceding studies, this remains a small synthetic lookup task on
one specific pretrained-model pair. A useful positive result would be a
bounded engineering finding. A negative result would constrain this fixed
implementation and budget, not prove that latent communication is impossible.
