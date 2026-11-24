# Running a real local-model bridge

**Status:** implemented, syntax checked, and data-validation tests executed.
The initial local environment lacks PyTorch, Transformers, and pretrained
weights. CI can execute the tiny random-model tests; those tests alone are not
pretrained-model research results. Consult an executed pilot report for any
subsequently completed pretrained run.

## Install and validate

```bash
python -m pip install -e '.[transformers]'
python -m unittest discover -s tests -v
```

The optional test constructs a tiny random GPT-2 from configuration, without a
download. It verifies that answer-loss gradients reach bridge parameters while
the frozen receiver has no parameter gradients. A passing smoke test proves
gradient plumbing, not useful communication.

Use local model directories compatible with `AutoModel` for the sender and
`AutoModelForCausalLM` for the receiver. Each directory must include compatible
tokenizer files. The runner always sets `local_files_only=True` and
`trust_remote_code=False`; it does not download missing weights or execute
custom model code. Models are loaded in float32. Confirm sufficient RAM/VRAM
for both base models, the hidden-state cache, and receiver backward activations.

## Data

Supply three separate JSONL files with nonempty `id`, `sender`, `query`, and
`answer` strings. Example format only:

```json
{"id":"train-001","sender":"The sealed box contains a blue triangle.","query":"What color is the object? Answer:","answer":" blue"}
```

`sender` is the private information. `query` must not reveal the answer. Place
an intentional delimiter at the end of the query and leading whitespace in the
answer when appropriate for the tokenizer: prompt and answer are tokenized
separately to enforce a precise supervision boundary. This generic runner does
not apply a model-specific chat template automatically; provide the correct
serialized prompt format for the chosen model.

IDs must be unique. An identical sender observation cannot appear in more than
one split. An optional `world_id` also prevents the same facts appearing across
splits under different phrasings. An optional `symbolic` string supplies the
same complete facts in compact structured form for an additional baseline.
Repeated observations within a split are permitted for different
queries, but the test set must admit a cyclic derangement that gives every row
a genuinely different sender observation. Otherwise the runner rejects the data
instead of calling an unchanged message a causal intervention. Exact-string
checks do not detect paraphrases, memorized benchmark answers, or facts leaked
inside the query. Independent dataset review remains necessary.

No input is silently truncated. Sender observation, complete full-text baseline,
and latent-prefix-plus-query/answer must fit the configured maximum. Choose a
maximum supported by both models' actual context windows.

## Train and evaluate

To create the bounded private-lookup pilot data:

```bash
python -m latent_bridge.private_lookup --output local_data/private_lookup
```

It assigns 64 training worlds and 16 worlds each to validation, test, and
generalization. Every world has four queries. Generalization changes source
wording/order and the query template on entirely different worlds; it does not
test transfer to a new receiver model. `manifest.json` records hashes, exact
answer frequencies, generation settings, and metric definitions.

```bash
python -m latent_bridge.transformers_runner \
  --sender-model local_models/sender \
  --receiver-model local_models/receiver \
  --train local_data/private_lookup/train.jsonl \
  --validation local_data/private_lookup/validation.jsonl \
  --test local_data/private_lookup/test.jsonl \
  --generalization-test local_data/private_lookup/generalization.jsonl \
  --slots 2 --bottleneck 8 --epochs 3 --seed 101 \
  --max-length 128 --max-new-tokens 4 --threads 2 \
  --device cpu --output checkpoints/local-bridge
```

This freezes both base models, caches sender hidden states, and trains only
the bridge. The learned compressed message contains `slots × bottleneck`
float32 values. A receiver-side expansion converts it to `slots` soft tokens.
The receiver cannot obtain sender facts through a text side channel in the
matched, zero, permuted, or no-channel conditions.

The training labels mask the prefix and query with -100. The model shifts
causal labels internally. `eval()` suppresses base-model dropout but **does not
disable autograd through the receiver**. The sender is run under `no_grad()`;
its states are detached.

Two bridges are trained: one receives genuine hidden states; the other always
receives the same seeded standard-normal sequence of length eight. They have identical
architecture and initial weights, see the same examples in the same order,
and receive the same update count, learning rate, optimizer, and clipping.
The constant input conveys no facts or source-length information. Its distinct
fixed positions retain trainable attention queries and different slot outputs;
all-zero positions would collapse that adaptation capacity. The seed and input
tensor hash are recorded. Each bridge
selects its own lowest-validation-NLL checkpoint. This distinguishes useful
communication from task adaptation through a learned prefix.

Test conditions:

| Condition | Receiver information |
|---|---|
| `matched` | Correct learned compressed message plus query |
| `trained_constant` | Separately trained constant-information bridge plus query |
| `zero` | Zero compressed message, decoded through the learned adapter, plus query |
| `permuted` | Another private observation's compressed message plus the original query |
| `no_channel` | Query alone |
| `text` | Complete sender observation as text plus query |
| `symbolic` | The same complete facts in the provided compact syntax plus query |

All conditions use the same frozen receiver. The text/symbolic baselines are
not separately fine-tuned, so a result above them alone can reflect training
asymmetry. Compare against `trained_constant` before attributing gains to
communication. Four queries share one world; do not treat them as independent
worlds when calculating uncertainty.

The runner records **two separate metrics**. Teacher-forced answer-token NLL
allows earlier answer tokens to help predict later ones. Greedy generated
exact match starts from the query alone, stops at newline/EOS or the fixed
global token budget, and never receives gold-answer tokens or gold length.
The full generated answer must match after whitespace/case normalization;
punctuation and additional words remain significant. The decoding loop is
uncached and intentionally simple, not a production speed benchmark.

Outputs are `bridge.pt`, `constant_bridge.pt`, `predictions.jsonl`, and
`results.json`. Results include per-condition metrics, configuration, split
hashes, model-file hashes, versions, validation history, selected epochs,
update counts, byte/storage estimates, and separate load/cache/train/evaluate
times. Optional `--model-manifest` embeds a downloader's pinned revision
manifest. Outputs are ignored by git under `checkpoints/`; preserve selected
reports and predictions explicitly. Timing excludes external downloads and
must not be reported as optimized end-to-end inference latency.

The initial committed synthetic measurements are separate from these outputs.
Do not replace their status with a real-model success claim until the local
experiment has actually run and its controls have been inspected.
