# Research3: learned latent communication

A research prototype for passing a small continuous message between a sender
and a receiver. This repository contains **an executed synthetic experiment**
and **an implemented, unexecuted local transformer experiment**. It does not
claim a new universal language, an LLM efficiency breakthrough, or a replication
of LatentMAS or Interlat.

## What is working

- A NumPy sender/compressor/receiver bridge with manual gradients and Adam.
- Distinct frozen sender and receiver coordinate systems, a learned bottleneck,
  and separate train/validation/test worlds.
- Causal controls that remove or replace the message; the receiver cannot see
  the private attributes anywhere else.
- Exact binary codecs and message-byte accounting, including a strong
  lossless **2-byte packed-symbol baseline**.
- A local-only Hugging Face training runner that freezes two base models,
  compresses sender hidden states, and injects continuous prefix embeddings into
  the receiver. It now measures generated-answer exact match as well as NLL,
  trains an equal-budget constant-information bridge, and tests unseen-world
  paraphrases. Optional transformer tests skip when dependencies are absent.

## Actual initial result

The synthetic task has four private attributes, each with eight possible values.
The sender sees the attributes before a receiver asks for any one of them. A
single learned vector must therefore support all four queries.

Across three bridge initializations, the 8-float channel reached approximately
**99.88% held-out query accuracy**. Replacing the message with zeros or another
world's message reduced performance to approximately chance (12.5%). The 4-float
channel was much more sensitive to initialization. See the complete
[executed report](results/synthetic.md) and [raw measurements](results/synthetic.json).

This demonstrates that the controlled receiver uses the learned channel. It
**does not demonstrate better bandwidth**: eight float32 values use 32 payload
bytes, while the lossless packed-symbol baseline uses 2 bytes and is 100%
accurate. Both must also carry the same 8-byte experimental frame header. The
task is deliberately favorable to semantic reconstruction, and a train-only
least-squares decoder is included to make that limitation visible.

No pretrained-model result is implied by these initial NumPy measurements.
The local environment lacked PyTorch, Transformers, and model weights. A
separately documented pretrained pilot must finish before its results can be
reported; a passing random-model gradient test is not that experiment.

## Run the executed experiment

Python 3.10+ and NumPy are sufficient. From this directory:

```bash
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m latent_bridge.experiment
python -m unittest discover -s tests -v
```

The full sweep uses widths 4, 8, and 16; seeds 101, 202, and 303; and 180 epochs.
It writes `results/synthetic.json` and `results/synthetic.md`. For a quicker
smoke experiment, use `--epochs 10 --seeds 101 --widths 8 --output results/quick.json`.
Changing the budget changes the experiment; do not compare its results as if it
were the published default run.

NumPy and Python versions, split hashes, seeds, fixed hyperparameters,
checkpoint selection, training times, message bytes, parameter bytes, and
training-storage estimates are recorded. No real network transfer, model-token
cost, energy measurement, or production inference speed is claimed.

## Local transformer experiment

The [local-model guide](docs/local_models.md) describes installation, JSONL
input format, exact metric semantics, and the command. The implementation is in
`latent_bridge/transformers_runner.py`.

The training module requires existing local model/tokenizer directories. Downloads and
remote model code are disabled. The sender and receiver can use different
hidden dimensions and tokenizers. Evaluation compares the correct message,
zeros, a genuinely different observation, no message, full sender text,
compact symbolic facts, and a separately trained constant-information bridge.
The last control receives the same architecture, initial weights, training
examples, update count, and optimizer as the matched bridge. Its one fixed,
seeded nonzero input is reused for every example and has no private facts or
source-length information. Distinct fixed positions retain attention-query
and slot adaptation capacity; an all-zero input would weaken that control.

**The upgraded runner reports both answer-token NLL and bounded greedy
generated-answer exact match.** Generation uses query tokens only, with global
newline/EOS/max-token stopping. It does not read target tokens or target length.
`private_lookup.py` generates a reproducible small task with complete-world
splits and an unseen-world paraphrase/order partition. This is still controlled
synthetic data; even success would not establish useful general agent reasoning.

## Research question and prior work

The [research protocol](docs/research_protocol.md) states the hypotheses,
controls, limitations, and next gates. Learned communication is a long-standing
research area; primary references include:

- Foerster et al., [DIAL/RIAL](https://arxiv.org/abs/1605.06676) (2016).
- Lowe et al., [On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168) (2019).
- [LatentMAS](https://arxiv.org/abs/2511.20639).
- [Interlat](https://arxiv.org/abs/2511.09149).
- [PLatour / PACT-Check](https://arxiv.org/abs/2611.06976), which motivates
  comparing against a trained bridge with no informative sender input.
- [A causal audit of relayed KV caches](https://arxiv.org/abs/2608.04893), which
  motivates testing whether sender-specific information actually matters.

This repository is an independent initial implementation for controlled
experiments, not a claim to have invented latent communication. Its companion
project, [Research2](https://github.com/ConfusedDuckCollectibles/Research2), tests
structured communication and shared task state through ordinary agent interfaces.
