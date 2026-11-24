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
  the receiver. Its optional smoke test is skipped when dependencies are absent.

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

No pretrained-model experiments were run for this release: this environment
did not have PyTorch, Transformers, local model weights, or provider API keys.

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

It requires existing local model/tokenizer directories. Model downloads and
remote model code are disabled. The sender and receiver can use different
hidden dimensions and tokenizers. Only the bridge is trained. Evaluation
compares the correct message, zeros, a genuinely different observation,
no message, and the full sender observation supplied as text.

**The runner measures teacher-forced answer-token negative log likelihood,
not generated-answer accuracy.** Its existence is not evidence that it works
well on a pretrained pair. Run its tiny optional gradient smoke test first,
then use independently curated tasks and measure generated results before any
performance claim.

## Research question and prior work

The [research protocol](docs/research_protocol.md) states the hypotheses,
controls, limitations, and next gates. Learned communication is a long-standing
research area; primary references include:

- Foerster et al., [DIAL/RIAL](https://arxiv.org/abs/1605.06676) (2016).
- Lowe et al., [On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168) (2019).
- [LatentMAS](https://arxiv.org/abs/2511.20639).
- [Interlat](https://arxiv.org/abs/2511.09149).

This repository is an independent initial implementation for controlled
experiments, not a claim to have invented latent communication. Its companion
project, [Research2](https://github.com/ConfusedDuckCollectibles/Research2), tests
structured communication and shared task state through ordinary agent interfaces.
