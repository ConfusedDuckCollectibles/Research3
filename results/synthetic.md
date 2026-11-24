# Executed synthetic experiment

This is a controlled learned-channel experiment, **not an LLM benchmark or a scientific breakthrough**.

| Latent width | Matched query accuracy | Zero message | Permuted message | Payload bytes | Framed bytes |
|---|---:|---:|---:|---:|---:|
| 4 | 90.13% | 12.83% | 12.00% | 16 | 24 |
| 8 | 99.88% | 12.40% | 11.89% | 32 | 40 |
| 16 | 99.90% | 12.49% | 11.87% | 64 | 72 |

Accuracies are means over the recorded bridge seeds. Each seed uses the same held-out world split. The receiver chooses among eight possible answers; uniform chance is 12.5%.

The train-only least-squares decoder using the full sender state scores 100.00%. This strong control exposes how favorable the synthetic representation is.

The lossless packed-symbol baseline scores 100% and uses **2 payload bytes / 10 framed bytes**. Four uint8 symbols also score 100% with 4 payload bytes / 12 framed bytes. These codecs have explicit access to the synthetic attributes, as does the sender before its frozen observation transform.

**Finding:** this pipeline can be tested for causal use of a learned continuous channel. These results do not establish better bandwidth than symbols, better inference than English, interoperability between pretrained models, or novelty.

## Reproduction and accounting

The adjacent JSON contains every run, configuration, split hashes, validation checkpoint selection, training times, bridge checkpoint size, frozen-agent setup size, raw training data size, and feature-cache size. Timing is environment-specific. Optimizer storage is an estimate for the arrays named; it excludes transient activations, process overhead, total I/O, and energy.

All interpretation limits:

- Held-out worlds are unseen combinations of familiar values; this is not a domain-shift test.
- Frozen coordinates are deliberately related to known attribute semantics. This is a favorable bridge smoke test, not a discovery of a universal representation.
- The symbolic baselines are exact codecs, not language models; fixed-grammar English bytes are counted but its task accuracy is not evaluated.
- Float latents use more payload bytes than the lossless packed-symbol baseline. No bandwidth, token, or end-to-end speed advantage is claimed.
- Multiple seeds vary bridge initialization and minibatch order, not the fixed train/test split. Standard deviations are not confidence intervals over independent tasks.
- Zero and permuted messages intervene on the receiver channel. Test labels and sender states are not otherwise accessible to the receiver.
- No pretrained model has been trained or evaluated in this experiment.
