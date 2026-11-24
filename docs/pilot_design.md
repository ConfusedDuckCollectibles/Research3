# Fixed pretrained-model pilot

This design is committed before inspecting this pilot's outcomes. It upgrades
the experiment from hand-designed vectors to two real pretrained language
models. It is a small, exploratory test, not evidence of a breakthrough.

## Question

Can a learned, short continuous message convey private object/color facts from
a frozen DistilGPT2 sender to a frozen Pythia receiver, on previously unseen
fact assignments, beyond what equal-budget adaptation without information can
achieve?

The receiver cannot answer the random assignment from its query alone. A sender
sees four facts, and the receiver is asked for one object's color. This makes
useful private information measurable, though the task remains synthetic and
very simple.

## Models and fixed budget

| Role | Public model | Exact source revision |
|---|---|---|
| Sender | `distilbert/distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` |
| Receiver | `EleutherAI/pythia-14m` | `cf967c0a9a04383db6f7b1108d86b2962634b4ac` |

The model families, hidden dimensions, and tokenizers differ. The bridge is
trained specifically for this pair; this is not zero-shot transfer to a new
receiver. Only public safetensors weights and tokenizer/configuration files
are downloaded. Model loading uses local files and disables remote code.

- Data seed: 20261123. Training seed: 101. No test-set tuning.
- 64 training worlds, 16 validation worlds, 16 ordinary test worlds, and
  16 generalization worlds; four questions per world.
- Each world assigns one of eight familiar colors to each of four objects.
  Entire worlds are disjoint across splits, including their paraphrases.
- The generalization split changes source order, source phrasing, and the
  question template, while retaining the objects and colors.
- Two latent slots with eight float32 coordinates each: 64 raw payload bytes.
- Three epochs; 768 optimizer steps for each trained bridge; AdamW at 0.001.
- Constant and informative bridges start from identical parameters and use
  the same example order, labels, optimizer, and update count.
- Separate checkpoints selected only by each bridge's validation answer NLL.
- CPU, two threads, maximum sequence length 128, maximum four generated tokens.
- GitHub workflow timeout: 20 minutes. Standard public-repository runner.

Small pretrained models and this short training budget may fail. A failed
pilot is retained, with its limitations, rather than relabeled as success.

## Conditions and interpretation

| Condition | Receiver receives | What it checks |
|---|---|---|
| `matched` | Trained message from the correct sender observation | Candidate information channel |
| `trained_constant` | Independently trained bridge fed the same fixed seeded random tensor for every record | Whether adaptation alone explains gains |
| `zero` | Informative bridge's message replaced with zero before decoding | Dependence on learned message values |
| `permuted` | Message from a different private world | Dependence on the correct source information |
| `no_channel` | Only the query | Frozen receiver and answer-prior baseline |
| `text` | Complete source facts in ordinary text | Direct readable communication baseline |
| `symbolic` | Same facts as compact key/value text | Strong simple representation baseline |

The fixed input always has shape `1 x 8 x sender_dim`. Its seeded random
values are independent of all worlds and labels and remain identical across
records. It cannot reveal private facts through input values or length.
Nonzero, varied positions let the attention queries learn and allow prefix
slots to differ. A cyclic permutation uses a different
sender observation for every evaluated record. A different world can still
share an individual queried color, so permutation is not an instruction that
every answer must be wrong.

### Design correction before reviewing outcomes

The first workflow launch used an all-zero constant tensor. Independent review
identified that zeros also force identical pooled slots and zero gradients for
the attention queries, reducing the control's effective adaptation capacity.
This was corrected to a fixed seeded nonzero tensor without examining any model
outcomes or changing data, training budget, seeds, or scoring. The first launch
is superseded; only the corrected design is the primary pilot. Both revisions
remain in commit and workflow history.

Text and symbolic baselines use the frozen receiver without separate task
adaptation. Their comparisons describe this pilot only; superior adapted
latent performance would not by itself prove superior encoding. A larger
follow-up must include equally adapted text and symbolic baselines.

## Outcomes specified before the run

Primary: greedy generated-answer normalized exact match on ordinary test and
generalization splits. Normalize whitespace and case only; punctuation stays
significant. Generation sees the query and condition-specific prefix, never
the expected answer or its length. Stop at the first newline/EOS or the fixed
four-token limit.

Secondary: answer-token negative log likelihood (NLL), with prompt and prefix
labels masked. Lower NLL can indicate better probabilities even if the model
still fails to generate the right complete answer. It must not be reported as
task success.

Evidence of useful communication would require the informative message to
outperform the trained constant control and suffer a loss when source worlds
are permuted. A convincing improvement needs replication across seeds and
world-level uncertainty estimates, plus stronger adapted baselines. This
single pilot cannot establish those claims.

Report all seven conditions and both splits, even if every accuracy is zero.
Publish predictions, data-generation manifest, model revisions/file hashes,
training history, selected epochs, setup sizes, and phase timings. Do not
equate latent float bytes with tokenizer tokens or claim production speed,
API cost, energy savings, or a universal cross-model language.

The sender encoding cache, full model parameters, bridge parameters and
checkpoint files are reported separately from per-message payload. The
generation code is an intentionally simple uncached reference implementation;
its timings are not an optimized end-to-end efficiency comparison.

## Why these controls matter

- [PLatour / PACT-Check](https://arxiv.org/abs/2611.06976) reports that some
  apparent transfer benefits can be reproduced with uninformative sender
  inputs or receiver-side adaptation. This motivates the trained constant
  control rather than relying only on zeroing an already trained message.
- [When Does Latent Communication Pay?](https://arxiv.org/abs/2608.04893)
  motivates controls that test whether the communication causes task gains.
- [LatComm](https://arxiv.org/abs/2610.26840),
  [XKV](https://arxiv.org/abs/2608.20617), and
  [CacheBack](https://arxiv.org/abs/2609.32046) already explore compressed,
  cross-model, or receiver-conditioned communication. Those broad ideas
  cannot be claimed as new contributions of this repository.

## Reproduction

Install the CPU dependencies pinned in
`.github/workflows/pretrained-pilot.yml`, then run:

```bash
python scripts/run_pretrained_pilot.py
```

This command downloads the two specified public model snapshots, generates
the fixed data, and executes the run. Results are written under
`checkpoints/pretrained-pilot/`. Small public reports are also preserved as a
workflow artifact and losslessly encoded in logs; weights and learned tensor
checkpoints are excluded from that artifact. Run the workflow manually to
repeat it. A results/documentation-only commit does not rerun the pilot.
