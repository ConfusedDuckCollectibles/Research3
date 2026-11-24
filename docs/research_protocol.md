# Research protocol

## Status and objective

Initial milestone: a reproducible learned channel, explicit receiver
interventions, exact byte accounting, and a path to accessible transformer
models. The NumPy run is an executed engineering experiment. The local-model
runner is implemented but unexecuted. No novel scientific result is claimed.

The long-term objective is a better tradeoff between task success and total
computation/communication. Merely avoiding English does not establish that
tradeoff.

## Synthetic experiment design

1. Enumerate all 8^4 = 4,096 attribute combinations. A fixed seed of 23 splits
   them into 2,457 training worlds, 819 validation worlds, and 820 test worlds.
   Every complete world occurs once. Attribute values themselves are familiar
   across partitions. This is combinatorial generalization, not new-domain or
   new-vocabulary generalization.
2. The sender sees the four private values. A fixed random tanh transform maps
   their one-hot representation into 48 dimensions. It is not a pretrained LLM.
3. A learned encoder compresses that state to 4, 8, or 16 continuous values.
   The receiver adapter expands it into 40 dimensions. A fixed orthonormal
   readout produces scores for four eight-way attribute queries.
4. Receiver readout coordinates were constructed from the same known attribute
   semantics. This deliberately favorable alignment is a testing assumption,
   not a discovery about unrelated pretrained models.
5. Train the bridge on all four queries per world. Use three bridge seeds and
   180 fixed epochs. Adam learning rate is 0.003, batch size 128. Restore the
   checkpoint with lowest validation cross entropy. The test partition is not
   an argument to the training function.
6. Evaluate every predefined width/seed after training. No winner is selected
   on test results. Report all runs, including poor initializations.
7. Serialize every test message as little-endian float32 and deserialize it
   before scoring. The receiver sees only that message and the attribute query;
   no world ID, label, or original sender state is passed into its decoder.

## Controls and interpretation

| Control | Question answered | Limitation |
|---|---|---|
| Zero message | Does a constant channel retain the result? | Zero can be out of the learned message distribution. |
| Deranged message | Does a plausible but wrong world's message destroy the result? | Different worlds can still share some attributes. |
| Train-prior/no-message answer | What can the receiver do without private facts? | The finite test sample can differ slightly from uniform chance. |
| Full-state least-squares decoder | Is the synthetic coordinate alignment already easy? | Has a larger channel and known labeled training examples. |
| Four uint8 symbols | How does a simple exact structured channel perform? | Sender already has the synthetic discrete attributes. |
| Twelve packed bits in two bytes | What is a strong lossless symbolic comparison? | Requires the small agreed schema. |
| Fixed-grammar English bytes | What is the UTF-8 payload for a concise description? | No LLM decoder is run, so no language inference claim follows. |

Message/label correlation alone is inadequate. Lowe et al. demonstrate why
apparent communication need not causally affect another agent, motivating
the explicit channel interventions here.

## Accounting

Payload bytes are separate from the common eight-byte frame. The float channel
uses 4k payload bytes; packed symbols use two. A float latent is not a language
model token. Neither these bytes nor vector count can be used as billed token
counts for a commercial API.

The result records bridge parameter count and float32 checkpoint size, frozen
agent weight storage, raw training/validation/test data storage, training
feature cache, float64 parameter plus Adam-moment storage, and training elapsed
time. Frozen weights, schema agreement, and model/adapter distribution are
setup costs. Costs cannot be amortized away without stating the expected number
of future messages. Storage estimates do not cover every transient activation,
filesystem format header, runtime allocation, total I/O, or energy. No physical
network traffic was measured.

The packed symbolic baseline has lower per-message payload and no learned
bridge, so these results cannot establish a communication-byte break-even in
the latent channel's favor.

## Local transformer milestone

The optional runner freezes two user-provided local base models. A learned
attention pool compresses the sender's sequence into `slots × bottleneck`
floats; a receiver-side linear expansion maps those floats into continuous
prefix embeddings. The receiver is in eval mode but its operations remain in
the autograd graph, allowing answer loss to train the bridge. Sender extraction
is detached and cached. Labels for prefix and query are -100; only the answer
is supervised, with the causal model performing its own label shift.

Validation loss selects a checkpoint. The upgraded runner additionally trains
a separate bridge on a fixed all-zero sequence of length eight, with the same
architecture, initial weights, training order, examples, update count, and
optimizer. It gets its own validation-selected checkpoint. PLatour/PACT-Check
shows why a trainable bridge may improve a task even with uninformative sender
signals, so an evaluation-time zero ablation alone is inadequate.

The held-out partitions are evaluated under matched, trained-constant, zero,
different-observation, no-channel, full-text, and compact-symbolic conditions.
Both NLL and generated exact match are reported. Generation receives the query
only and uses a global newline/EOS/token-budget stopping rule, never target
length. Whitespace/case normalization is permitted; punctuation or extra words
remain errors. NLL and exact match answer different questions.

The private-lookup generator assigns complete worlds to train, validation,
test, and generalization. The last partition uses unseen worlds with changed
fact order and source/query wording. Shared `world_id` values are rejected
across partitions, including paraphrases. Four questions share one world and
are correlated. This experiment is still synthetic and model-pair-specific;
it is not a reproduction of the cited systems or a novel method claim.

Before claims about efficient LLM collaboration:

- Run the optional gradient test and then train/evaluate on accessible model
  pairs, retaining exact model revisions, data manifests, and package versions.
- Extend generated-answer correctness to executable agent tasks, multiple seeds,
  and held-out task families. Review for leakage beyond exact strings/world IDs.
- Compare strong concise-text and structured protocols with equivalent facts,
  task budgets, receiver adaptation, and validation tuning budgets.
- Include all sender inference, receiver prefill/generation, adapter compute,
  training amortization, retries, memory, wire bytes, and total wall-clock time.
- Test a model pair and task family absent from adapter training. Compatibility
  between one trained pair is not evidence of a universal representation.
- Investigate adaptive message width with a validation-calibrated accuracy
  target. Treat that as a research hypothesis, not a novelty assertion; conduct
  a focused prior-art review before presenting it as a contribution.

## Primary sources

1. Foerster et al., *Learning to Communicate with Deep Multi-Agent Reinforcement
   Learning*, 2016: https://arxiv.org/abs/1605.06676. Prior art for learning a
   channel through task gradients.
2. Lowe et al., *On the Pitfalls of Measuring Emergent Communication*, 2019:
   https://arxiv.org/abs/1903.05168. Motivation for causal receiver controls.
3. Zou et al., *Latent Collaboration in Multi-Agent Systems*:
   https://arxiv.org/abs/2511.20639. Related latent multi-agent research.
4. Du et al., *Enabling Agents to Communicate Entirely in Latent Space*:
   https://arxiv.org/abs/2511.09149. Related learned cross-model latent adapters
   and compression. Its results are not results of this repository.
5. Hugging Face GPT-2 model API:
   https://huggingface.co/docs/transformers/model_doc/gpt2. `inputs_embeds`,
   internally shifted causal labels, and ignored label value -100.
6. Hugging Face Auto classes:
   https://huggingface.co/docs/transformers/model_doc/auto. Local loading and
   model selection. Consult the documentation for the installed version.
7. Shen et al., *On the Opportunities and Challenges of Prefill-to-Autoregressive
   Cross-Model Transfer*, 2026: https://arxiv.org/abs/2611.06976. PLatour and
   PACT-Check motivate the independently trained no-information control.
8. Cheng et al., *When Does Latent Communication Pay? A Causal Audit of Relayed
   KV Caches in Multi-Agent LLMs*, 2026: https://arxiv.org/abs/2608.04893.
   Motivation for private-information-required tasks and mismatched messages.
