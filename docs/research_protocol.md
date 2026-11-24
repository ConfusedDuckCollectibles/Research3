# Research protocol

## Status and objective

Initial milestone: a reproducible learned channel, explicit receiver
interventions, exact byte accounting, and a path to accessible transformer
models. Both the NumPy experiment and a pinned DistilGPT2-to-Pythia-14M pilot
have now executed. The [pretrained pilot report](../results/pretrained_pilot/README.md)
records a negative result: the informative bridge did not outperform an
equally trained control with no private facts. No novel scientific result is
claimed. Its posthoc formatting diagnosis is separate from the original
primary outcome.

A subsequent [three-seed paired-fact study](../results/counterfactual_study/README.md)
uses fixed-choice scoring and equal task adaptation. It detects a small
correct-versus-shuffled message effect, but the latent bridge remains far
below the adapted full-text baseline and reacts poorly to irrelevant edits.
Its design, primary contrast, family-cluster uncertainty and all outcomes are
preserved separately. This does not change the first pilot's negative result.

The subsequent [public-query-conditioning follow-up](../results/query_conditioned_study/README.md)
also misses its predeclared practical targets. Query-aware paired accuracy is
7.8125%, versus 5.7292% for the matched query-blind arm. Their primary
difference is +2.0833 percentage points, with a descriptive fixed-seed
family-bootstrap interval of [-0.5208, +4.9479] points. All adapted-text
validation gates pass at 100%, but latent nuisance stability is only 61.9792%.
Wrong public questions leave paired accuracy unchanged in every seed. The
results do not support advancing to efficiency claims or declaring successful
query-specific selection.

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
a separate bridge on one fixed seeded standard-normal sequence of length eight, with the same
architecture, initial weights, training order, examples, update count, and
optimizer. This exact tensor is reused for every example, independent of all
facts/labels. Unlike all-zero positions, distinct nonzero positions preserve
attention-query gradients and distinct slot outputs. It gets its own
validation-selected checkpoint. PLatour/PACT-Check
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

## Completed public-query-conditioning follow-up

The [design](query_conditioned_design.md) was published before the new results.
It tests one hypothesis suggested by the previous failure: a public question
may help sender pooling retain the relevant private fact. The source and
public question are encoded separately by frozen DistilGPT2. A normalized
mean question state enters each attention query through one learned scalar
gate initialized to zero. Pythia-14M remains frozen; only the bridge adapts.

Five arms share architecture, initialization, training order and budget:
query-aware, query-blind, private-constant with the true public question,
adapted complete text and adapted complete symbolic facts. The fixed-question
and fixed-private-state controls use nonzero random inputs so the controls
retain adaptation capacity. The query vectors use nonparametric
zero-mean/unit-variance normalization, epsilon 1e-5, rather than unit-L2
normalization. Wrong-source, wrong-public-question, zero-message and
frozen-no-channel interventions are evaluated separately.

Data seed 20261125 reserves the union of 960 prior counterfactual worlds and
112 first-pilot worlds, totaling 1,042 unique exclusions. The new 128/32/32
train/validation/test families reserve 960 distinct base/variant worlds.
Four views reuse the same 32 test families. Eight epochs, three seeds, batch
size 16, two slots of eight float32 coordinates and all other settings remain
as preregistered. No test-driven architecture or threshold changes were made.

The primary comparison is original-view query-aware minus query-blind paired
both-correct accuracy. It is small, varies across seeds and has an interval
including zero. Correct sources outperform wrong-family sources by a
secondary 5.9896 percentage points, but query-aware absolute accuracy is poor
and below the 23.4375% symbolic and 100% text paired scores. Targeted edits
change the latent prediction 35.6771% of the time; irrelevant edits change it
38.0208%. The channel is not selective enough to meet the practical targets.

A descriptive post-run row comparison finds that wrong public questions
change 2/1,152 original-view predictions and 7/4,608 across all four correlated
views. Only one of the 768 original-view target records changes its predicted
choice, and both choices are wrong. Nonzero learned gates alone do not
establish useful question-specific binding. These diagnostics add no new
training, model selection or held-out runs.

The unchanged advancement targets require at least 80% paired accuracy,
90% nuisance stability, and every seed's adapted-text validation accuracy at
least 80%. The competence gates passed; the two latent-performance targets
failed. A future bounded diagnostic could freeze the learned messages and
test whether a small controlled readout recovers the relevant fact, to help
locate the loss in sender compression versus receiver use. This proposed next
experiment would freeze the selected query-aware, query-blind and
private-constant message generators, then train identical readout probes
with the same public-question input, initialization rules, examples, update
budget and validation selection. It would retain wrong-source and nuisance
controls and evaluate paired correctness on newly reserved worlds. The
original frozen receiver and the probes must be compared on those same worlds.

Reliable probe recovery with poor original-receiver performance would show
that the message contains information accessible to that readout, directing
attention to receiver integration. Probe failure would only establish that
the prespecified readout failed to recover the fact; it would not prove the
message contains no information. A result that fails wrong-source or nuisance
controls would not establish useful selective transfer. This is a diagnostic,
not a new communication mechanism or a novelty claim. It needs a separate
fixed design, verified fresh data and retained negative outcomes; it has not
been executed here.

The 64-byte latent reply additionally requires a public-question request,
averaging 24 UTF-8 bytes on the original templates. The conservative total is
88 payload bytes before framing. Question-specific messages cannot be reused
across different questions. Cached question encoding calls, time, vector
storage and logical uncached work are recorded separately. Equal update
budgets do not imply equal FLOPs, and this task's two-byte packed facts remain
a strong communication-size bound.

The previous 3.65% and current 7.81% paired scores are not a controlled
cross-study improvement: the data and bridge architecture differ. Causal
comparisons must use the matched arms within each study. All raw reports and
predictions from both studies are preserved without retrospective changes.

Before claims about efficient LLM collaboration:

- Meet the fixed competence, paired-accuracy and nuisance-stability gates
  under a controlled mechanism comparison. The completed multi-seed studies
  already include equal-budget text adaptation and private-fact edits, but
  their latent results fail the practical gates. Retain exact model revisions,
  data manifests, source revisions and package versions for each next study.
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
