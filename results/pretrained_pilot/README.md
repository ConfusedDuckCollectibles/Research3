# Executed pretrained-model pilot: no demonstrated communication advantage

**The informative bridge did not outperform the equally trained control that
received no private facts.** This run does not establish useful latent
communication, a bandwidth advantage, or a breakthrough. It does establish a
reproducible experiment with real pretrained models and stronger controls than
the initial synthetic-vector experiment.

- [Executed workflow](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45322539575)
- [Code used for this run](https://github.com/ConfusedDuckCollectibles/Research3/tree/5ea60e25e771275ddf3fac9ae3f6e6d5aa70bfc2)
- [Design specified before examining outcomes](../../docs/pilot_design.md)
- [Full measurements](results.json), [individual predictions](predictions.jsonl),
  [data manifest](dataset_manifest.json), [model hashes and revisions](model_manifest.json),
  [execution details](execution.json)

The sender was frozen DistilGPT2 and the receiver frozen Pythia-14M. Their model
families, tokenizers, and hidden dimensions differ. Only the small bridge was
trained, separately for informative and constant-input conditions. The control
received one fixed seeded random tensor across every example, with no private
facts; it retained attention-query gradients and distinct slot capacity.

## Primary outcomes

Each evaluation split contains 64 questions from 16 independent worlds. The
generalization split uses unseen worlds with different source wording, fact
order, and question phrasing. Training used 256 questions from 64 worlds, three
epochs, one seed, and validation-only checkpoint selection. These are small
synthetic lookup tasks, not production agent work.

| Condition | Test exact match | Test answer NLL | Shift exact match | Shift answer NLL |
|---|---:|---:|---:|---:|
| Correct learned message | 6/64 (9.375%) | 1.1988 | 0/64 | 4.3861 |
| Trained constant input | 7/64 (10.9375%) | 1.1444 | 0/64 | 2.8749 |
| Zeroed learned message | 0/64 | 4.9207 | 0/64 | 5.3060 |
| Message from another world | 5/64 (7.8125%) | 1.2103 | 0/64 | 4.3856 |
| No channel | 0/64 | 5.2218 | 0/64 | 6.1420 |
| Complete text facts | 0/64 | 4.0166 | 0/64 | 6.2755 |
| Compact symbolic facts | 0/64 | 3.9170 | 0/64 | 5.1168 |

Exact match requires the whole generated answer after whitespace/case
normalization. NLL is a separate teacher-forced probability metric; lower is
better. Generation does not receive the expected answer or its length.

The informative bridge beats the frozen no-channel receiver, but the trained
constant-input bridge also does so and performs slightly better than the
informative bridge. Replacing the correct source world changes only one
additional exact-match answer relative to the matched condition. This is no
convincing evidence of useful world-specific transfer. One seed and 16 worlds
do not support a significance or general superiority claim.

## Posthoc diagnosis: output formatting confounds text comparisons

The full-text receiver frequently generated a color followed by more text,
for example ` red. The light`. Those responses fail the prespecified strict
metric. The adapted latent bridges learned to produce a color followed by a
newline. Therefore the strict metric alone creates a misleading appearance
that latent communication beats text.

An explicitly **posthoc descriptive diagnostic** extracts a permitted color
only at the start of the answer, with a word boundary, using the same rule for
every condition. It gives:

| Condition | Test leading-color correctness |
|---|---:|
| Complete text facts | 32/64 (50%) |
| Trained constant input | 7/64 (10.9375%) |
| Correct learned message | 6/64 (9.375%) |
| Message from another world | 5/64 (7.8125%) |
| Other conditions | 0/64 |

All conditions remain at zero on the generalization split under this diagnostic.
The original primary scores remain unchanged. This diagnosis was chosen after
looking at outputs and cannot be treated as a prespecified positive finding.
See [diagnostic data](formatting_diagnostic.json) and reproduce it with:

```bash
python scripts/analyze_pilot_predictions.py
```

The informative bridge predicted green for 48 ordinary-test questions, blue
for 15, and black for one. The permuted condition had the same marginal answer
counts. The constant-input control predicted yellow for 32 and black for 32.
The matched and permuted answers were identical on 62/64 ordinary-test queries
and 64/64 shifted queries. These patterns are consistent with learning
answer-format and label preferences without learning a reliable private-fact
lookup channel. Because the shifted split changes sender and query wording
together, that failure cannot identify which change caused the problem.

## Costs and limits

The latent payload is **64 raw float32 bytes**, before any transport framing.
The ordinary-test complete text averaged 74.75 UTF-8 bytes, while the compact
symbolic facts averaged **42.75 bytes**. Thus even a successful 64-byte latent
message would not beat this simple symbolic serialization on payload size.

The learned bridge checkpoint occupied 44,025 bytes, in addition to both base
models and the cached sender states. The receiver consumed two latent prefix
positions versus a mean of 21 text-prefix tokens. Fewer positions alone do not
prove faster or cheaper end-to-end operation.

Measured runner time was 28.91 seconds, with 34.76 seconds for the wrapper
including model downloads. These numbers exclude workflow provisioning and
dependency installation. They cover this small CPU experiment, with deliberately
uncached reference decoding, and are not production speed comparisons. Both
base models, data hashes, phases, sizes and dependency versions are recorded.

The workflow ran all **15 tests successfully**, including bridge gradient,
constant-control capacity, and prompt-only generation checks, before training.
The report files recovered from its logs were SHA-256 verified against the
original emitted bytes. Raw predictions contain all 896 condition/example
records.

## Next experiment and gates

1. Establish a receiver that reliably solves the task from complete text and
   compact facts. Prespecify a shared answer format or a closed-answer scoring
   rule. A communication experiment is hard to interpret when the receiving
   model cannot perform the task even with all facts.
2. Give text, symbolic, latent and constant-information conditions comparable
   adaptation budgets. Keep the current negative result unchanged and use a
   newly generated held-out split for any revised experiment.
3. Create paired worlds in which only the queried object's color changes;
   hold the query and all other facts fixed. Test whether predictions follow
   the changed fact. Expand to at least three seeds and report uncertainty
   over worlds, not individual correlated questions. Test sender-only and
   query-only wording shifts separately.
4. If private-information transfer is demonstrated, measure the task-quality
   versus total-cost frontier, including encoding, decoding, setup, training
   amortization, retries and simple cache/delta baselines. Do not optimize a
   latent representation merely to beat an unnecessarily verbose baseline.

The first workflow launch used a degenerate all-zero training control. Review
identified and corrected that design issue before any outcomes were inspected;
the earlier run is superseded and remains in workflow history. This report
uses only the corrected fixed-random-input design.
