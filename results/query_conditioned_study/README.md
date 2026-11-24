# Query conditioning: practical targets missed, weak question-specific behavior

The follow-up did **not** demonstrate a reliable benefit from giving the
sender the public question. Query-aware paired accuracy was **7.8125%**, versus
**5.7292%** for the matched query-blind arm. The primary difference was
**+2.0833 percentage points**, with a descriptive family-bootstrap interval of
**[-0.5208, +4.9479] percentage points**. The interval includes zero and one of
the three seed-level differences is negative. Adapted full text reached 100%.

Both predeclared engineering targets failed: paired accuracy was far below
80%, and nuisance stability was **61.9792%**, below 90%. There remains a small
source-specific signal, but changing the sender's public-question input
barely changes its answers. These results do not establish useful selective
communication or an efficiency advantage.

## Provenance and retained records

- [Executed workflow, run 45329986439](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45329986439)
- [Execution job 138228616914](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45329986439/job/138228616914)
- [Exact executed source, fb793ba31d7a10975a9e5c6032b44ecb5eb5774f](https://github.com/ConfusedDuckCollectibles/Research3/tree/fb793ba31d7a10975a9e5c6032b44ecb5eb5774f)
- [Design published before outcomes, commit 5440d5b3472c1e95565f24a7fab1b8a7b3319f01](https://github.com/ConfusedDuckCollectibles/Research3/blob/5440d5b3472c1e95565f24a7fab1b8a7b3319f01/docs/query_conditioned_design.md)
- [Complete measurements](results.json), [original generated report](report.md),
  [compressed individual predictions](predictions.jsonl.gz),
  [dataset manifest](dataset_manifest.json), [model manifest](model_manifest.json),
  and [execution details](execution.json).

The generated `report.md` is preserved exactly as recovered. Its opening
difference displayed as `2.08%` means **2.08 percentage points**, not a relative
percentage improvement; its accompanying interval is stored in probability
units. This explanatory README supplies the explicit units without modifying
the original report or measurements.

The frozen sender was `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`; the frozen receiver was
`EleutherAI/pythia-14m` at revision
`cf967c0a9a04383db6f7b1108d86b2962634b4ac`. The models used different tokenizers
and hidden dimensions. This was one model pair and a controlled synthetic
lookup task, not general agent work.

## What the experiment isolated

The sender separately encoded its complete private source and the receiver's
public question. The mean question state received nonparametric
zero-mean/unit-variance normalization with epsilon 1e-5. Each attention slot
used a learned base query plus a learned scalar gate times that public vector.
The gates started at zero.

Five trained arms had identical bridge architecture, initial weights, example
order, optimizer and update count within a seed. The query-blind arm received
the actual source but one fixed normalized question vector. The
private-constant control received fixed nonzero private states and the actual
public question. Text and symbolic controls received the same adaptation
architecture plus complete facts at the receiver. Every arm trained for eight
epochs, or 256 updates, under seeds 101, 202 and 303; each selected its own
lowest-validation-loss checkpoint.

All three adapted-text validation accuracies were **100%**, passing the 80%
competence gate in every seed. Failure of the latent channel on the original
templates therefore cannot be attributed to an inability of this adapted
receiver to solve the task when given complete text.

Data seed 20261125 excluded all **1,042** distinct worlds from the two prior
pretrained studies before reserving **960 new base/variant worlds**. The new
dataset contained 128 training, 32 validation and 32 test families. The four
test views reused those same 32 families. Source, target and nuisance members
never crossed family or partition boundaries.

## Primary original-template result

The primary metric requires both the original and the single-target-fact-edit
answers to be correct. Individual record accuracy covers those two members;
nuisance members are reported separately.

| Condition | Both target answers correct | Individual target-record accuracy | Nuisance prediction stability |
|---|---:|---:|---:|
| Query-aware message | **7.8125%** | 27.8646% | 61.9792% |
| Query-blind message | **5.7292%** | 28.6458% | 63.5417% |
| Query-aware, wrong private source | 1.8229% | 13.5417% | 61.9792% |
| Query-aware, wrong public question | 7.8125% | 27.8646% | 62.2396% |
| Trained private-constant input | 0% | 12.2396% | 100% |
| Zero compressed message | 0% | 11.7188% | 100% |
| Frozen receiver, question only | 0% | 12.5000% | 100% |
| Adapted complete symbolic facts | 23.4375% | 39.8438% | 75.0000% |
| Adapted complete text | **100%** | **100%** | **100%** |

Across seeds the query-aware arm correctly answered **30/384** pair
evaluations, versus **22/384** for query-blind. These are repeated evaluations
of only **32 independent test families**. The prespecified bootstrap averages
each family's four query scores over the three fixed seeds, then resamples
32 families 2,000 times with seed 66173. The interval is conditional on these
seeds; it does not establish population-level model or task uncertainty.

| Training seed | Query-aware correct pairs | Query-blind correct pairs | Aware minus blind, percentage points |
|---:|---:|---:|---:|
| 101 | 2/128 (1.5625%) | 3/128 (2.3438%) | -0.78125 |
| 202 | 16/128 (12.5000%) | 10/128 (7.8125%) | +4.68750 |
| 303 | 12/128 (9.3750%) | 9/128 (7.0313%) | +2.34375 |

The query-aware arm did not improve individual target-record accuracy over
query-blind. Constant/no-channel inputs have structurally zero paired
correctness because their input does not change when the required answer
changes. Their perfect nuisance stability also does not imply competence.
Neither comparison should be used alone to claim useful communication.

## What the interventions show

Wrong-family private sources reduced query-aware paired accuracy from
7.8125% to 1.8229%. This secondary difference is **+5.9896 percentage points**,
with a descriptive paired-family interval of **[+3.6458, +8.5938] points**.
Source information matters under this setup, but absolute performance remains
poor. This is secondary evidence, not a substitute for the primary
query-aware versus query-blind comparison.

The wrong-public-question intervention left paired correctness unchanged in
**every seed**. A post-run descriptive comparison of individual predictions
shows that the intervention was not literally inert, but its behavioral
effect was very small:

| Paired rendering view | Changed predicted choices under wrong public question | Fraction changed |
|---|---:|---:|
| Original templates | 2/1,152 | 0.1736% |
| Sender wording/order shift | 0/1,152 | 0% |
| Question wording shift | 4/1,152 | 0.3472% |
| Both shifts | 1/1,152 | 0.0868% |
| All correlated views, descriptive total | 7/4,608 | 0.1519% |

These counts include original, target and nuisance members and all three
seeds. Among the 768 original-template target records, only **one** prediction
changed, and both its original and substituted-question choices were wrong.
The aggregate paired accuracy consequently stayed the same. The original-view
mean absolute change in gold-class restricted cross-entropy was about
0.00119 nats; this small continuous change does not establish successful
object-specific selection.

The selected query-aware gates were nonzero:

| Seed | Selected epoch | Slot 1 gate | Slot 2 gate |
|---:|---:|---:|---:|
| 101 | 5 | -0.020849 | -0.024433 |
| 202 | 8 | -0.031673 | -0.019755 |
| 303 | 8 | -0.024102 | -0.022951 |

These are descriptive parameter values. Nonzero gates prove that optimization
updated the query path; they do not prove it learned the intended binding or
explain the primary difference. The very small intervention effect provides
little behavioral support for successful question-specific selection in this
run. It does not prove the path contains no information.

Targeted edits changed the query-aware prediction **35.6771%** of the time,
while irrelevant edits changed it **38.0208%** of the time. Its low nuisance
stability therefore accompanies poor selectivity, rather than a reliable
response to only the relevant fact.

## Wording shifts

| View | Query-aware paired accuracy | Query-blind paired accuracy | Adapted-text paired accuracy |
|---|---:|---:|---:|
| Original templates | 7.8125% | 5.7292% | 100% |
| Sender wording/order shift | 0% | 1.0417% | 100% |
| Question wording shift | 0.7813% | 0.5208% | 0% |
| Both shifts | 0% | 0.5208% | 0% |

The source-only shift reveals a clear failure of the latent pipeline while
adapted text remains competent. Question wording shifts also break the text
interface, so those outcomes cannot be assigned exclusively to latent
communication. High nuisance stability in a shifted view can merely reflect
repeated wrong answers. These views are secondary analyses of the same worlds.

## Costs and audit

The latent reply contains **64 float32 payload bytes**. Original-template
questions average **24 UTF-8 bytes**, making the conservatively charged
request/reply payload **88 bytes before framing**. Complete private text
averages 75.10 bytes, and compact symbolic facts 43.10 bytes; those are their
private fact payloads, not measurements of a network stack. Query-aware
messages cannot be reused across different questions. A deterministic
two-byte packed representation can convey all four categorical facts in this
task, so this experiment does not establish a bandwidth advantage.

The run encoded eight distinct public questions, totaling 60 sender tokens,
and cached their vectors in 24,576 bytes. Actual query encoding and
normalization took approximately **0.175 seconds**. The whole runner took
**159.57 seconds**, or **166.03 seconds** including pinned downloads and wrapper
preparation; provisioning and dependency installation are excluded. These
are experiment durations, not production latency or communication speedups.
Logical uncached calls/token work, model storage, source/receiver caches,
training budgets, input lengths and phase timings remain in `results.json`.

An independent post-run reconstruction checked all **41,472** unique
condition/member predictions and reproduced every reported condition mean
and family-bootstrap interval, including the primary contrast. All **4,608**
wrong-source choices matched the declared coherent source-family mapping.
Regenerating the data reproduced every partition hash, the prior-world
exclusion set and the new reserved-world identities. Model/design provenance
and recovered raw/gzip/report hashes were also verified.

Key integrity hashes:

- Raw prediction JSONL: `324d8184f7730507da651eea0c49e16b646563d0746bc4d54f5747dfa798757c`.
- Compressed predictions: `b57c09e0b3afa8e12b357ea74055047f8414213421df729c9eaac54198bf5a8c`.
- Frozen design: `25701024eaf35d112e2f4eae80f094370154abc041018e2e973233f9438f33c0`.
- Prior-world exclusion identities: `4ef5acf31e0db8cc8fe507b7dcbbc16fcd93623f6072c66379c8323236eb366c`.

The earlier study's 3.65% and this study's 7.81% are **not a controlled causal
comparison**: the data and architecture differ. The relevant comparison is
the matched query-aware versus query-blind arms within this study. The
practical targets remain unchanged and failed. A useful subsequent diagnostic
would distinguish information lost during sender compression from information
the frozen receiver fails to use, under a separate fixed design. No new model
run or architecture change was performed as part of this result audit.
