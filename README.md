# Research3: learned latent communication

A research prototype for passing a small continuous message between a sender
and a receiver. This repository contains **an executed synthetic experiment**
and **executed pretrained-model studies with explicit causal controls**. It does not
claim a new universal language, an LLM efficiency breakthrough, or a replication
of LatentMAS or Interlat.

## Latest: audited coefficient transport results

The [complete coefficient-packet report](docs/coefficient_results.md) preserves a completed, negative engineering result. All **720 actual TCP trials**, **300 numerical diagnostics**, and **nine primary float64 full-vector fidelity gates** passed their correctness checks. Both performance criteria failed: the shaped warm gain over the strong optimized Gram codec was only **1.97%**; unshaped warm time was **20.98% higher** than the retrospective best comparator, and cold first-message time was **6.23×** that comparator. The shared-table setup cost is charged explicitly.

The independent audit found no issues within its scope across **233,297 integrity**, **106,829 report-consistency**, and **17,025 scientific validation** checks. Actual packets, setup/ACK streams, arrays, timings, audit limitations and all original results are permanently retained in [the evidence directory](results/coefficient_packet/). The shaped profiles impose application pacing and reply holdoff on physical loopback; this is not a WAN, language-model-quality or complete agent-speed result. No breakthrough is claimed.

## Earlier alignment result and current mathematical leads

The [complete alignment report](docs/alignment_results.md) found a **1.17–3.02× faster alignment core** than the efficient thin-SVD baseline across twelve fixed CPU fixtures. All 15,759 required checks passed, and the retained-array audit passed. The [original evidence](results/alignment_numerical/) is unchanged. The original Gram packet was smaller but used 1.86–2.29× the arithmetic time of fast dual-direct. A nonunique canonical float32 packet also showed **35.89% relative vector error despite unchanged nearest token IDs**.

The new [rank-deficient feasibility note](docs/rank_deficient_feasibility.md) derives a representation of the sender's chosen legacy output using a reference-space coefficient plus explicit residual coordinates. It compares strong generic low-rank representations and established prior art. It is an analytical note with no new measurement, bitwise-equivalence or novelty claim. The separately reviewed [information-recovery note](docs/statebridge_information_recovery.md) gives an exact conditional inverse for the centered source Gram. It distinguishes broader recoverability from unique-product equivalence and shows how a known reference mean can undo row calibration when the actual pre-prefix has full row rank. This requires known reference side information; upstream StateBridge does not transmit those reference IDs. Finite-precision recovery on authentic model states remains unmeasured. The [new frozen assay](docs/contextual_information_design.md) will test one pinned small homogeneous model on 256 fixed teacher-forced sequences, with exact byte-collision controls and a fixed numerical inverse. Its known public reference tokens, narrow model scope and 600-second execution budget are explicit; no pretrained outcome exists yet. The [implementation and evidence contract](docs/contextual_execution_notes.md) preserves the untouched author method, a public-input-only inverse, independent offline tests and immutable per-record extraction checkpoints. The source-only commit passed all **200 offline tests** (zero skips/failures), and the separately reviewed one-run workflow is now activated. The scientific outcome remains pending.

The separate [lossless-predictor feasibility assessment](docs/lossless_predictor_feasibility.md) records why another model study on vocabulary-byte XOR compression is not justified by the current artifacts.

## Previous binding-sensitive training

The [complete audited binding study](docs/binding_results.md) found **0% target-pair correctness** for both the matched CE-only and auxiliary-trained latent bridges, versus **99.54%** for adapted full text. All 18 fits and 86,016 fresh records completed. An independent auditor reconstructed the dataset and reported statistics and passed 1,056,048 consistency checks; its scope and limits are recorded in the [permanent evidence](results/binding_study/).

The new cohort holds the available words constant while changing which object has which color. High answer stability did not imply successful transfer: the auxiliary arm was 99.61% stable under irrelevant changes, but only 12.43% of those pairs were both correct. Zeroing its message changed 5,632 of 6,144 choices despite zero correctly answered target pairs. The fixed auxiliary-loss hypothesis failed; no practical advance or breakthrough is claimed.

## Previous frozen-message diagnostic

The [readout diagnostic](docs/readout_results.md) reproduced all 41,472 earlier
prediction records exactly, then tested additional readers on 32 fresh families.
A small linear reader of the frozen message reached **9.90% paired correctness**,
versus **7.81%** for the original receiver. The primary gap was **+2.08 percentage
points**, with a descriptive family interval of **[-0.78, +4.95] points**.
The prespecified small MLP reached only **13.54%**. Neither met the practical
accuracy and nuisance-stability targets.

A much larger full-source reader reached 82.03% on the original template but
**0% when the source wording and fact order changed**; adapted full text remained
at 100% under that source shift. The larger reader has more parameters and
positional access, so its advantage cannot prove irreversible information loss
in the compressed message. This is a measured diagnostic, not a breakthrough.
The complete predictions, selected-state records and bounded NPZ exports are
retained in [the result directory](results/readout_study/).

## Previous query study and implemented controls

The preceding [public-query-conditioning study](results/query_conditioned_study/README.md)
misses its predeclared practical targets. Query-aware paired accuracy is
**7.81%**, versus **5.73%** for the matched query-blind arm. The primary
difference is **+2.08 percentage points**, with a descriptive family-bootstrap
interval of **[-0.52, +4.95] points** conditional on three fixed seeds. The
interval includes zero. Adapted full text reaches **100%**, while latent
nuisance stability is only **61.98%** against the 90% target.

Wrong-family private sources reduce paired accuracy to 1.82%, so a small
source-specific signal remains. Giving the sender the wrong public question
leaves paired accuracy unchanged in every seed and changes only 2 of 1,152
original-view predictions. This provides little behavioral evidence that the
new query path learned useful selection. The study excludes all 1,042 worlds
from the previous pretrained experiments; its 7.81% cannot be treated as a
controlled improvement over the earlier study's 3.65% because both data and
architecture changed. The matched arms within each study are the valid
comparisons.

The [preceding fact-change study](results/counterfactual_study/README.md) and
[first negative pilot](results/pretrained_pilot/README.md) remain unchanged.
No executed model study establishes useful general communication or a complete
agent-efficiency advantage. Designs, all seeds and controls, raw predictions, model/data hashes
and interpretation limits are retained with each result.

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
- Paired single-fact and nuisance interventions, eight-choice scoring,
  equally adapted text/symbolic controls, and a fresh-world public-query
  follow-up with a matched query-blind bridge and explicit question-feedback
  accounting. See the [fixed design](docs/query_conditioned_design.md) and
  [execution guide](docs/query_conditioned_study.md).

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
The separate [executed pretrained pilot](results/pretrained_pilot/README.md)
used DistilGPT2 and Pythia-14M. The learned message answered 6/64 test questions
correctly, versus 7/64 for an equally trained bridge receiving no private facts;
all conditions failed strict exact match under the wording/order shift.
This does not demonstrate a useful communication advantage. A posthoc formatting
diagnostic also exposed why strict exact match understated the frozen text
baseline: it often began with the right color and continued with extra words.
The [fixed design](docs/pilot_design.md), raw predictions, all controls, model
hashes, and execution details are preserved with the report.

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
