# Binding-sensitive messages: fixed competence study

## Status, question and limits

This document defines a new experiment before implementation and before any new model outcomes. Its commit, implementation commit, data hashes and model manifests must be recorded before execution. The completed pilot, counterfactual, query-conditioned and readout studies remain unchanged.

The question is whether a fixed, task-supervised message loss improves reliable transfer of object–color bindings through the existing small latent bridge, when both comparison arms receive identical edited training examples. Every world contains all eight objects and all eight colors exactly once. Color presence or an unordered token bag therefore cannot identify the answer.

The primary comparison is a query-aware bridge trained with restricted-choice cross-entropy plus the fixed loss below, versus the identical query-aware bridge trained with the same quartet cross-entropy alone. This tests a particular intervention on a deliberately small system. It does not establish a new communication method, an emergent universal language, open-ended reasoning, compression superiority or a performance advantage over English.

Contrastive communication, invariance pressure, entity grounding and causal message substitutions already have substantial prior art. Relevant primary references include [CACL, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html), [MACC](https://arxiv.org/abs/2006.07200), [object constancy and inductive biases, EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.397/), [binding identities in language models, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9d1b7fc578c0d2d6431fc26d736ecaf3-Abstract-Conference.html), [XBridge](https://arxiv.org/abs/2608.11676) and [Hidden Messages](https://arxiv.org/abs/2610.28905). The bounded contribution sought here is evidence about whether this fixed loss repairs the measured binding failure in this particular bridge. A positive result would motivate stronger replications and comparisons with existing methods, rather than justify a novelty claim.

## Models and interface

Use fresh bridge initialization with the same frozen pretrained base models as the previous study:

- Sender: distilbert/distilgpt2, revision 2290a62682d06624634c1f46a6ad5be0f47f38aa.
- Receiver: EleutherAI/pythia-14m, revision cf967c0a9a04383db6f7b1108d86b2962634b4ac.


Use the previously validated CPU runtime: Python 3.11.18, PyTorch 2.6.0+cpu,
Transformers 4.48.2, NumPy 2.4.6, Tokenizers 0.21.4, Safetensors 0.8.0,
and Hugging Face Hub 0.36.2. Verify these versions before fitting and record
them in the execution manifest.

Download only the pinned public files, use safetensors, disallow remote code, and record every file hash. No historical trained bridge checkpoint is used.

Retain the existing bridge architecture: two attention-pooling slots, a LayerNorm–linear–Tanh compressor with eight coordinates per slot, and a linear receiver-side expansion. The transmitted message is the actual 2 by 8 float32 Tanh output, or 64 payload bytes. No additional classifier, learned query projection or auxiliary projection head is introduced.

Encode the public question alone with the frozen sender. Mean-pool over actual tokens and apply nonparametric LayerNorm-style normalization with population variance and epsilon 1e-5. Each attention query remains its learned base vector plus a learned scalar gate times the normalized public question vector. Both gates start at zero. All arms retain this architecture and parameter count; the query-blind arm supplies its fixed seeded vector instead of the real question.

Use the previous seeded nonzero private-input and fixed-question controls, with seed offsets 9841 and 17291 respectively, and record the actual tensors' hashes. Base models stay in evaluation mode with gradients disabled on their parameters. Gradients must still propagate through the frozen receiver to the bridge. Receiver logits are the eight fixed leading-space color-token logits at the final real query position; no answer token or answer-length signal enters the input. Reject duplicate or non-single-token color candidates and reject sequence overflow rather than truncate. Maximum sequence length is 192, including prefix slots where applicable.

Only the sender's private source text and public question may produce the learned message. Family identifiers, variant names, canonical permutations, fact dictionaries, answers and edit metadata are forbidden model features. Quartet membership is available to the training loss only; this is extra relational supervision, not an input channel and not unsupervised learning.

## Worlds, finite families and splits

Fix the object order as apple, book, cup, lamp, key, plate, shoe, ball. Fix the color order as red, blue, green, yellow, black, white, purple, orange. A world is a bijection from these eight objects to these eight colors, represented canonically as the eight color indices in object order. There are 8! = 40,320 worlds.

For a base world, reserve that world and all 28 worlds obtained by exchanging the colors of one unordered object pair. This is a finite 29-world reservation. It is not closure under arbitrary sequences of swaps, which would connect the entire permutation space.

Use data seed 20261127. Enumerate permutations of indices 0 through 7 in lexicographic order, shuffle once with Python random.Random(20261127), then greedily accept a base only if its complete 29-world reservation is disjoint from every already accepted reservation. Stop after exactly 160 accepted families. Assign the first 64 to training, the next 32 to validation and the last 64 to test. Require exactly 4,640 distinct reserved world identities across all partitions. If this fixed construction cannot produce the requested cohort, abort during data-only preflight; do not try seeds until model results look favorable.

Canonical identity includes the full eight-object mapping and task schema, and ignores sentence order. The old experiments used four-object worlds, so this is a changed task domain rather than an additional draw from their 8^4 universe. Reusing the fixed pretrained models is intentional; reusing trained historical bridges or importing previous labels is prohibited.

Use a separate edit RNG, random.Random(20261129). For each family in accepted order, sample a permutation p of the eight indices by repeated uniform shuffling until p has no fixed point. This gives a uniformly sampled donor derangement: each queried index i has partner p[i], and each donor appears exactly once. Then, for each query index i in ascending order, sample an unordered nuisance pair uniformly from the seven indices other than i. Store the resulting partner and nuisance pair in the data manifest. These edit draws are independent of base-color selection and rendering-order draws and never consult model outputs.

For each accepted base world and queried object index i, construct a quartet:

| Member | World | Correct answer |
|---|---|---|
| Original O | Base permutation | Base color at i |
| Target T | Exchange colors at i and its sampled donor p[i] | Base color at p[i] |
| Nuisance N | Exchange colors at the sampled unordered nuisance pair, excluding i | Base color at i |
| Reorder R | Same base world as O, different sentence order | Base color at i |

Both factual edits exchange exactly two values. The target answer always changes; the nuisance and reorder answers always remain unchanged. Donors and nuisance pairs vary across families and queries, rather than defining one globally fixed intervention. The donor derangement deliberately preserves exact color balance across all eight target questions within a family. All eight queries are included in every partition. The scored target and nuisance worlds are members of the broader 29-world reservation; unscored neighbors are still excluded from all other families.

Each family has eight quartets and 32 model records. Training therefore contains 512 quartets and 2,048 records; validation contains 256 quartets and 1,024 records; test contains 512 quartets and 2,048 records. Queries and variants are dependent observations within a family, not additional independent test units.

## Rendering and shortcut checks

Use a separate order RNG, random.Random(20261128), after base families have been selected. For each family in accepted order, draw a uniform permutation sigma of the eight object indices, then draw a second independent permutation rho, rejecting equality with sigma. Use sigma for O, T and N; use rho for R. The order draw does not consult colors, questions, labels or model results.

Render each fact as "The {object} is {color}." followed by a newline, including the final fact. The public question is "The color of the {object} is". The complete symbolic rendering uses "{object}={color};" followed by a newline, with the same sigma or rho order as its text counterpart. Full-text and symbolic receiver prompts include the complete corresponding rendering, a fixed separator and the public question; the exact separator is fixed in implementation before source publication.

Within a family, any occurrence of the same world and rendering view must have exactly the same source string, regardless of which question or quartet role caused it to appear. The source must not reveal the public query through a query-dependent sentence order. O, T and N share order so that their paired comparison changes bindings only. R changes order only.

Before training, verify actual tokenizer-ID multisets, not merely whitespace words: every rendered private text has the same token multiset across all worlds and orders under both pinned tokenizers. Perform the analogous check for symbolic renderings. Also verify complete text/symbolic prompts have the same multiset across a quartet conditional on the fixed public question. A tokenizer-boundary mismatch is a preflight failure, not permission to silently relax this requirement. Any rendering correction must be reviewed and committed before training.

Record canonical permutations, reservations, source/order identities, all partition and manifest hashes, and answer frequencies per query and member. All eight colors occur once across the eight queries of each family/member, but individual object priors may still differ in a finite sample. Preserve the information-free controls rather than assuming a nominal chance rate.

## Trained arms

Train six arms for seeds 101, 202 and 303:

| Arm | Private input to bridge | Sender question input | Receiver input | Objective |
|---|---|---|---|---|
| aware_ce | Actual frozen source states | Actual public question | Learned prefix and actual question | Quartet CE |
| aware_aux | Same actual source states | Same public question | Learned prefix and actual question | Same quartet CE plus fixed auxiliary loss |
| blind_ce | Actual frozen source states | Fixed seeded question vector | Learned prefix and actual question | Quartet CE |
| private_constant | Fixed nonzero private states | Actual public question | Learned prefix and actual question | Quartet CE |
| text | Same fixed private states | Actual public question | Learned prefix, complete text facts and question | Quartet CE |
| symbolic | Same fixed private states | Actual public question | Learned prefix, complete symbolic facts and question | Quartet CE |

Within each seed, all six arms start from exact copies of one initialized bridge. They receive the same quartet order, four members, update count, optimizer settings and epoch count. The primary CE-only arm receives every target, nuisance and reorder example used by the auxiliary arm. An improvement cannot therefore be attributed simply to additional training examples or augmentation.

The full-text and symbolic arms receive all facts at the receiver and are competence/information controls. Equal trainable parameters and steps do not imply equal sequence lengths, FLOPs or runtime. The private-constant arm retains the real public question but has identical inputs across O and T, making its target-pair both-correct score structurally zero. Beating that zero alone is not evidence of useful communication.

## Exact loss and optimization

For each quartet, let zO, zT, zN and zR be the actual flattened 16-coordinate transmitted messages before receiver expansion. Do not replace them with unit-normalized directions, project them through another head, or detach their gradients.

Define:

    D(a,b) = (1/16) * sum_j (a_j - b_j)^2
    CE_quartet = (CE_O + CE_T + CE_N + CE_R) / 4
    L_aux = 0.5 * [D(zO,zN) + D(zO,zR)] + max(0, 0.25 - D(zO,zT))
    L_aware_aux = CE_quartet + 1.0 * L_aux

Average this objective over the quartets in the batch. All other arms minimize CE_quartet alone. The coefficient 1.0 and margin 0.25 are fixed engineering choices, not estimates of optimal settings. Because Tanh bounds each coordinate, D lies in [0,4]; magnitude changes are counted and arbitrarily increasing vector norms cannot satisfy separation.

This objective encourages invariance to the sampled nuisance edit and reorder, and a fixed minimum message distance for a target edit. Only a sparse subset of possible swaps and orders is observed; it does not enforce invariance over every irrelevant edit or the entire permutation group, and no new group-invariance theorem is claimed. It does not make separated messages correct by itself. CE provides answer supervision; the known quartet relationships provide additional task-dependent relational supervision. The joint-loss comparison cannot isolate invariance and separation components, and is not a comparison against every existing contrastive method.

Use eight epochs, batches of four whole quartets (16 records), AdamW learning rate 0.001, weight decay 0.01 and global gradient clipping norm 1.0. For epoch e numbered 1 through 8, shuffle the 512 training quartet indices with random.Random(seed + 1009 * e); flatten each batch in O,T,N,R order. Record order hashes and exact initialization hashes. Each completed arm has 128 updates per epoch and 1,024 updates total; the full study has 18 fits and 18,432 updates.

Select the earliest checkpoint achieving the minimum mean validation restricted-choice CE over all quartet members, independently for each arm. Do not select on validation message distance, the auxiliary objective, paired accuracy or any test metric. Log CE, both invariance terms, target distance, hinge contribution, query gates and times. There is no loss-weight, margin, epoch, architecture or seed search.

## Receiver competence gate and execution order

Train and select the three text-control fits first. Before any other arm is trained, require each seed's selected text checkpoint to achieve all three validation criteria:

- At least 80% target-pair both-correct accuracy.
- At least 90% nuisance prediction stability.
- At least 90% reorder prediction stability.

Use all 32 validation families and all eight queries. Record exact counts as well as percentages. All three text fits must finish before this gate is evaluated. If any seed fails, retain the three fits, histories, manifests and validation records, mark the study receiver-limited, and stop without fitting the latent arms or scoring the fresh test partition. A changed receiver, architecture, training budget or task after such a failure requires a separate design; it is not a retry of this study.

If the gate passes, finish the other five arms in every seed using the fixed settings. Train/validation data may be loaded and frozen features cached in advance. Fresh model predictions must not be computed until every required fit is complete and a selection manifest has recorded all selected checkpoint hashes, epochs, gates, configuration and completion counts. Load fresh test model cases only after that manifest is written and checked. Data-only split, tokenization and label-consistency checks before training are permitted and required.

This gate prevents an incapable receiver from making a channel comparison uninterpretable. Passing it establishes competence on this task and template under full information; it does not establish general language or reasoning ability.

## Fresh evaluation and metrics

Evaluate the six trained arms plus the following fixed conditions on all fresh quartets:

- For each of aware_ce and aware_aux, wrong private source, wrong sender-side public question and zero compressed message.
- Frozen no-channel receiver with its actual public question.
- Exact permutation-rank codec and deterministic lookup described below.

Wrong-source intervention uses the cyclic successor of the sorted test-family list, keeping query index and quartet member fixed. Reuse that mapping for every seed and arm. It replaces only private source information. Record whether the substituted world happens to retain the original answer. Also score predictions against the substituted source's actual answer, so following the supplied fact can be distinguished from arbitrary degradation.

Wrong-question intervention encodes the cyclic next object's public question at the sender while preserving the original private source and receiver question. Its rule must not inspect fact values. Zero-message intervention replaces the actual 2 by 8 compressed tensor with zeros before that arm's trained expansion.

For every condition report record accuracy and CE where applicable, target-pair both-correct accuracy, target answer-change rate, nuisance stability and both-correct accuracy, reorder stability and both-correct accuracy. Stability with wrong answers is not robustness. Do not compare paired accuracy with an assumed independent 1/64 chance rate; the errors and pair members are dependent.

The primary metric is target-pair both-correct accuracy. The primary contrast is aware_aux minus aware_ce on the one fixed rendering protocol above, averaged across the three fixed seeds. Compute each family's mean over its eight queries and the three seeds, then use 2,000 whole-family bootstrap resamples of the 64 test families with seed 66173. Reuse sampled family indices for all paired contrasts. Report the point difference and descriptive 95% percentile interval in percentage points, plus each seed's difference and range. The interval is conditional on these fixed seeds and is not a population-of-models claim.

Secondary contrasts are each main arm versus its wrong-source, wrong-question and zero-message conditions, aware_aux versus blind_ce, and all arms' absolute competence and stability. There is no outcome-driven choice among views; new wording or new vocabulary is outside this study. Wrong-question sensitivity supports a claim about using that question path, but lack of sensitivity does not by itself disprove a useful query-independent full-world code.

A practical advance requires the text validation gate, at least 80% fresh target-pair both-correct accuracy and 90% fresh nuisance and reorder stability for aware_aux, and a strictly positive mean correct-versus-wrong-source target-pair difference. A supported comparative improvement additionally requires the predeclared primary interval's lower endpoint to exceed zero. Report every component separately; no threshold is relaxed after outcomes. A positive small effect below these competence targets remains a limited result. No outcome from this small study is a breakthrough claim.

## Exact symbolic baseline and cost accounting

Fix the lexicographic order of all 8! color permutations. Parse the exact source grammar into its complete eight-object mapping, encode the permutation's rank from 0 to 40,319 in two unsigned big-endian bytes, decode it, and answer the parsed public object query. Verify round-trip reconstruction of every dataset world and reject out-of-range ranks. This is a hand-written, task-specific exact codec and lookup, not a neural language-model result.

Two bytes suffice for the complete world because 40,320 is less than 65,536. The 64-byte latent message is therefore 32 times larger than this exact task-specific payload before framing or public-question costs. A query-aware latent request also requires the question; conservatively charge its full UTF-8 bytes for each request. The two-byte complete-world code can be reused for different questions. The trained symbolic LM arm and deterministic rank-code oracle must remain separately labeled.

Report actual source and receiver token counts, text and symbolic UTF-8 bytes, latent payload bytes, public-question bytes, parameters and selected-checkpoint bytes. Record actual unique frozen-source and public-question encoding calls, cache bytes, preprocessing time, training/validation/test times and model download time. Logical uncached calls are a separate count, not a measured latency. No network latency, production cost, energy or bandwidth superiority is inferred from this experiment.

## CPU budget, retention and review gates

Use one CPU workflow with two PyTorch threads, a 35-minute job limit, and a 1,800-second study-process wall budget covering model downloads, data preparation, fits and fresh scoring. This is a maximum, not a promised runtime. The cap was fixed before implementation/outcomes to allow the 4.8-fold increase in training/validation records and the longer eight-fact full-information prompts relative to the previous query study. The wrapper must stop the child process at the fixed budget, retain completed work, and must not reduce epochs, omit arms, change settings or selectively retry to obtain a complete result. Dependency installation and tests precede the study; time remains for artifact retention. No paid API, GPU purchase or external model service is used.

Before launch, run the full suite with PyTorch and Transformers present. Tests must verify complete finite-family disjointness, exact token bags, query-independent source rendering, quartet edits, answer balance records, rank-code round trips, constant-input target invariance, all arm routing, raw-distance gradients and bounds, zero-gate initialization equivalence, selected-logit/gradient equivalence, exact shared initialization/orders/update budgets, validation-only selection, competence-gate sequencing and the all-selections-before-test rule.

Write data/model/code manifests before fitting; retain per-epoch history and current status incrementally; retain each completed fit's selected bridge in a portable small tensor format with hashes. Write the competence gate immediately and the complete selection manifest before fresh scoring. Append raw predictions after each completed evaluation condition, preserving partial records on failure, then produce deterministic compressed predictions and verify their uncompressed hash. Incomplete conditions must be identified explicitly and cannot be used for the primary complete-study claim.

Upload results, logs, manifests, histories, selected small bridge tensors and predictions with an always-run artifact step. A subsequent results commit must preserve all complete or failed conditions and the exact source/runtime provenance. Existing results and historical source modules are not edited. Any pre-execution change to this design or its fixed settings must be reviewed and committed before outcomes; a result-driven change belongs to a new experiment.
