# Prior art and a candidate direction: persistent state updates

Review date: **2026-11-24**.

## Conclusion and scope

The next counterfactual experiment is a **causal-validity milestone**, not a
novel communication method. Existing work already studies compressed latent
messages, heterogeneous receivers, receiver conditioning, reusable packets,
uninformative-input adaptation controls, unrelated-message interventions, and
targeted changes to sender facts.

A potentially useful next question is whether a learned communication interface
can **correct or delete facts in a persistent heterogeneous receiver under
uncertain cache state**, with a measurable advantage over compact symbolic
updates and whole-packet refresh. This is an **unestablished candidate research
niche**. This review does not prove that the combination is unoccupied, nor does
combining familiar mechanisms automatically create a research contribution.

This was a bounded search of primary research on arXiv, ACL Anthology, and
publisher pages, with terms covering latent communication, receiver conditioning,
incremental updates, state deltas, stale caches, receiver side information,
successive refinement, semantic retransmission, and cache-cost probing. Recent
LLM papers and older information-theoretic/networking precedents were included.
The review examined the full text and limitations of LatComm and Latent Cache
Flow; several other entries were checked at the primary abstract or publisher
description level. It was not an exhaustive systematic review, a reproduction
of the cited experiments, or a novelty certification. Reported results in those
papers remain their authors' findings.

## Existing overlaps

| Prior work | Relevant established idea | Boundary for our claims |
| --- | --- | --- |
| [On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer / PLatour and PACT-Check](https://arxiv.org/abs/2611.06976), 2026 | Task gains can arise from uninformative sender signals, self-communication, or receiver-only adaptation. | A trained information-free control is necessary. Better accuracy than a frozen no-channel model is insufficient evidence of useful communication. |
| [Draft-KV: Learning Useful Latent Communication Between Language Models](https://arxiv.org/abs/2609.34754), 2026 | Studies unrelated-message controls and trains an interface between frozen models to deliver useful source-dependent information. | Useful cross-model content transfer by itself is already a research objective and reported result. |
| [Hidden Messages Between Diffusion Language Model Agents: Causal Evidence for Latent Communication](https://arxiv.org/abs/2610.28905), 2026 | Tests message derangement and targeted substitutions of sender facts; also studies timing of exchanges. | A fact-change intervention is a sound validation method with precedent, not our invention. The diffusion setting differs from our autoregressive model pair. |
| [When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs](https://arxiv.org/abs/2608.04893), 2026 | Distinguishes having a cache from receiving the correct example's cache, including tasks where private source information is needed. | Zeroing a message alone is inadequate; correct-source versus mismatched-source comparisons remain essential. |
| [Latent Cache Flow: Model-to-Model Communication Without Text](https://arxiv.org/abs/2605.22863), 2026 | Compresses cache transfer and supports differing contexts; explicitly aims to summarize information the target lacks. | “Send only what the receiver does not know” is not a sufficient novelty claim. |
| [Dual-Cache Latent Space Communication between Heterogeneous Language Models / XKV](https://arxiv.org/abs/2608.20617), 2026 | Uses both sender and receiver caches and supports different model families, depths, cache geometry, and tokenizers. | Conditioning on receiver state or bridging heterogeneous architectures is already covered in relevant work. |
| [Receiver-Conditioned Latent Communication gives 94% CacheBack](https://arxiv.org/abs/2609.32046), 2026 | Uses a description of receiver needs to filter and compress sender cache content. | Query-dependent selection alone is not new. |
| [LatComm: Reusable Context Compression for Latent Communication between Large Language Models](https://arxiv.org/abs/2610.26840), 2026 | Encodes context once into a fixed-size packet, then lets a different frozen receiver read it for multiple questions. | Reusable, query-independent, cross-model packets already exist. |
| [Augmenting Multi-Agent Communication with State Delta Trajectory](https://aclanthology.org/2025.emnlp-main.518/), EMNLP 2025 | Communicates token-wise changes in internal model state alongside text. | “State delta” is established terminology. Successive token-state differences must be distinguished from updates to an acknowledged version of persistent receiver knowledge. |

### Older theory and systems precedents also matter

- [Source Coding When the Side Information May Be Delayed](https://ieeexplore.ieee.org/abstract/document/6468092), IEEE Transactions on Information Theory, 2013, studies source coding when the encoder may not know the delay of decoder side information. Uncertainty about what information the receiver possesses is not a new information-theoretic problem.
- [Adaptive Bit Rate Control in Semantic Communication with Incremental Knowledge-based HARQ](https://arxiv.org/abs/2203.06634), 2022, combines incremental semantic retransmission and adaptive coding rates. Progressive refinement and paying more bits when needed have clear precedent.
- [CRRP: Cost-Based Replacement with Random Placement for En-Route Caching](https://globals.ieice.org/en_transactions/information/10.1587/transinf.E97.D.1914/_p), 2014, uses probes to detect changes in cache-related costs. Bounded reprobes and recent feedback in the companion transport study should be described as engineering choices, not new generic caching concepts.

## What the examined LLM papers leave unresolved

The [LatComm full text](https://arxiv.org/html/2610.26840v1), especially its
conclusion and limitations, explicitly says that its motivating multi-agent and
agent-memory applications were not evaluated. Its evaluation uses multiple-choice
questions over supplied contexts. A new model pair also requires its own trained
interface and packets.

The [Latent Cache Flow full text](https://arxiv.org/html/2605.22863v1), section 5,
describes its partitioned-context experiment as controlled and identifies richer
workflows with longer contexts, tool outputs, heterogeneous roles, and multi-turn
communication as further work.

These limitations support investigating persistent, changing receiver state.
They do **not** demonstrate that no other paper has addressed it. The bounded
search did not establish an exact prior experimental match combining all of:

1. Distinct frozen sender and receiver models with a persistent communication state.
2. Repeated factual corrections and deletions, not only independent static documents or appended dialogue.
3. Uncertainty about which packet/source version the receiver actually retains.
4. Explicit charges for feedback, repair, refreshes, and interface setup.
5. Causal checks that changed target facts change answers while irrelevant edits preserve them.
6. Strong compact symbolic-delta and cached full-text/packet baselines at matched task quality.

That conjunction is a **candidate empirical question**, not a categorical claim
of novelty. A broader search should examine streaming context compression,
incremental RAG/cache maintenance, distributed source coding, cache invalidation,
and state synchronization before any publication claim.

## A falsifiable next study

### Hypothesis

After useful source-dependent transfer is established, a recipient-specific
update interface may preserve factual correctness across edits while reducing
the total cost of refreshing a long-lived receiver, relative to practical
alternatives. The experiment must be allowed to find that exact symbolic updates
or ordinary caching are better.

### Workload and controls

Use evolving private evidence with insertions, corrections, deletions, irrelevant
edits, and controlled cache loss/version mismatch. Hold receiver questions fixed
within targeted interventions. Keep unchanged-fact and nuisance interventions,
and retain trained information-free, wrong-source, and stale-source controls.
Separate failure to read the evidence from failure to apply an update by using a
competent full-information receiver control.

Compare against at least compact symbolic patches with version validation,
compressed text patches plus caching, cached full-text inference with the required
invalidation/recomputation, and whole-packet refresh using the same interface.
Give each method the same source facts, allowed receiver feedback, recovery
options, and adaptation budget. A method with a status API has an extra capability;
charge it and identify it separately.

### Evidence and accounting

Measure answer correctness after each update, correct tracking of changed facts,
stability under irrelevant edits, stale-fact error rate, recovery success, and
the cost/quality frontier. Count actual finite-precision message bytes including
headers, source/version identifiers, receiver queries, acknowledgments, retries,
and refreshes. Separately measure encoding, receiver computation, retained state,
training/setup cost, and the reuse required to amortize them. Do not substitute
message length or fewer receiver prefix positions for end-to-end cost.

Freeze task families, workloads, comparison rules, and selection criteria before
held-out evaluation. Group uncertainty by independent source-world or trajectory,
and report multiple training seeds. Preserve the current negative pilot and every
subsequent negative result.

## What would and would not be a research advance

A useful result could be a reproducible operating region where incremental
updates improve the cost/quality frontier after all feedback and recovery costs,
or a clear failure mechanism showing why an apparently efficient latent channel
cannot safely maintain evolving facts. Either would need strong baselines and
evidence beyond the four-object lookup task.

Passing the next lookup experiment would establish a more credible prototype.
It would not establish a universal latent language, a bandwidth advantage, or a
breakthrough. Four eight-valued facts require only 12 bits of state, and a selected
object plus a replacement color requires five bits before framing; the current
64-byte latent payload has no bandwidth advantage over those symbolic encodings.

