# Prior art for the next research stages

Search date: 23 November 2026 (America/Chicago). Status: **research leads, not established novelty**.

The frozen-message readout study is a diagnostic. The completion-time
controller is an engineering experiment. Neither becomes a new communication
method merely by improving on an earlier implementation in these repositories.
The search below also rules out broad novelty claims for incremental latent
reuse, receiver conditioning, latency-aware caching, or margin-certified reuse.

## Closest overlaps

| Proposed idea | Existing primary work | Consequence for this project |
| --- | --- | --- |
| Reuse latent information already held by a receiver and schedule updates by bandwidth/deadlines | [SCALE](https://arxiv.org/abs/2611.08390), submitted 5 November 2026, matches current tokens to acknowledged receiver state, uses cosine change scores, and schedules transmitted token counts. Its metadata includes reference-state versions. | Incremental reuse, acknowledgment-based state tracking, and deadline-aware latent transmission are already explicit ideas. |
| Certify a cache hit using a classification margin and a perturbation bound | [LipCache](https://arxiv.org/abs/2608.13144), also available from [TU Wien](https://repositum.tuwien.at/handle/20.500.12708/232424), constructs per-sample reuse radii from a local margin and classifier-head spectral norm. | The proposed geometric reuse certificate has a direct precedent. LipCache distinguishes consistency with its guard classifier from end-to-end accuracy. |
| Replace one similarity threshold with a correctness-aware cache policy | [vCache](https://arxiv.org/abs/2502.03771), accepted at ICLR 2026, learns entry-specific thresholds online with user-specified error-rate guarantees. | Correctness-constrained semantic caching is already a research area. Its reference response is full model inference; response agreement does not establish factual truth. |
| Recover a binding from frozen representations when the model answers incorrectly | [Propositional probes](https://arxiv.org/abs/2406.19501) decode entity/property bindings; [Legible Failures](https://arxiv.org/abs/2609.11216) studies recoverable in-context bindings and repair. | A successful readout would localize an accessibility problem in this particular bridge, not establish that the general phenomenon is new. |
| Condition cross-model communication on receiver state | [XKV](https://arxiv.org/abs/2608.20617) pools both caches and translates into heterogeneous receiver geometry. | Receiver-specific communication is not itself an original contribution. |
| Optimize caching for latency rather than hit rate or bytes | [Caching with Delayed Hits](https://www.microsoft.com/en-us/research/publication/caching-with-delayed-hits/) and [Latency-Aware Caching with Delayed Hits](https://www.usenix.org/conference/nsdi26/presentation/keren) explicitly optimize latency-sensitive caching. | A faster controller must be evaluated against strong simple policies; changing the objective alone is not a new research claim. |

SCALE's evaluated domain is cyber-physical perception and control. A small
language-binding counterexample would test an extension of its reuse principle,
not refute its reported results. The relevant mechanism is described in its
[full text, Sections 4.2–4.4](https://arxiv.org/html/2611.08390v1): maximum cosine
matching, additive change scores, and reuse of an acknowledged reference state.

## What a linear certificate would actually establish

Suppose a frozen, query-specific decoder has scores
`f_j(z) = w_j^T z + b_j`, and cached representation `z` predicts class `y`.
For a new representation `z + delta`, Cauchy–Schwarz gives the sufficient rule

`f_y(z) - f_j(z) > ||w_y - w_j||_2 * ||delta||_2` for every `j != y`.

If the inequalities hold, the fixed decoder's winning class cannot change.
This is standard robustness geometry, with a direct caching precedent in
LipCache. It is not a proposed new theorem.

The certificate concerns **decoder agreement**. It cannot establish that a
private fact stayed unchanged, that the encoder noticed an edit, or that either
prediction is correct. A decoder that ignores the relevant fact can consistently
return the same wrong answer. Quantization and numerical error also need to be
included in the perturbation bound. A changed query, decoder, encoder, precision,
or cached representation invalidates an unchecked reuse decision.

## Narrow testable question

One research lead remains worth investigating, without claiming it is unoccupied:

> In a stateful cross-model channel with an independently competent decoder,
> can a query-specific consistency certificate plus explicit cache-version
> validation reduce stale-answer errors at a useful total cost, compared with
> similarity reuse, verified semantic caching, and exact structured updates?

The unresolved quantity is the **complete operating tradeoff**: useful answer
accuracy, false reuse after factual edits, unnecessary refresh after harmless
changes, feedback traffic, and total computation. Combining known ingredients
does not establish novelty. A meaningful result would need to identify and
explain a repeatable operating regime that the strong alternatives miss.

A bounded CPU experiment could use the existing frozen encoders and a readout
that first passes fresh-family competence and source-dependence checks. Use
separate trajectories containing queried-fact replacements, irrelevant edits,
meaning-preserving paraphrases, cache eviction, and missing or delayed
acknowledgments. Keep source content and communication failures independently
generated. The public query and verified cache metadata may inform a policy;
gold facts and future failures may not.

Compare full refresh, exact text/structured updates, a tuned cosine threshold,
an appropriate correctness-aware cache policy, and the certified rule. Match
error budgets or transmitted-byte budgets on validation data before evaluation.
Charge the current sender encoding even when transmission is skipped, as well
as certificate computation, setup, fingerprints, acknowledgments, recovery,
and receiver work. Report disagreement with full refresh separately from
ground-truth errors. Exact structured updates remain a particularly strong
baseline for the current four-attribute toy task.

Do not proceed to a communication-efficiency claim if full refresh cannot
recover changed facts reliably. Do not interpret failed probes as proof that
information is absent. For probe selectivity and capacity, relevant precedents
are [control tasks](https://aclanthology.org/D19-1275/) and
[minimum-description-length probing](https://aclanthology.org/2020.emnlp-main.14/).

## Search boundaries

This was a bounded search across primary arXiv papers, ACL/ICLR proceedings,
USENIX publications, an institutional repository, and author-maintained code.
SCALE and vCache mechanisms were checked in full-text sections; the institutional
LipCache abstract explicitly states the margin/norm reuse construction. The
OpenReview PDF endpoint returned a browser-verification page, so vCache was
checked through its official proceedings page and arXiv v5 full text instead.
The search is neither an exhaustive novelty proof nor an independent replication
of those papers' experimental claims. The proposed question has not been tested
in this repository, and no performance or novelty result is asserted here.
