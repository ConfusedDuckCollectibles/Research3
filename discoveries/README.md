# Experimental findings

These are reproducible observations, including negative results. This index does not claim that the underlying methods are novel.

| Study | Finding | Evidence |
| --- | --- | --- |
| Synthetic bridge | A learned channel can solve the aligned toy task, but an exact packed code uses fewer bytes | [Report](../results/synthetic.md) |
| First pretrained pilot | The informative bridge does not outperform the equally trained no-private-information control | [Report](../results/pretrained_pilot/README.md) |
| Fact-change study | A small correct-source effect appears, with poor absolute accuracy and nuisance stability | [Report](../results/counterfactual_study/README.md) |
| Public-query conditioning | Primary improvement is inconclusive; the public-question intervention barely changes answers and practical targets fail | [Report](../results/query_conditioned_study/README.md) |

Read [continuation instructions](../docs/continuation.md) and the machine-readable [checkpoint](../research_state.json) before starting new work.


## Frozen-message readout diagnostic

[Complete findings](../docs/readout_results.md): byte-exact behavioral
reconstruction passed, but the small-message linear probe reached 9.90% paired
correctness versus 7.81% for the original receiver; the +2.08-point primary
interval includes zero. The conditional MLP reached 13.54%. A much larger
source-state reader reached 82.03% on the original template and 0% under the
source wording/order shift. These findings localize limitations of this
implementation and its readers; they do not establish irreversible information
loss, a useful communication method, or scientific novelty.

## Binding-sensitive training

[Complete audited findings](../docs/binding_results.md): matched CE-only and auxiliary-trained latent arms both answered 0 of 1,536 fresh target pairs correctly, while adapted full text answered 1,529. All 18 fits and 86,016 fresh records completed. The auxiliary arm's 99.61% nuisance stability and large zero-message intervention effect did not establish useful private-information transfer. The two-byte task-specific oracle remained exact. The fixed auxiliary comparison failed; all original evidence and the independent audit are retained in [the result directory](../results/binding_study/).


## Alignment computation and packet limits

[Complete audited findings](../docs/alignment_results.md): the guarded dual core was 1.17–3.02x faster than the efficient thin-SVD baseline over twelve CPU fixtures. All 15,759 required checks passed. The original Gram packet reduced warm bytes but cost 1.86–2.29x the arithmetic time of fast dual-direct; one accepted nonunique canonical float32 packet had 35.89% relative vector error despite unchanged nearest IDs. The independent audit and every original outcome are preserved. This is a scoped numerical implementation result, with no model-quality, complete agent-speed or novelty claim. The next [coefficient-packet TCP design](../docs/coefficient_packet_design.md) is separately preregistered before outcomes.
