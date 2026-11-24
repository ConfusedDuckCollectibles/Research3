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
