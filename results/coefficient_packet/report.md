# Coefficient packet v2: original execution report

Study status: **complete**.
Complete frozen engineering success: **False**.

This is a fixed synthetic CPU/loopback experiment, with imposed application pacing.
It measures no pretrained model, answer quality, WAN RTT or universal communication protocol.

## Coverage and numerical gates

- Planned transport slots: 720; completed: 720.
- Diagnostic rows: 300/300; failed required diagnostic checks: 0.
- Eligible primary f64 routes: 54/54.
- Practical all-row coefficient/dual gates passed: 9/9.
- Float32 output errors are descriptive; direct wire bytes and declared fallback targets remain checked.
- Every omitted/error/ineligible slot stays in the fixed 720-slot denominator.

## Paired actual completion time

Ratios are coefficient wall time divided by comparator wall time; lower is faster.
Each cell is an equal-case geometric mean of four-pair median ratios.
An incomplete fixed population is not replaced by an available-only success claim.

| Population | Dual direct | Thin direct | Optimized Gram | Original Gram | Hindsight envelope |
|---|---:|---:|---:|---:|---:|
| Shaped warm (18 cases) | 0.272667 | 0.266467 | 0.980299 | 0.896141 | 0.980299 |
| Loopback warm (9 cases) | 1.209805 | 0.813874 | 0.883942 | 0.611237 | 1.209805 |
| Cold constrained (3 cases) | 6.229016 | 6.200734 | 0.999818 | 0.999273 | 6.229016 |

The hindsight envelope picks the fastest comparator median separately per case.
It is not a deployable policy. Token lookup is a non-equivalent byte/time reference and is excluded.
Cold trials include actual table/context transmission and setup receipts; warm gains do not imply cold gains.

## Criteria

- complete_correct_720_slots_and_required_diagnostics: True
- shaped_warm_against_all_three_and_hindsight_at_most_0.97: False
- loopback_at_most_1.10_and_worst_warm_case_at_most_1.25: False
- all_nine_primary_all_row_fidelity_gates: True

## Retention and interpretation

plan.json fixes all trial IDs/order before any diagnostics. trials.jsonl retains raw wall times,
phase times, waits, frames, endpoint byte/hash reconciliation and every failed/ineligible slot.
objects/ retains exact content-addressed packets, wire streams and receiver arrays;
object_manifest.jsonl maps their actual bytes. fixtures.npz and fixture_manifest.json retain all inputs.
diagnostics.jsonl records both precisions, native/fallback routes and full-vector errors.
No expected prefix, oracle decision, precomputed coefficient or prepared packet enters a measured encoder.
No canonical rank-deficient codec is tested here; unsupported paths transmit the declared dense prefix.
A successful numerical guard is not a universal final-vector or vocabulary-snap guarantee.
The comparison is against both efficient direct routes and an independently cheaper Gram guard.
This report is the original worker output; any later independent audit is a separate retained document.

Frozen design: 49cfba7261326d58e174e51c8ddf5abba6f4a966; SHA256 b8307677f5320c3413b5a1007bdfe26230a844c57503d53b4443e5fc542647ba.
