# Contextual information recovery: original worker report

This is a frozen-model numerical assay with known public reference tokens. It contains teacher-forced prediction-step states, not generated agent messages.

Recorded 256/256 numerical records; 256 completed and 0 operational errors.
Required validation passed: **True**.
The fixed primary 0.1% recovery target: **False** (39/128 successes; 122 required). Missing or unsupported records are not removed from the planned denominator.

## Recovery from final outputs

| Probe | Inverse/forward convention | <=1e-6 | <=1e-4 | <=1e-3 | <=1e-2 | Finite-error median | Finite-error maximum |
|---|---|---:|---:|---:|---:|---:|---:|
| unique | author_matched_public32 | 0/128 | 0/128 | 39/128 | 128/128 | 0.00167594 | 0.00703223 |
| unique | author_ideal_public64 | 0/128 | 0/128 | 39/128 | 128/128 | 0.00180058 | 0.00675987 |
| unique | reference_f64 | 128/128 | 128/128 | 128/128 | 128/128 | 3.02363e-12 | 1.36746e-11 |
| unique | reference_f64_cast_f32 | 0/128 | 128/128 | 128/128 | 128/128 | 7.50194e-06 | 2.29621e-05 |
| repeated | author_matched_public32 | 0/128 | 2/128 | 53/128 | 128/128 | 0.00121922 | 0.00444973 |
| repeated | author_ideal_public64 | 0/128 | 1/128 | 51/128 | 128/128 | 0.00128275 | 0.00453809 |
| repeated | reference_f64 | 128/128 | 128/128 | 128/128 | 128/128 | 3.20099e-12 | 1.27397e-11 |
| repeated | reference_f64_cast_f32 | 0/128 | 128/128 | 128/128 | 128/128 | 2.33385e-05 | 7.17754e-05 |

Error medians and maxima describe finite available estimates only; the success counts retain all 128 planned records in each cohort.

## Exact finite-codebook collisions

| Probe | Channel | Confirmed collisions | Confirmed distinct | Missing/invalid | Full ceiling | Full ceiling bounds |
|---|---|---:|---:|---:|---:|---|
| unique | author_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| unique | reference_f64_cast_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| unique | direct_gram_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| unique | raw_state_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| unique | token_only_f32 | 64 | 0 | 0 | 0.5 | [0.5, 0.5] |
| repeated | author_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| repeated | reference_f64_cast_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| repeated | direct_gram_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| repeated | raw_state_f32 | 0 | 64 | 0 | 1 | [1, 1] |
| repeated | token_only_f32 | 64 | 0 | 0 | 0.5 | [0.5, 0.5] |

unique: preexisting identical-numerical-source collision IDs: none observed.
unique: alignment-created witness IDs: none observed.
unique: exact centered-Gram-loss witness IDs: none observed.

repeated: preexisting identical-numerical-source collision IDs: none observed.
repeated: alignment-created witness IDs: none observed.
repeated: exact centered-Gram-loss witness IDs: none observed.

## Input regime

- unique: native first-state color accuracy 0.171875; paired both correct 0.03125; the two probes' first-state readouts are duplicates. Observed source-whitening inverse-condition-factor maximum 1033.79.
- repeated: native first-state color accuracy 0.171875; paired both correct 0.03125; the two probes' first-state readouts are duplicates. Observed source-whitening inverse-condition-factor maximum 917.631.

## Interpretation and scope

An exact answer-changing collision with different source values establishes a limitation on that retained pair under its identical public conditions. The ceiling is a finite paired-codebook statement, not population accuracy.

An absent collision does not establish useful semantic transfer. Failed inversion or small distance does not prove absence of information. Positive regularization permits exact recovery under the stated conditions while the numerical inverse can be poorly conditioned.

The actual author float32 return is distinguished from the NumPy float64 mathematical reference and the latter's single float32 cast. Different nonunique repeated-ID completions are not prefix-equality failures. Float64 validation failure may itself reflect conditioning.

The forced tokens may be outside the original generation protocol's top-p support. There is no trained adapter, receiver generation, timing-performance claim, or established novelty claim.

All private truths and captured anchor choices are used only after the public-input inverse returns. Raw record and array manifests retain rejections, numerical residuals, exact comparisons and partial evidence.
