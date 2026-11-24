# Contextual information recovery: frozen-model numerical assay

**Status: frozen preregistration under the owner's existing authorization to continue this research.** Implementation may proceed after this design is published. Activate the single empirical run only after the complete reviewed source is published and the offline test gate passes. No pretrained extraction or outcome has been executed for this study.

## 1. Question and interpretation

On authentic contextual activations from one frozen Pythia-14M checkpoint, with the reference token IDs fixed publicly, how accurately can the actual StateBridge float32 prefix recover the centered source Gram matrix? Do answer-changing paired contexts ever produce exactly identical final payload bytes?

The [information-recovery note](statebridge_information_recovery.md), frozen at commit ac53603c20de1d28e01cb697cb81b94f6c06db40 (SHA-256 f78a41cee2b1ee90b349bcbee0fe00a588b9738863034172a8c8c09ddc3992d7), supplies the exact-arithmetic motivation. Positive regularization can preserve the source Gram while making recovery poorly conditioned. Full row rank of the actual pre-prefix permits scale recovery from the known reference mean. Repeated IDs do not automatically imply loss: a selected residual completion can retain additional information.

This assay concerns a known, fixed reference. The original StateBridge prefix-only interface does not necessarily transmit its reference IDs. Here the predetermined probes are public, so their IDs and embedding rows are legitimate inverse inputs. This side information must not be silently assumed in a broader operational claim.

A failed inverse or small inter-world distance does not prove information destruction. An exact answer-changing payload collision, under identical public conditions, does establish a limitation on the evaluated pair. No collision does not establish semantic usefulness. The retained states come from teacher-forced sequences; they are not naturally generated messages. A forced token may lie outside the authors' default top-p sampling support, so matching prediction-step indices does not establish that the original generation protocol would produce these messages.

No training, learned probes, adapters, receiver generation, model selection, parameter search, networking benchmark, or performance sweep is included. No novelty or efficiency claim is a success criterion.

## 2. Fixed model, runtime and provenance

Use a homogeneous model and its own input embedding table:

- Model: EleutherAI/pythia-14m.
- Revision: cf967c0a9a04383db6f7b1108d86b2962634b4ac.
- Model computation and weights after loading: CPU float32.
- Attention implementation: eager, explicitly requested.
- Evaluation mode, gradients disabled, cached incremental inference.
- One Torch intra-op thread, one inter-op thread, and one BLAS/OpenMP thread.
- Python 3.11.18; torch 2.6.0+cpu; transformers 4.48.2; NumPy 2.4.6; tokenizers 0.21.4; safetensors 0.8.0; huggingface-hub 0.36.2.
- Set the Python, NumPy and Torch seeds to 41029. Dataset generation uses its own Python random.Random(41029), so unrelated RNG consumption cannot change the cohort.
- Enable deterministic Torch algorithms. An unsupported operation is a recorded failure, not permission to switch implementations.

Use the existing six-file public model lock, verify every byte count and SHA-256 before loading, then load locally with local_files_only=True and trust_remote_code=False.

| File | Bytes | SHA-256 |
|---|---:|---|
| config.json | 698 | f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17 |
| generation_config.json | 111 | 494cbba887cc299273f0c54d7612596918e74c7881a27cbd95e4e7e2464ea090 |
| model.safetensors | 28,143,920 | 116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1 |
| special_tokens_map.json | 441 | 10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c |
| tokenizer.json | 2,114,042 | 870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2 |
| tokenizer_config.json | 4,834 | eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809 |

The launcher records its source commit, workflow/run/attempt, design and theory hashes, all executed source hashes, installed versions, OS, CPU model, available CPU count/affinity, BLAS details and thread settings. Thread environment variables are set before scientific-library imports.

## 3. Exact paired-context generator

Ordered specimen names:

    alder, birch, cedar, elm, fir, maple, oak, willow

Ordered colors, also the native readout candidate order:

    red, blue, green, yellow, purple, orange, black, white

There are exactly 64 families, F000 through F063. A world is a permutation p of color indices 0 through 7 in the ordered specimen positions.

Use this generation order:

1. Initialize a fresh Python random.Random(41029) and an empty set of reserved world tuples.
2. For family index f from 0 through 63, set target t=f mod 8 and partner u=(t+1+((f//8) mod 7)) mod 8. These indices use no random draws and no model outputs.
3. For each candidate, create the list [0,1,2,3,4,5,6,7], then call that generator's shuffle exactly once.
4. Create p_swap by exchanging entries t and u.
5. Reject the candidate if p is already reserved; otherwise reject it if p_swap is already reserved. No other rejection or balancing rule applies.
6. Accept the first remaining candidate; reserve both tuples before advancing to the next family.
7. A global limit of 1,000,000 candidate draws is a generation-failure guard. Do not replace the seed or generator if reached.

Record all accepted permutations, total candidate count, per-family draw/rejection counts, target and partner indices, ordered world IDs, and the complete 128-world reservation-set hash. Labels are p[t] and p_swap[t], which differ because p is a permutation. The swap changes two bindings while preserving the complete specimen/color word bag. It must not be described as changing only one fact.

The public question is exactly:

    What color is the {target_name} specimen?

The source prompt, with ASCII LF separators and no final space or newline, is exactly:

    Family F000.
    Private record:
    alder: red
    birch: blue
    cedar: green
    elm: yellow
    fir: purple
    maple: orange
    oak: black
    willow: white
    Question: What color is the alder specimen?
    Answer:

The displayed mapping is a formatting example; the generator supplies the permutation, family and target. Row order never changes. The actual prompt ends at the colon in Answer:.

Public inverse/grouping information consists of the family ID, exact question, probe identity/IDs, fixed model/table identity and algorithm parameters. Original/swapped variant, world permutation, color answer, private source prompt, true source states and native logits are not inverse inputs. Family IDs do not identify the within-pair variant.

No historical learned adapter is used. This is a fresh fixed generator and a new assay; it is not a matched comparison against earlier task-study accuracy.

## 4. Tokenization and the two public probes

Use the pinned tokenizer with add_special_tokens=False. Encode each string independently:

    " one", " two", " three", " four", " five", " six", " seven", " eight"

Each must produce exactly one token, and the eight IDs must be distinct. The unique probe is those eight IDs in that order. The repeated probe is the first ID eight times.

Encode each leading-space color candidate independently in the ordered list from Section 3. Each must be one token, and all eight IDs must be distinct. Require all relevant IDs to be within the actual input/output tables.

Tokenize all 128 prompts before any model forward. Require each paired prompt to have the same token length and token multiset, and every prompt to have at most 256 tokens. Retain exact IDs and lengths. A failed tokenizer/shape preflight produces an explicit unsupported_preflight result and stops before model extraction. Do not choose replacement words, trim tokens, alter a template, or select another checkpoint.

The unique-ID reference's actual centered and uncentered singular spectra are measured and retained. Distinct IDs alone are not a rank guarantee. Rank deficiency does not trigger replacement IDs or suppress the planned record.

There are 64 families ×2 worlds ×2 probes =256 sequences. The fixed record order is family increasing, original then swapped, and unique then repeated within each world. Record IDs are F000/original/unique and the corresponding fixed alternatives. The 128 unique-probe records are the primary numerical cohort. The other 128 are descriptive structural controls.

## 5. Authentic prediction-step extraction

Use the output of model.gpt_neox.layers[-1], before gpt_neox.final_layer_norm. This architecture-location adapter is explicit: the authors' hook-selection helper does not contain a GPTNeoX branch. The captured tensor has the same before-final-normalization semantics.

For each sequence, perform exactly eight cached prediction calls:

1. Call the model on the complete source prompt. Capture the last-position final-block state; it predicts forced token 1.
2. Feed forced token 1 with the returned cache and updated attention mask/positions. Capture the state predicting token 2.
3. Continue through the call consuming token 7 and predicting token 8.

Token 8 is scored but need not be consumed. Do not substitute states after the corresponding token or hidden_states[-1]. Retain a K=8 by d=128 float32 source matrix, exact consumed/forced IDs, prompt length, cache/position metadata and the hook identity. Each sequence starts with a fresh cache; no cache is shared between worlds or probes.

The model remains frozen/eval, with no sampling or generation API. Each alignment call has batch size one. The authors' method flattens batch and sequence together, so aligning different worlds jointly would be invalid.

At every step retain the forced token's full-vocabulary log probability, computed from the actual float32 next-token logits after conversion to float64 for stable log-softmax analysis. Retain their sum/mean; no probability threshold selects or excludes a sequence. Full vocabulary logits need not be stored: retain their hashes, relevant logits and the source states, which permit a separate pinned final-normalization/head reconstruction.

At the first state retain the eight color logits, restricted eight-way probabilities and first-index argmax in the fixed color order. Report native per-world accuracy and paired-both-correct on the primary 128/64 denominators. This is the frozen model's native readout, not a trained probe. Its success or failure does not select contexts or gate the numerical analysis. The same-world first prediction-step state and retained eight float32 color logits must agree byte-for-byte across the unique/repeated probes. A mismatch is a contract failure. Those duplicate readouts are not independent extra evidence.

## 6. Actual author float32 forward and captured intermediates

Vendor the exact complete authors' file and Apache-2.0 license with provenance:

- Repository: YanwenPneg/StateBridge.
- Commit: 3f6bf5442c6e8848555a6132516e6d36f35444fb.
- File: methods/state_bridge.py.
- File SHA-256: 415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d.
- Verified license blob: 261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64.

Parse the file as Python AST, locate exactly StateBridge._align_hidden_sequence, and compile only that function with its original body, signature and source locations. Supply torch and torch.nn.functional as its globals and a minimal self context. Do not import or execute the original application's module-level imports, CLI or other methods. Retain the selected AST/body digest and vendor-file identity.

The self context supplies only the frozen embedding layer, CPU device, float32 output dtype, detached float32 vocabulary, its author-computed target norm, adaptive_reg=0.001 and snap_ratio=0.3. The method itself fixes the covariance floor at 0.000001, row-norm floor at 0.000001 and functional normalization floor at 0.000000000001. No later pooling, noise or prefix processing is applied.

The exact unmodified method performs its original float32 centering, repeated symmetric matrix powers, dense SVD, determinant correction, dewhitening, row calibration, full-vocabulary anchoring and output cast. Require its actual returned tensor to be float32; author_f32 refers to those actual return bytes, not a post-hoc precision conversion. Canonical serialization changes layout/endianness only, not numerical values. Do not replace it with the faster dual solver or remove repeated operations.

A scoped Python return-event profiler may copy locals only when the returning frame's code object is this exact function. Retain available Hc, H_w, E_w, mu_H, mu_E, Cov_H, Cov_E, the actual rotation named R, sqrt_Cov_E, aligned_flat, flat and top_ids, plus the actual returned tensor. Capture occurs after the function has computed its return. Save and restore the prior profiler in finally; no captured local is injected back into the function. Test byte-identical return values with and without capture on fixed offline fixtures, including proper-rotation and repeated-ID cases. The profiler is instrumentation, not an alternative implementation.

Copying, hashes and retention are charged to the scientific budget. Float32 covariance/root operations and approximate orthogonality are measured artifacts; no exact-mathematics identity is assumed for them.

## 7. Fixed forward controls and public precision contracts

For each retained source S and publicly gathered reference R, compute these controls without another model call:

| Arm | Forward/output contract | Public inverse convention |
|---|---|---|
| author_f32 | Actual unmodified authors' method, float32 return | Public quantities computed in float32, then interpreted in float64 |
| reference_f64 | NumPy float64 mathematical equations on the same retained S values and table values | Public quantities computed in float64 |
| reference_f64_cast_f32 | The reference_f64 final prefix rounded once to canonical float32 | Same public64 quantities as its originating forward |
| direct_gram_f32 | Float64 reduced centered source Gram rounded once to float32 | Direct read, no alignment inverse |
| raw_state_f32 | The authentic source array itself | Byte/value control only |
| token_only_f32 | The eight gathered embedding rows | Constant within each public paired group |

The float64 reference uses the previously published dense_reference and postprocess equations from latent_bridge/alignment_core.py, pinned to commit 7aee330f7799751a6a9f4b2a8decb612abe9c266. Hash the imported file and retain its output/intermediates. It is a mathematical reference with covariance eigenbasis reuse, not an execution of the authors' float32 body in another dtype. At nonunique completions, a different float64 output is not automatically an error relative to the float32 completion.

Public32 means, covariance and square root are computed independently using only the fixed float32 reference rows. They must mirror the authors' Torch operation order, including the keepdim mean, float32 centering/covariance, symmetric averaging, clamp_min(eps).pow(0.5), and matrix multiplication grouping; NumPy float32 or a regrouped root is not a substitute. The target norm uses the authors' float32 vocabulary norm/mean expression. Public64 mirrors the frozen NumPy dense reference's _eigh/_from_values and np.sqrt path, with its corresponding float64 mean and vocabulary target norm.

For public32, scalar coefficients are the actual float32 representations of the declared operations: lambda=float32(0.001), anchor coefficient=float32(0.3), complement=float32(1.0-0.3), and the declared floors cast to float32. Public64 uses the declared float64 constants. Record every numerical value. The inverse may not infer these from source-dependent captured locals.

Both public contexts are computed and retained before any private-state model extraction, from the table, probe IDs and declared parameters alone. The public32 square root must agree byte-for-byte with the captured author sqrt_Cov_E for the same probe; means and other public computations are similarly checked where retained. A mismatch is a public-contract validation failure, not permission to pass the captured tensor secretly to the inverse or choose another convention.

A fixed secondary inverse also applies ideal-public64 quantities to author_f32. It is reported separately for all planned records. It never replaces the matched-public32 primary result, and the better convention is not selected per record or after outcomes.

## 8. Primary inverse: final output and public inputs only

The primary inverse accepts the emitted final prefix and a public context containing the probe IDs/rows, full vocabulary, numerical parameters, public mean, public dewhitening root and target norm. It never reads the source, true Gram, true pre-prefix, true anchor IDs, private rotation, answer or native readout.

All inverse arithmetic is float64. Recompute vocabulary cosine directions in float64 from the stored table values and the declared floor, including in public32 mode; never reuse a rounded float32 normalized-vocabulary cache. Public32 means, roots, gains, lambda and other scalar values still retain their producing-dtype values. The fixed procedure is:

1. Re-select the vocabulary anchor from each final row using the public normalized vocabulary, the declared normalization floor, and first-index argmax. The cheap inverse supports zero vocabulary rows and ordinary nonzero rows whose norm is at least the normalization floor. If any nonzero vocabulary row is below that floor, mark the cheap procedure unsupported; do not enumerate candidates or choose an alternative inverse.
2. Subtract the selected anchor contribution and divide by the declared complement to recover calibrated rows C. Reapply the same public selection rule to C and require its selected IDs to agree with the IDs selected from the final rows. Retain both derived choices and margins. A disagreement is an anchor_self_consistency numerical rejection, not permission to use the captured true IDs.
3. Use a float64 SVD least-squares solve for C transpose times s = K times public_mean. The fixed relative cutoff is machine_epsilon_float64 ×max(C transpose shape). Require numerical row rank K, finite scales and strictly positive scales. Retain singular values, cutoff, condition number, scales and mean residual. Do not classify slightly sub-target observed norms as definitely clamped: the scale solve handles every row together.
4. Set Z_hat=diag(s) C and Z_centered=Z_hat minus the broadcast public_mean. Solve D_public transpose times T transpose = Z_centered transpose in float64, where D_public is the actual declared public dewhitening-root matrix. This gives T=Z_centered times D_public inverse without explicitly forming that inverse. Do not substitute an ideal root or symmetrize the supplied root. Retain the reconstructed pre-prefix mean residual and the rewhitened centering residual before Helmert reduction.
5. Use the fixed float64 Helmert basis H, with column j (zero based) equal to 1/sqrt((j+1)(j+2)) in rows 0 through j and -(j+1)/sqrt((j+1)(j+2)) in row j+1. Form A_hat=H transpose times T and X_hat=A_hat A_hat transpose. Symmetrize X_hat by averaging with its transpose.
6. Eigendecompose X_hat. Every eigenvalue must be finite and lie in [0,K). Negative eigenvalues and values at or above K are explicit domain failures. Do not clip, truncate, regularize, replace, or retry them.
7. Apply g=K lambda x/(K-x) to every eigenvalue and reconstruct G_hat. A nonfinite reconstructed matrix is an explicit nonfinite_reconstruction failure. Here lambda exceeds the declared covariance floor in both public precision conventions, so no alternate spectral branch is selected.

For interpretation only, compare recovered anchors to the captured forward choices after the inverse has completed. Record source/target ranks and source whitening inverse-condition factors 1+g/(K lambda), public-root conditioning, mean-solve conditioning, anchor margins, scale and calibration residuals, Gram errors and every rejection reason.

Captured intermediates are used only after the fixed inverse returns, for the declared public-contract, anchor, calibration, mean and numerical residual diagnostics. They may not produce an alternative recovery answer or repair a primary inverse.

## 9. Ground truth, numerical target and validation

The numerical target G is computed in float64 from the retained float32 source interpreted as exact values: explicitly center its rows in float64, multiply by the fixed Helmert basis and form the symmetric reduced Gram. Retain both the reduced 7×7 Gram and full centered 8×8 Gram. A zero reference norm gives relative error zero only if the estimated difference is exactly zero; otherwise mark relative error nonfinite/failure.

The sole practical numerical reporting target is:

> At least 95% of all 128 primary unique-probe records recover G from author_f32 with the fixed matched-public32 inverse at relative Frobenius error at most 0.001.

This requires at least 122 successes. Unsupported/ineligible inversions and domain failures count as not recovered; they are never dropped from the denominator. Missing records make the completed-cohort claim unavailable. This 0.1% threshold is a reporting target, not a semantic, privacy, efficiency or originality threshold.

Also report success counts at the fixed thresholds 0.000001, 0.0001 and 0.01, all with the same 128 denominator; exact count, median and maximum error among successful finite estimates are supplementary and explicitly conditional. Report the repeated-control 128 denominator separately, without pooling it into the primary criterion.

For the full float64 reference, recovery error at most 0.00000001 on structurally eligible primary records is a validation check. Eligibility is determined before invoking the inverse from full centered source rank K-1 and uncentered reference rank K, using the same machine-epsilon SVD rank convention, plus the declared public anchor-geometry support. It does not depend on inverse success or error. Retain every eligible failure, source/reference spectrum and excluded count. Zero eligible cases cannot validate the procedure. A failed 0.00000001 float64 check may reflect conditioning or finite-precision arithmetic; full numerical rank alone does not guarantee that error bound. Such a failure is not automatically a code defect or a counterexample to the exact-arithmetic theorem.

All rows also retain the actual author-versus-float64 vector difference, the one-cast float32 difference, and the direct rounded-Gram reconstruction error. A nonunique-completion vector difference is labeled descriptive, not a required equality. Full-vector numerical error, finite-byte equality, exact Gram equality and anchor agreement are separate quantities.

No model-quality gate selects the numerical cohort. Native readout accuracy and forced-token likelihood only describe the input regime.

## 10. Exact collision endpoints and label ceiling

Serialize tensors for byte comparisons as C-contiguous little-endian float32 with no rounding beyond the arm's prescribed dtype conversion. Prefix shape, public model/configuration and probe IDs are fixed within each comparison. Compare bytes directly; a hash is only an index. Nonfinite or missing arrays do not count as collisions.

Within each family/question/probe group, compare original and swapped worlds. An alignment-created collision requires all of:

- The two required color answers differ.
- Actual source numerical values differ, not merely float32 signed-zero bits.
- Both final author_f32 arrays are finite and have exactly identical canonical bytes.
- Every legitimate public group input is identical.

Record raw-state byte equality separately from exact numerical value equality, so a signed-zero-only difference cannot establish lost source information.

Distinguish exact centered-Gram equality from rounded-Gram byte equality. Because every retained binary32 source value is dyadic, compute pairwise centered-Gram equality using integer arithmetic: express both source matrices under one common power-of-two denominator D; for each integer matrix A form T=K A minus its column-sum row broadcast; compare T T transpose exactly. Both denominators are K squared times D squared. This tests the exact real centered Gram of the stored states without depending on floating-point accumulation or the irrational Helmert basis. Retain the arithmetic convention, common exponent, exact comparison outcome and hashes of the canonical integer matrices. Separately retain direct_gram_f32 byte collisions and stored-float64 Gram equality. A claimed Gram-loss collision additionally requires exact dyadic centered-Gram inequality; equal-Gram source orientations are an expected collision class in the unique-product real-arithmetic characterization.

For each probe, compute the uniform finite paired-codebook optimal-label ceiling conditioned on public family ID/question/probe:

    (1/n) sum_over_public_groups_and_payloads max_label count(group, payload, label)

Each complete paired group contains two equally weighted, answer-different worlds. A collision contributes one correct maximum out of two; distinct payloads contribute two. Thus with 64 complete groups the ceiling is 1 minus collisions/128. Report primary and repeated ceilings separately, and distinguish preexisting source collisions from alignment-created ones.

This is a finite conditional codebook statement, not generalization accuracy or evidence that a practical decoder attains the ceiling. For each planned 64-pair cohort report c confirmed final-byte collisions, d confirmed distinct pairs and m missing/invalid pairs, with c+d+m=64. The count c includes every valid final-byte collision, including preexisting identical-source cases; alignment-created and exact-Gram-loss witnesses are separate subsets and must not replace c in the ceiling. If m=0, report the exact full ceiling. Otherwise report the bounds [1-(c+m)/128, 1-c/128], and label any completed-pair descriptive ceiling with its smaller denominator. An inverse failure does not remove a pair from collision analysis when its valid final arrays were retained. Missing/invalid arrays are never counted as noncollisions. No collisions is an inconclusive result about semantic accessibility.

## 11. Execution, retention and failure policy

The scientific run is a single child process with a hard 600-second deadline, beginning before its imports and including public model downloads, hash verification, tokenization, extraction, all forward controls/inverses, exact integer comparisons, summaries and scientific retention. Parent packaging/archival work after termination is timed separately and does not resume analysis.

Use the first workflow run and first attempt only. Publish the design, tests and all scientific source before activation. Do not rerun a model or change a scientific setting after observing outcomes. Pre-outcome offline test corrections are allowed only before the one empirical workflow is activated and must be documented.

Persist plan.json with all 256 record IDs at the earliest safe point. Write tokenizer/data/model/runtime/source manifests and public-array retention before extraction. After each record, save its source/output/intermediate arrays and append/flush its result row. After each completed pair/probe, append the collision result. Preserve partial files at timeout or failure; record missing planned IDs without synthesizing outcomes.

Retain:

- All contexts, permutations, questions, labels, prompt/forced/candidate IDs, lengths and generation manifest.
- The entire frozen float32 input embedding table, its exact shape/hash, and both public precision contexts.
- Every authentic source matrix, actual author final and specified intermediates, float64 reference pre/calibrated/final output and one-cast output.
- Every inverse output or partial diagnostic, source Gram, native scores, forced-token log probabilities and relevant logits/hashes.
- Exact numerical/byte collision metadata, full raw rows, all domain failures, plan coverage, summary and provenance.
- The vendored source/license, executed source hashes and the six model-file identities.

Content-addressed deduplication is allowed for identical arrays, provided every record maps to exact dtype, shape, byte digest and retained object. It must not replace equality checks with hashes alone. In particular, repeated public covariance/root/table arrays need not be stored hundreds of times.

The output includes concise results.json/report.md and raw records.jsonl/collisions.jsonl plus arrays and manifests. Flush partial results; do not hold the whole scientific outcome only in memory. A bounded log summary is a recovery aid; complete artifacts are the primary record.

Completeness/public-contract/required float64-validation failures remain visible and produce a nonzero scientific exit status, with all completed evidence retained. A fully completed valid run that misses the 95% author_f32 recovery target is a legitimate negative result and may exit successfully. Numerical rejection is recorded once; another method may not replace the failed primary answer. Unrecoverable process failures stop the run.

The resource estimate is feasible for one 128-dimensional model and 256 short sequences, but no runtime or target-passing prediction has been measured. Exceeding the budget is a retained incomplete outcome.

## 12. Required pre-execution tests and scope limits

Before model extraction, run offline tests of meaningful contracts: the exact generator and pairing/token-bag invariants; prediction-step hook indexing on a tiny local random GPTNeoX configuration; untouched author return equality with profiler restoration even on exceptions; a full-rank fixed mathematical inverse round trip; rank/domain/tiny-anchor rejection without clipping; a constructed exact collision and signed-zero distinction; independent dyadic Gram equality; no private arrays or true anchor IDs accepted by the primary inverse; output persistence and timeout accounting.

These tests are not pretrained outcomes. No threshold is calibrated against the planned 256 states.

Prior art includes [StateBridge](https://arxiv.org/html/2608.13317v1), [Wadia et al., ICML2021](https://proceedings.mlr.press/v139/wadia21a.html), [PACT-Check](https://arxiv.org/abs/2611.06976), and [Legible Failures](https://arxiv.org/abs/2609.11216). The assay applies an independently reviewed exact-arithmetic characterization to one pinned finite-precision implementation and a small authentic-state cohort. It does not claim a new whitening theorem, a new latent language, task-level benefit, privacy guarantee, or general model limitation.

Every interpretation must preserve the distinctions between exact arithmetic and numerical recovery, available and usable information, fixed forced probes and generated messages, the known-reference conditional setting and the original unknown-reference handoff, and this tiny homogeneous model and larger deployed systems.
