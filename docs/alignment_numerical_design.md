# Preregistered numerical study of a Gram-based alignment implementation

## Status, question, and scope

This design is fixed before inspecting any new numerical outcomes. It tests a specialization of standard Procrustes, polar-factor, and Gram-matrix identities to the StateBridge alignment interface. It is an algebra and implementation study: no language model inference, model downloads, task-accuracy experiment, or English-versus-latent comparison is included.

The primary question is whether a small-matrix implementation reproduces the unique aligned product of a dense reference within fixed float64 tolerances. A second, explicitly different arm specifies a deterministic optimum when that product is nonunique. Its correctness criterion is feasibility and optimality, not equality to an arbitrary dense SVD completion. Timing and packet size are engineering measurements after correctness checks.

The analytical assumptions and derivations are in [alignment_geometry.md](alignment_geometry.md). The pinned upstream source is [methods/state_bridge.py](https://github.com/YanwenPneg/StateBridge/blob/3f6bf5442c6e884855a6132516e6d36f35444fb/methods/state_bridge.py), commit 3f6bf5442c6e884855a6132516e6d36f35444fb, SHA256 415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d.

The upstream code uses float32 alignment. Our dense reference is an independently written NumPy float64 evaluation of its operations, with explicit ideal-equation and eigenvalue-floored tracks. Agreement here is not bitwise equivalence to the upstream PyTorch implementation or to a model-dtype output.

## 1. Fixed conventions and four implementations

Use one message at a time, with sender states S and reference embedding rows R of shape K by d. The guaranteed fast-path scope is 2 <= K < d. K counts all aligned rows; this study does not silently change the upstream batch-flattening convention.

A fixed Helmert matrix U spans the centered sample subspace. Column j, numbered from zero, has entries 1/sqrt((j+1)(j+2)) in rows 0 through j, entry -(j+1)/sqrt((j+1)(j+2)) in row j+1, and zero elsewhere.

### Dense reference

Center both inputs. Form both d by d covariances with divisor K, add lambda times identity, and apply symmetric eigendecomposition. In the upstream-faithful mathematical track, floor covariance eigenvalues at epsilon=1e-6 before either square-root power. Whiten, compute the full d by d cross-product SVD, and enforce determinant +1 by flipping the last left singular vector if necessary. Dewhiten with the paired floored receiver square root and restore the receiver mean.

The ideal-equation track has epsilon=0 and strictly positive lambda. It is reported separately. The dense reference records the whitened aligned product as well as the pre-calibration output. Within a call, reuse the computed sender whitening and receiver eigendecomposition for its paired powers; do not inflate the reference time by repeating identical eigendecompositions. No arm caches covariance decompositions across timed invocations.

### Guarded unique-product dual

Let A=U^T S, B=U^T R, G_A=A A^T, and G_B=B B^T. For the floored track apply the spectral map

    F(g) = g / max(g/K + lambda, epsilon).

Set X=F(G_A), Y=F(G_B), and

    M = X^(1/2) (X^(1/2) Y X^(1/2))^(-1/2) X^(1/2).
    Z = U M B + receiver_mean.

This branch requires full centered row rank on both sides and the numerical guard below. It never substitutes a pseudoinverse for a failed positive-definite inverse. Unsupported inputs receive an explicit status and, in the packet experiment, a raw-output fallback.

### Strong thin-SVD baseline for unique products

This is a standard small-matrix Procrustes implementation, included so a speed comparison does not rely only on avoidable d by d work. For full centered-rank A and B, take thin SVDs

    A = L_s diag(s) V_s^T, B = L_r diag(t) V_r^T.
    a = s / sqrt(max(s^2/K + lambda, epsilon)).
    b = t / sqrt(max(t^2/K + lambda, epsilon)).
    diag(a) L_s^T L_r diag(b) = P diag(c) Q^T.

Its pre-calibration result is

    Z = U L_s diag(a) P Q^T
          diag(sqrt(max(t^2/K + lambda, epsilon))) V_r^T
        + receiver_mean.

Apply the same full centered-rank and numerical-support guard, obtaining rank information from the squared input and cross singular values when available. Both reduced-input SVDs and the small cross SVD are inside each timed invocation. No source/receiver SVD is cached across calls. Do not impose a determinant correction on the reduced rotation: the spare feature direction already permits a proper ambient extension.

This baseline also uses O(K^2 d+K^3) arithmetic and no d by d decomposition. Report Gram/dual timings against it and against the dense reference separately. A speed gain over the dense implementation alone is not a new fast-Procrustes result. The thin baseline consumes A rather than a Gram-only packet; reconstructing a factor from a packet is a separate transport question.

### All-rank canonical variant

This is a different specified optimum in nonunique cases. Compute the whitened sender Gram G and whitened receiver rows Y using sample-space matrices. Take the compact nonzero SVD of C=G^(1/2)Y. If its rank is q and rank(G)=r, form

    X_parallel = G^(1/2) U_q V_q^T
    D = G - X_parallel X_parallel^T
    X = X_parallel + F W^T, with F F^T = D.

Fix F by pivoted Cholesky of D: choose the largest remaining diagonal, break exact ties by the smallest sample index, take a positive pivot square root, and use the standard residual-column update. Take exactly r-q columns; inconsistent rank or residual checks produce an unsupported status rather than an extra fitted correction.

Fix W by projecting receiver coordinate axes, in increasing index order, away from the span of V_q; orthogonalize twice against accepted columns, normalize, and make the largest-magnitude coordinate positive, breaking ties by smallest index. Examine only the first r coordinate axes. In exact arithmetic these suffice for r-q columns; failure of the numerical acceptance threshold produces an unsupported status. No d by d projector is formed.

The variant must preserve X X^T=G and attain the nuclear-norm trace objective. When q=r, its product is unique and must agree with the dense result. When q<r, output differences from the dense SVD are recorded, not treated as a failed equality theorem.

For dewhitening, put G_R=R_c R_c^T, beta=sqrt(max(lambda,epsilon)), and f(t)=sqrt(max(t/K+lambda,epsilon)). Use

    X sqrt(Cov_R) = beta X + (X R_c^T) h(G_R) R_c,
    h(t) = (f(t)-beta)/t for t>0.

Evaluate h by rationalizing the numerator. Below the floor its numerator is exactly zero. At zero use 0 if lambda<epsilon and 1/(2K sqrt(lambda)) otherwise. For the ideal track use 1/[K(sqrt(t/K+lambda)+sqrt(lambda))]. Do not use the ideal expression below the upstream eigenvalue floor.

Fast branches use O(K^2 d + K^3) arithmetic and O(Kd+K^2) working arrays. This excludes the separate vocabulary search, which remains O(K V d). Tests inspect eigendecomposition/SVD argument sizes and source code; array dimensions are not reported as measured peak memory.

## 2. Postprocessing and early returns

For all aligned outputs, calibrate each row to the mean Euclidean norm of the entire vocabulary table, dividing by max(row_norm,1e-6). Cosine normalization uses max(row_norm,1e-12), matching the documented default of torch.nn.functional.normalize. Find the largest cosine against every vocabulary row, choosing the first index on an exact tie. Record both alpha=0 and alpha=0.3 outputs, where the latter blends 70% calibrated output with 30% of the nearest vocabulary row.

Report pre-calibration error, calibrated error, nearest-neighbor IDs, top-two cosine margins, and final blended error separately. For float64 unique-product mathematical-equivalence comparisons only, discrete ID equality is a required stable-choice check only when the dense top-two margin exceeds 1e-6 and both cosine normalizations have norms above their clamps. Float32 packet rounding is evaluated separately and its changed IDs are descriptive quantization outcomes, not subjected to an exact-output claim. Ties and smaller margins remain in the results as sensitive cases. A mismatch is never hidden by reporting only mean vector error. Canonical-versus-dense snap equality is not required when q is smaller than sender rank, because those prefixes can legitimately differ.

The author path returns raw hidden states when token IDs are missing/empty or fewer than two rows remain. Those cases bypass both alignment and calibration here too. A centered Gram packet cannot recover that raw state; the encoder must use the explicit raw-output fallback. Inputs outside 2 <= K < d also use a guarded fallback. The K=d rejection is conservative; the K=d+1 fixture tests a boundary where the proper-rotation restriction can change the optimum.

## 3. Fixed generation and numerical policy

Runtime: Python 3.11.18, NumPy 2.4.6, float64 arrays unless a packet-rounding case explicitly says float32. Set OpenBLAS, MKL, and OMP thread counts to one before NumPy import. Record NumPy/BLAS configuration, CPU, OS, source hashes, and actual thread settings.

Random fixtures use NumPy default_rng with SeedSequence([seed, suite_id, K, d, case_id]). The primary seeds are 101, 202, and 303. Suite IDs and case IDs are fixed in source before execution. QR factors are made deterministic by forcing positive diagonal entries of R. A matrix with prescribed centered singular values is built from U, an independent sample-space orthogonal factor, that spectrum, and a thin feature-space orthogonal factor. Means are independent Gaussian vectors scaled by 0.2.

For a nonnegative spectrum, use spectral scale max(1,largest_eigenvalue). The declared numerical rank threshold is 1e-10 times this scale. Eigenvalues whose scaled magnitude lies in [1e-12,1e-8] are an ambiguity/stress band and disqualify exact-equivalence claims. Negative eigenvalues below -1e-10 times scale are rejected. Smaller roundoff negatives may be clipped to zero, with the clipped mass recorded. Apply rank checks to centered sample Grams and C C^T, not to regularized feature covariances, which are full rank by construction.

Fixtures also retain their construction ranks. A tiny known nonzero singular value truncated by numerical policy is recorded as an approximation, not relabeled an exact mathematical zero. A failed inverse, inconsistent residual factor, nonfinite value, or ambiguous rank is recorded with its reason. No post-outcome tolerance relaxation, new seed selection, or automatic retry is permitted.

For well-conditioned supported cases, matrix equality passes only if

    Frobenius(error) <= 1e-10 + 1e-8 * Frobenius(reference).

Use the same absolute/relative tolerances for Gram feasibility, orthogonality, factor residual, dewhitening identity, and scalar optimal-objective checks, with the relevant reference norm/value. Report absolute and relative residuals even when a check passes. The tolerances are chosen before calibration runs; failures remain results.

## 4. Fixed fixture suites

### A. Primary full-rank equality: 120 cases

Take K in {2,4,8,16}, d=64, all three primary seeds, and two paired source/receiver smallest-to-largest singular ratios: (1,1) and (0.01,0.03). Singular values are sqrt(K) times a geometric progression from 1 to the chosen ratio, with independently drawn sample and feature bases. For a one-dimensional centered space the spectrum has its single value sqrt(K).

For each of these 24 matrix pairs use lambda in {0,1e-8,1e-6,1e-3,0.1}, epsilon=1e-6. Reference rows are vocabulary entries 0 through K-1; append 64 independently drawn standard-normal vocabulary rows. All four implementations and both alpha values are evaluated. Exact dense/thin-SVD/dual/canonical agreement is expected only within the declared tolerance and stable-anchor rule.

### B. Exact rank boundaries: 27 cases

Use K=8, d=32, seed 101, epsilon=1e-6, and lambda in {1e-8,1e-3,0.1}. Construct the following nine geometries with nonzero singular ratios at least 0.1:

1. Zero sender; full centered-rank receiver.
2. Full centered-rank sender; zero centered receiver.
3. Both centered matrices zero.
4. Sender rank 3; full centered-rank receiver.
5. Full centered-rank sender; receiver rank 3.
6. Sender rank 3 and receiver rank 2 with nested sample subspaces.
7. Sender rank 3 and receiver rank 2 with disjoint sample subspaces.
8. Full centered-rank sender with receiver token IDs [0,0,1,1,2,2,3,3].
9. Full centered-rank sender with distinct IDs whose embedding rows have centered rank 2.

Here "zero centered receiver" may have a nonzero common row mean. Record source rank, receiver rank, cross rank q, uniqueness, proper-rotation feasibility, all canonical residuals, and any dense/canonical output difference. Both full-rank methods, dual and thin-SVD, must reject deficient inputs. The canonical arm must satisfy Gram/objective checks and match the dense product whenever q equals sender rank.

### C. Near-rank stress: 8 cases

Use K=8, d=32, seed 101, lambda=1e-3, epsilon=1e-6. Make either the sender or receiver smallest singular ratio one of {1e-4,1e-5,1e-6,1e-8}, leaving the other ratio at 0.1. These cases are always labeled stress because their known small directions test the numerical policy. Report inferred ranks, ambiguity decisions, residuals, and packet fallbacks; do not count them as a successful exact-equivalence demonstration merely because truncation lowers error.

### D. Structural and control fixtures

Keep each of the following fixed, with exact arrays or generation rules present in the committed source:

- Full-rank K=8,d=32,lambda=1e-3, all three seeds: compare the baseline with a common sender-row translation and with a sender right-orthogonal transformation. The target rows and vocabulary stay fixed. Unique-product outputs should be unchanged.
- The analytical sign-bit example S_b=[[1,b,0],[-1,-b,0]], b in {-1,+1}, R=[[1,0,0],[-1,0,0]], vocabulary R, lambda in {0,1e-8,1e-3,0.1}, epsilon=1e-6. Verify unchanged decoded IDs, different raw sign bit, equal Gram, the closed-form pre-calibration scalar, and the final collision after calibration/snapping.
- K=2,d=4,lambda=0.1: sender rows are +/-s(e1+e2), s in {0.25,1,4}. With receiver rows +/-e1 and vocabulary {+/-e1,+/-e2}, calibrated outputs should coincide. With receiver rows +/-e1+e2 and vocabulary containing those rows plus +/-e3, calibrated directions should differ between the extreme scales by more than 1e-4. This prevents a universal two-token-collapse claim.
- A nonunique null-completion example: K=3,d=4, S rows e1,e2,-e1-e2, and R has three identical rows e1. Compare S with its signed-permutation rotation into coordinates 3 and 4, lambda=0.1. Record identical sender Grams, potentially different dense completions, and canonical objective/Gram correctness. It is an example of the boundary of author-output reconstruction, not a refutation of an isometry theorem.
- K=1 with two distinct sender vectors but the same token ID, and a K=3 missing-token case: verify exact raw-output fallback.
- Dimension guards (K,d)=(4,4) and (4,3). The latter uses a full-rank centered source and a reflected target so that a spare feature direction is unavailable; it must not pass the fast guarantee.
- Standalone postprocessing controls with a zero row and duplicate vocabulary rows: verify finite clamp behavior and the first-index argmax rule.

### E. Ideal limit versus floor: 56 cases

Use K=8,d=32,seed 202 and four geometries: full/full centered rank, full sender with rank-3 receiver, full sender with zero centered receiver, and disjoint rank-1 sender/receiver sample subspaces. Nonzero singular ratios are 0.1.

Use lambda in {0.1,0.01,0.001,0.0001,0.00001,0.000001} for epsilon=0. For epsilon=1e-6 use the same six values plus 1e-8 and 0. Record error to raw R before calibration, error after calibration, target-row-span residual, and full per-lambda outputs.

For the ideal full-sender-rank cases check the proved unmatched-component bound sqrt(K(K-1)lambda), with the fixed numerical tolerance. Finite-grid convergence rates are descriptive; do not turn visual decrease into a general proof. The deficient sender is a negative boundary control. The floored track must not be summarized as the literal unfloored lambda-to-zero theorem.

### F. Synthetic repeated-token applicability: 2,304 sequences

Take K in {2,8,16}, vocabulary size in {32,256}, each primary seed, and either uniform sampling or probabilities proportional to (index+1)^(-1.2). Draw 64 token sequences per cell, with d=64 Gaussian embedding tables fixed by the cell seed. Record duplicate count, centered target rank, ambiguity status, and guarded full-rank eligibility for every sequence.

These are synthetic sampling distributions, not natural-language or model measurements. They quantify the scope of the guarded formula and the need for the distinct canonical/fallback paths. They do not establish receiver quality.

## 5. Bounded packet experiment

Implement one small explicit binary format, not a general networking stack. Every packet has a 56-byte little-endian header with struct layout <4sBBBBIIQ32s: magic, version, mode, floating-point code, flags, K, d, payload length, and SHA256 of the shared session manifest.

A Gram payload contains K uint32 token IDs followed by the lower triangle of A A^T, where A=U^T S. Serialize in row order, first as float64 and separately as float32. The receiver reconstructs the symmetric Gram, obtains R from the shared vocabulary, and executes either the guarded dual or the specified canonical decoder. The sender mean and orientation are intentionally absent. Float32 Gram transport is a lossy experiment, not exact compression.

Run round trips for all primary and exact-rank cases and the K=1/dimension guards. Before emitting a Gram packet, the sender performs the rounded-Gram decoder preflight using the shared session/table. Charge that preflight to packet encoding, including any raw-output fallback computation. The receiver receives only packet bytes and the shared session/table; it never accesses S. A decoder rejection after receipt is an explicit failure, not a hidden receiver-side raw fallback. Retain decoding decisions, PSD/rank failures, output errors, nearest-neighbor changes, and exact serialized byte counts. Ambiguity, PSD, and unsupported-dimension failures use a tagged raw prefix payload at the requested transport precision. For either fast variant, an unsupported case falls back to the dense author-style reference output. This is an explicitly labeled composite fallback mode, not a claim that a nonunique canonical prefix equals that output. Its sender-side dense computation is not free and must be recorded separately. For early-return cases the payload is the original raw hidden state.

Compare against:

- A minimally framed direct full prefix of the chosen algorithm's already postprocessed output: 56 header bytes plus K*d floating-point values. It does not need token IDs or a vocabulary table for reconstruction.
- A token-only packet: 56 header bytes plus K uint32 IDs. Reconstructing an embedding prefix from these IDs requires the receiver's embedding-table lookup, with its cold table requirement recorded. This lacks the extra continuous statistic and is not an equivalent-output baseline.
- Explicit raw-prefix fallback bytes and frequency for each variant/precision.

For each float32 Gram comparison, also serialize and decode the matched direct full prefix as float32. Record both methods' error against their chosen float64 final output, and nearest-vocabulary ID changes computed on those final outputs by the evaluation harness. The direct-prefix recipient need not perform this diagnostic lookup or store the vocabulary. Keep these final-output diagnostics separate from the Gram decoder's internal pre-blend snap IDs. This prevents attributing ordinary float32 rounding to Gram transport alone.

For a Gram packet the warm payload total is 56+4K+b*K*(K-1)/2 bytes, where b is 8 or 4. The session manifest contains the vocabulary hash/shape/dtype, algorithm and rank-policy version, lambda, epsilon, alpha, normalization constants, and reference source identity. Charge its actual canonical-JSON UTF-8 bytes and the full synthetic vocabulary table bytes separately as cold setup. The vocabulary table is an extra decoder requirement: a recipient of an already computed dense prefix need not receive or store this table, so cold table bytes are not automatically charged equally to that baseline. Warm Gram runs assume this exact session/table is already shared. Record the actual float64/float32 transport error instead of describing either as a lossless reproduction of upstream float32/BF16 output.

## 6. Timing and execution budget

Timing uses full-rank fixtures with K in {8,32,64}, d in {128,384,768,1024}, seed 404, smallest singular ratios (0.1,0.1), lambda=1e-3, epsilon=1e-6: 12 sizes. Before timing each fixture, run its required unique-product equality and Gram/objective checks. If any required check fails, retain the failure and mark that fixture's timing skipped; it cannot count as an accelerated valid case.

Inputs and vocabulary setup are outside the timed core. Warm each core once in the fixed order dense, thin-SVD, dual, canonical. Take eight measured repetitions. For repetition j, rotate this four-arm order by j modulo 4, so every arm is first twice. Use perf_counter_ns and retain every duration. No case is dropped as an outlier. Report medians, ranges, and paired speed ratios per size against both the dense and thin-SVD baselines, never a timing significance claim from eight repeats on one host.

Measure postprocessing, packet encoding, and packet decoding separately with the same one-warmup/eight-repetition rule. Include receiver reconstruction cost in the packet account. The full vocabulary for timing is the K reference rows plus 256 Gaussian rows. A faster alignment core is not a measured end-to-end LLM speedup.

A single numerical child process has a 300-second wall cap; the Actions job cap is 10 minutes including dependencies and tests. No paid service, GPU, model weights, or repeated empirical run is used. If the cap or an exception stops the study, retain the completed cases and classify the run as incomplete. Correctness failures are recorded and remaining independent cases continue if execution is safe; the final status cannot be "verified" with a failed required check.

## 7. Outputs, decisions, and publication

Append per-case JSONL as each case finishes, including suite/case identity, seed, input SHA256s, construction and numerical ranks, guards, array-operation shapes, residuals, snap margins/IDs, packet counts, and failures. Retain timing JSONL without aggregation-only replacement. Save a bounded NPZ retaining S, R, and every available pre-calibration output for all algebraic fixtures in suites A through E, including the structural controls. Also retain the largest timing fixture's inputs, vocabulary, and available pre-calibration outputs. Other timing fixtures retain deterministic generation parameters, input hashes, and all raw durations. Duplicate arrays may be stored once with a manifest mapping every case to its array keys. Do not retain d by d transient covariance/rotation matrices. Write and flush the algebraic NPZ and its manifest before timing starts, then atomically add the largest timing fixture so a timing timeout cannot erase completed mathematical evidence.

Write results.json and a readable report with separate summaries for: unique-product equivalence; canonical feasibility/optimality; stress/unsupported cases; structural collisions/boundaries; ideal/floored limits; applicability; packet sizes/error/fallbacks; and timing. Expose the fixed study as python -m latent_bridge.alignment_numerical --output checkpoints/alignment-numerical. The output directory contains cases.jsonl, timings.jsonl, arrays.npz, arrays_manifest.json, results.json, report.md, and source/runtime manifests. Flush JSONL after each completed record. Print the bounded summary JSON between fixed log sentinels for recovery. Preserve the full files in the Actions artifact, including partial evidence on failure. Record source/design/runtime hashes and a run identifier.

Numerical support requires all declared well-conditioned unique cases to meet their fixed comparisons, and all supported exact-rank canonical cases to meet Gram/objective checks. Any failed case is visible. Timing improvement is an additional observation, not a substitute for equivalence. A canonical nonunique optimum needs later receiver evaluation before claiming interchangeable behavior. Successful compression here establishes a sufficient representation for this mathematical interface at recorded precision and cost; it does not establish a new semantic language or superiority to token-only communication.

No design changes or parameter search follow outcome inspection within this run. A necessary implementation correction is a separately identified version, with previous failures retained.

## Prior-art boundary

This work uses established matrix geometry, including [Schönemann's Procrustes solution](https://engineering.purdue.edu/~bethel/schonemann_orth_proc.pdf), [Bures–Wasserstein matrix formulas](https://arxiv.org/abs/1712.01504), and [Gram invariants for orthogonal orbits](https://web.math.princeton.edu/~amits/publications/1907.01145.pdf). [FLoRG](https://arxiv.org/html/2602.17095v1) already combines Gram parameterization, Procrustes alignment, and communication concerns in federated fine-tuning. These precedents rule out a generic claim that Gram matrices or Procrustes communication are new. The limited contribution under test is an explicit, guarded computational and transport specialization for this alignment interface, plus a separately specified canonical completion and documented failure boundaries.
