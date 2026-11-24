# Coefficient packet study v2: fixed design before outcomes

Status: preregistered before implementation or outcomes. No v2 implementation, timing, numerical outcome, or model run has been executed. This study follows the completed alignment study at source commit [7aee330f7799751a6a9f4b2a8decb612abe9c266](https://github.com/ConfusedDuckCollectibles/Research3/tree/7aee330f7799751a6a9f4b2a8decb612abe9c266). It preserves that study's source, fixtures, results, and negative packet-cost finding. Original fixtures may appear only in explicitly labeled regression tests.

## 1. Question and scope

Does transmitting the already-computed symmetric Bures coefficient reduce **actual network-inclusive completion time**, while preserving the supported aligned prefix, relative to both strong direct-prefix paths and an independently improved Gram codec?

This is an engineering study of a CPU implementation and controlled loopback transport. It contains no pretrained-model inference, task-quality experiment, new-theory claim, or claim about production networks. The primary population is full centered row rank with K<d. Rank-deficient and numerically ambiguous cases exercise a declared raw-prefix fallback separately.

There is no tuning phase, validation-selected hyperparameter, or adaptive policy. The parameters, fixtures, schedules, comparisons, and pass criteria below are fixed before any v2 outcome. The published design precedes code publication; reviewed code precedes the one authorized measurement run. Any necessary amendment must be published before outcomes.

## 2. Algebra and fixed reconstruction

Let r=K-1 and H be the existing deterministic Helmert basis, H^T H=I and H^T 1=0. Set A=H^T S, B=H^T R, GA=AA^T, GB=BB^T, and mu=mean(R). With the same spectral covariance floor as the frozen core,

f(t)=t/max(t/K+lambda,epsilon), X=f(GA), Y=f(GB),

M=X^(1/2) (X^(1/2) Y X^(1/2))^(-1/2) X^(1/2).

For full-row-rank A and B, M is symmetric positive definite, MYM=X, and the unique aligned pre-calibration prefix is

Z=(H @ M) @ B + mu.

Matching floored whitening/dewhitening cancels exactly in this identity. K<d supplies an unused ambient direction for the proper-rotation correction. The mathematical derivation and limitations are in [alignment_geometry.md](alignment_geometry.md).

The grouping **(H @ M) @ B + mu** is fixed for this study. It matches the existing left-associated dual expansion and avoids the extra r-by-r times r-by-d product of H @ (M @ B). Direct encoding reuses B, mu, H, and decompositions already computed within its call. The coefficient receiver derives B and mu from the actual token IDs and table; this additional receiver work is charged. Neither endpoint caches per-message GA, GB, M, B, prefixes, or packets across trials. Alternative grouping through H M H^T is out of scope.

Constants remain lambda=1e-3, epsilon=1e-6, alpha=0.3. Source/receiver/cross rank threshold 1e-10 and ambiguity band [1e-12,1e-8] retain the frozen core's normalization by max(1,largest eigenvalue). A fixed covariance floor is not interpreted as an unfloored zero-regularization limit.

## 3. Six arms and fair shared work

| Arm | Timed sender work | Timed receiver work | Role |
|---|---|---|---|
| coefficient | Form M and original rank guards; symmetrize; round/serialize actual M; small-matrix guards below | Validate actual packet/M; reconstruct Z; one postprocess | Candidate |
| dual_direct | Best existing dual computation, reusing its B/mu/intermediates; one postprocess; direct encode | Raw decode | Strong primary comparator |
| thin_direct | Existing thin-SVD computation; one postprocess; direct encode | Raw decode | Strong primary comparator |
| gram_guard | Form rounded GA; two-eigensolve spectral preflight below; serialize GA | Existing dual solve from actual GA; reconstruct; one postprocess | Strong primary comparator isolating representation benefit |
| gram_original | Frozen dual Gram codec with full sender decoder/postprocessing preflight | Frozen decode and postprocess | Historical diagnostic comparator |
| token_lookup | Encode uint32 IDs | Embedding lookup producing R | Non-equivalent reference |

The token arm does not reproduce the aligned prefix. Its receiver needs the table because this arm explicitly produces embeddings. Plain ID transport is a separately stated byte floor, not an equivalent reconstruction method and not another measured arm.

All v2 equivalent routes use the same postprocessing implementation and may cache vocabulary row norms, normalized vocabulary rows, and target norm once per session, at whichever endpoint performs postprocessing. There is still one real nearest-vocabulary search per reconstruction. Such setup is outside warm-message timing but inside cold timing and is recorded. The coefficient sender has no per-message full-vocabulary scan, nearest-neighbor search, prefix expansion, or full decoder preflight. The optimized Gram sender also has no such work.

The frozen original Gram API is retained unchanged as a historical diagnostic; its uncached implementation cost cannot establish a coefficient-specific benefit. Primary comparisons include gram_guard and both direct methods. Any factorization or postprocessing optimization introduced for coefficient must be offered to the applicable v2 comparators. The dual baseline must not compute M twice, discard B and recompute it, or incur candidate-only guard work.

For each equivalent arm, an unsupported fast path uses the same frozen dense-reference plus postprocess raw-prefix fallback, charging the failed attempt and fallback. A decoder never reconstructs from private S or silently falls back using unavailable sender state.

## 4. Packet and numerical guard contract

Use the existing 56-byte header (<4sBBBBIIQ32s), with an explicit new coefficient mode. The coefficient payload is K little-endian uint32 IDs followed by r(r+1)/2 lower-triangle scalars. The sender first replaces numerical M by (M+M^T)/2, then serializes. The receiver mirrors that actual triangle. The session digest binds table bytes/order, dimensions, precision conventions, an explicit v2 algorithm/session version and coefficient mode identifier, the deterministic Helmert basis rule, the fixed reconstruction grouping, and postprocessing parameters.

Framed packet size before TCP framing is 56+4K+b*K*(K-1)/2 bytes, exactly the Gram layout's size at scalar width b. Direct/raw size is 56+b*K*d. Float64 is primary; float32 packets are a separate descriptive numerical assay, without transport timing.

Coefficient guards inspect the decoded coefficient Mhat, not only the pre-rounding array:

- Existing source, receiver, and cross support checks during M construction.
- Finite entries; symmetry by the wire construction; successful Cholesky of Mhat.
- Normalized residual ||Mhat Y Mhat-X||F/max(1,||X||F) <= 1e-8 for f64, <=1e-5 for f32.
- Quantization bound delta=||Mhat-Msym||F*sqrt(lambda_max(GB)).
- Scale z=sqrt(tr(Msym GB Msym^T)+K*||mu||^2), calculated at small-matrix scale without expanding Z. Reject nonfinite or materially negative computed squared scale; only a negative value within 64*r*u*max(1,absolute term sum) may be clipped to zero, where u is float64 unit roundoff.
- Require delta <= 1e-10+1e-8*z for f64, or <=1e-7+1e-5*z for f32.
- All prerequisite norm bounds must be finite and an upper bound on pre-prefix Frobenius norm must be below 1e100. This is a numerical-range guard, not a model-derived threshold.

The receiver validates framing/session/IDs/finite values and Cholesky of the decoded coefficient, then checks finite reconstructed/postprocessed output. Both sender and receiver guard costs are counted. SPD alone is only structural validity: the receiver cannot verify MYM=X without the source Gram.

The gram_guard sender must **not solve for M merely to validate a Gram packet**. It diagonalizes the actual rounded GA and GB, applies the same source/target support rules and floor, and obtains spectral extrema xmin,xmax,ymin,ymax of X,Y. It accepts cross support only if

xmin*ymin > (1e-8 + 64*r*u) * max(1,xmax*ymax).

The product bounds enclose the spectrum of X^(1/2) Y X^(1/2) in exact arithmetic. Their floating evaluation is a diagnostic sufficient-condition guard, not an interval-arithmetic certificate. A failed bound takes the declared fallback, even when a full cross solve could have succeeded. The finite-output bound uses ||M||2 <= sqrt(xmax/ymin), so it requires no expansion or vocabulary search. Both sender eigensolves are charged. The actual receiver still executes the normal source/target/cross guarded solve.

The quantization inequality ||Zhat-Z||F <= ||Mhat-M||F*||B||2 is exact for fixed exact matrices. It does not include coefficient-solver or BLAS roundoff. Small prefix error alone does not certify vocabulary-snap identity. No cheap guard is described as a universal final-output certificate or as a lossless cross-host codec.

## 5. Fresh fixed fixture cohorts

Primary dimensions are (K,d)=(12,192),(36,512),(60,960). Seeds are 17011,17027,17041, giving nine fixed primary fixtures.

For each fixture, use NumPy PCG64 SeedSequence([seed,K,d,slot]) with separate integer slots for each random object. Obtain orthogonal factors by reduced QR of independent standard-normal matrices; normalize QR signs so R's diagonal is nonnegative (zero chooses +1). Build

S=H Qs diag(ss) Vs^T + 1 muS,
R=H Qr diag(sr) Vr^T + 1 muR,

where Qs,Qr are r-by-r orthogonal, Vs,Vr are d-by-r orthonormal, ss=geomspace(1,0.2,r), sr=geomspace(1,0.1,r), muS~N(0,(0.2/sqrt(d))^2), and muR~N(0,(0.1/sqrt(d))^2). The vocabulary starts with R, followed by 256 independent Gaussian distractor rows, each scaled to the median norm of R's rows. IDs are 0..K-1. Zero/nonfinite construction events are recorded failures, not resampled.

Freeze slots 1=Qs,2=Vs,3=Qr,4=Vr,5=muS,6=muR,7=distractors. Store every fixture's S, table, IDs, construction metadata and byte hashes. Primary cases are not selected by margins, speed, guard success, or preliminary outcomes.

A separate stress cohort uses K=6,d=24 and seeds19001,19013 with the same construction rules except these eight declared geometries:

1. Sender rank r-1, receiver full.
2. Receiver rank r-1, sender full.
3. Both rank one, independently chosen reduced row directions.
4. Zero centered sender, receiver full.
5. All receiver rows identical (zero centered rank; repeated ID), sender full.
6. Sender minimum singular ratio1e-5, receiver otherwise ordinary.
7. Receiver minimum singular ratio1e-5, sender otherwise ordinary.
8. GA=diag(1,1,1,1,4e-8), GB=diag(4e-8,1,1,1,1), using the constructed independent right bases and means. This is the conservative-cross-bound stress: an actual cross solve can be supported although the extrema bound rejects.

There are sixteen stress fixtures. They receive actual f64/f32 encode/decode tests and complete encoder/decoder duration measurements as descriptive diagnostics, but no network-speed success claim. The token reference remains separately labeled. Existing sign-bit, rounded-Gram, malformed-packet, and bypass tests are regressions only.

## 6. Numerical eligibility and fidelity

Before transport timing, actually encode and decode every primary fixture/arm in f64 using the receiver-only interface. Run f32 primary diagnostics and the stress roundtrips separately. The external checker may compute frozen dense/dual/thin reference outputs; none of those arrays, outcomes, or eligibility labels enter measured encoder decisions or receiver inputs.

Use the original Frobenius comparison rule ||a-b||F <=1e-10+1e-8*||b||F. Check all-row pre-calibration and calibrated output against the appropriate numerical reference where those stages exist. Check internal snap and final output on rows with margin>1e-6, pre-norm>1e-6, query-norm>1e-12, and chosen-vocabulary norm>1e-12. Retain all-row errors, margins, exclusions, and every snap change.

In addition to those margin-scoped mathematical checks, **the practical f64 transport-fidelity gate requires coefficient final output on every primary row to agree numerically with dual_direct, with no internal snap-ID disagreement**, including any unstable primary rows. This is an empirical goal for this fixed cohort, not a theorem about arbitrary near ties. Stress ties/ambiguities remain descriptive. Direct/raw roundtrip equality is bytewise at the actual wire dtype, including signed zeros.

For every f32 primary/stress packet, retain pre-calibration, calibrated and final absolute/relative Frobenius errors, maximum coordinate error, norm changes, snap changes, and fallback status. Compare with direct transmission rounded to the same f32 precision against the same unrounded reference. Equal nearest-token IDs are not a proxy for vector fidelity; large vector errors remain visible even when IDs agree. F32 mathematical/output errors are descriptive and cannot establish the primary fidelity or speed claim.

Every predetermined trial remains in the result inventory. An eligibility failure marks its trials numerically_ineligible and blocks a complete success claim; it does not shrink the planned denominator. Primary coefficient and gram_guard fixtures must use their native f64 route; unexpected fallbacks remain visible and prevent a claim over the complete supported primary grid. Stress fallbacks are expected and audited against their actual dense fallback target.

No replayed successful preflight, prepared packet, precomputed M, reference prefix, or fixture-specific matrix cache is reused inside a measured encoder.

## 7. Actual TCP, clock boundaries, and link profiles

Use real TCP on 127.0.0.1, TCP_NODELAY, one sender and one receiver thread, with no concurrent heavy job. The receiver function accepts only its socket and legitimate context, never S, the fixture object, a reference prefix, or oracle decisions. It reconstructs from the received bytes. Each trial uses fresh transport/endpoints and input objects; there is no cross-trial numerical memoization.

Every transmitted frame has a four-byte unsigned big-endian payload length. Maximum permitted payload is fixed at 16MiB. Receiver output stays in receiver memory for post-timing verification. The reply is a **40-byte ACK payload**: little-endian <4sB3x32s>, magic ACK2, status byte, and SHA256 of canonical little-endian float64 decoded output bytes; an error ACK uses a zero digest. With framing it is44bytes. A failed decode is retained with local error metadata.

The receiver hashes its actual output before sending ACK; this work is timed. The sender validates ACK structure/status and receipt during timing. A coefficient sender cannot derive the expected final-output hash without doing the work being avoided: hash comparison against the retained array and external numerical oracle happens only in the benchmark audit, not as free deployed semantic verification.

Profiles are fixed:

| Profile | Application pacing, each direction | Holdoff before each receiver reply |
|---|---:|---:|
| loopback | Unlimited | 0 |
| moderate | 10,000,000bytes/s | 5ms |
| constrained | 1,000,000bytes/s | 20ms |

These are imposed application service/pacing conditions on physical loopback TCP, not measured WAN RTTs. Pace every frame byte, including length prefixes, with chunks no larger than16KiB. Wait for each chunk's service time **before sending it**, including the final ACK chunk. Receiver reply holdoff occurs before sending each reply. Record requested and observed pacing/holdoff waits separately from wall completion, plus both endpoints' actual frame bytes/counts and stream hashes. Compute stream hashes after timing from retained actual wire bytes; do not add per-byte trace hashing to the timed transport. The actual decoded-output hash needed for the ACK remains timed. No dummy-size payloads.

Warm timing begins before the complete encoder call and ends after receipt/validation of ACK. It includes private-state processing, guards, serialization/framing, sender and receiver pacing, receiver decoding/reconstruction/postprocessing, receiver output hashing, and ACK. Connected socket creation and declared session construction are outside this boundary and reported separately. Warm sessions may preload legitimate shared tables/configuration; direct receivers have only their required dimension/digest context.

Cold timing begins before client socket creation/connect, sender context construction, receiver session transmission/construction, and complete message encoding, and ends at its final ACK. The receiver starts without a table. Send one length-framed setup payload: uint32 little-endian canonical-JSON length, canonical UTF-8 JSON metadata, then declared raw little-endian f64 table bytes if required. Metadata identifies version/kind/dimensions/configuration/digest and exact table length; metadata is capped at4096bytes. Coefficient/Gram/token_lookup supply actual table bytes; direct arms send only required context. Receiver validation/hash/postprocess-cache setup and a fixed setup ACK are charged. The setup ACK uses the same 40-byte <4sB3x32s> layout with magic SET2, status, and SHA256 of the exact received setup payload (zero digest on error). Hashing and sender validation against its actual emitted setup payload occur inside the cold timer. This setup receipt is distinct from the decoded-output ACK2. Then send the actual message and final ACK. No model loading or synthetic table generation is counted as transport setup; those are common fixture preparation outside both timers.

All receiver arrays and wire packets may be deduplicated **after timing** by exact content hash for retention. Hashes never stand in for actual decoding.

## 8. Fixed timing grid, ordering, and budget

Warm:3shapes x3seeds x3profiles x6arms x4repetitions=648trials.

Cold:3shapes xseed17011 xconstrained profile x6arms x4repetitions=72trials.

Total:720network trials. Each cold trial is one genuinely cold message, not an amortized estimate.

For each warm(shape,profile) block, derive one six-arm permutation from PCG64 SeedSequence([23017,K,d,profile_index,0]). Warm repetition j for seed index i uses rotation(2*i+j)mod6. Across the three seeds, each arm occupies every position exactly twice. Cold uses one common permutation from SeedSequence([23017,0,0,2,1]) and rotation(2*shape_index+j)mod6; across its three shapes each arm occupies every position exactly twice. Four repetitions cannot give perfect position balance within an individual case; report the stated block balance. Use a separately fixed PCG64 seed23029 to permute complete six-arm blocks; all arms of a paired block run consecutively. Do not re-randomize after observing timing.

Use distinct warmup seed23041 and one untimed functional warmup per arm and shape under unshaped transport; warmup arrays cannot be primary arrays. All warmup, eligibility work, diagnostics, setup, measurements, checkpointing and finalization count toward the child wall budget.

Planning arithmetic, **not observed performance**:

| Component | Approximate imposed service/holdoff time |
|---|---:|
| Warm paced frame bytes across both shaped profiles, native f64 routes |17.4s |
| Warm receiver holdoffs |5.4s |
| Cold receiver-table transfer (four table-requiring arms), first-seed grid |64.6s |
| Cold data/ACK framing plus capped setup metadata |about5.6s |
| Cold receiver holdoffs (two replies per trial) |2.9s |
| Total declared waiting before scheduler overhead |about96s |

The nine primary fixtures have approximately4.03MB of table data summed over the three shapes for one seed. Numerical work and actual TCP overhead consume the remaining budget. Run one child on a CPU runner with a strict300s wall deadline; socket operations use10s timeouts. No automatic rerun, budget extension, performance-triggered pruning, or changed pacing is allowed. Preserve completed/error rows and mark every unexecuted planned slot if the budget or process fails.

Pin Python3.11.18, NumPy2.4.6 and one BLAS/OpenMP thread, matching the original numerical environment where available. Record actual versions, OS/CPU, BLAS configuration/thread settings, perf_counter resolution, workflow/run IDs, source/design hashes and discrepancies. An unavailable pinned environment requires a disclosed pre-outcome amendment.

## 9. Fixed analysis and engineering criteria

Report actual wall time as primary. Phase times, planned pacing, bytes, and ratios explain it; a sum of independently timed medians is not substituted for measured completion.

For each fixture/profile, report four paired coefficient/comparator ratios and their median. Aggregate case medians geometrically with equal weight per fixed fixture/profile. Compute each direct comparison from its own paired observations; never divide two separately aggregated ratios.

Primary performance population:all eighteen warm fixture/shaped-profile cases. The strongest retrospective reference chooses, separately for each case, the smallest four-run median among dual_direct,thin_direct,gram_guard. This is explicitly a **per-case hindsight envelope**, not a deployable selector. Calculate paired coefficient ratios against that chosen arm. Also report all three direct aggregate comparisons and the original-Gram diagnostic.

A complete v2 engineering success requires:

1. Every planned primary f64 route is eligible; all720network slots complete correctly; byte/frame/hash accounting reconciles; every required numerical/raw/fallback check passes.
2. On the eighteen shaped warm cases, coefficient/hindsight-reference geometric mean <=0.97, and coefficient geometric mean <=0.97 against each of dual_direct,thin_direct,gram_guard separately.
3. On unshaped warm cases, coefficient/hindsight-reference geometric mean <=1.10; worst per-case paired-median coefficient/hindsight-reference ratio across all warm cases <=1.25.
4. The all-row practical primary coefficient-versus-dual final-output and snap gate in section6 passes.

These are descriptive engineering thresholds on this frozen synthetic cohort, not confidence bounds or universal speed claims. Report all failures/negative results even if the full success conjunction fails. Token lookup is excluded from equivalent-method winner counts and speed gates.

Report cold first-message comparisons separately, including table/setup bytes and construction times. A warm success does not imply a cold success. Any break-even reuse count inferred from cold and warm measurements is labeled a derived model, with its assumptions; it is not an observed repeated-session experiment or an additional success gate. Do not hide cold regressions in the warm aggregate.

## 10. Retention, independent audit, and prior art

Retain the exact planned grid/order; per-trial outcomes/timestamps/phase times/waits; endpoint frame lengths/byte totals/stream hashes; actual packets; decoded arrays or exact content-deduplicated arrays with complete trial mappings; fixture inputs/tables/IDs; numerical/fallback diagnostics; runtime/source manifests; stdout/stderr and partial status. Hash every retained file. Do not retain only successful trials or only summaries.

An independent read-only audit can reconstruct wire sizes/frames, verify ACK hashes from retained actual outputs, replay numerical reconstruction, and recompute comparisons/gates. Audit computations are not new timing trials. CI must cover malformed frames, exact direct bytes, receiver-only input boundaries, roundoff guards, the conservative-cross-bound false fallback, pacing before final sends, setup failures/timeouts, partial accounting, fixed-grid completeness, and no oracle leakage into encoder decisions. Freeze reviewed source before execution; publish raw outcomes before interpretation.

Relevant established work:

- [Bhatia, Jain and Lim, Bures-Wasserstein distance](https://arxiv.org/html/1712.01504v1): symmetric optimal transport map and covariance identity.
- [StateBridge](https://arxiv.org/html/2608.13317v1): the alignment/calibration/anchoring pipeline being represented.
- [FLoRG](https://arxiv.org/html/2602.17095v1): low-rank communication, Gram aggregation and Procrustes alignment.
- [Palu](https://arxiv.org/html/2407.21118v2): compact intermediate states, shared reconstruction matrices, and reconstruction/transfer cost tradeoffs.

The study tests a particular packet placement and guard strategy. It does not establish novel Bures theory, a new general-purpose AI language, preservation of arbitrary source information, downstream model accuracy, or cross-host bit-exact losslessness.
