# Alignment geometry, sufficient statistics, and a constructive variant

## Status and scope

This is an analytical note for a proposed numerical study. The derivations have
received an independent algebra review, but numerical verification is pending.
This note contains no measured runtime, model inference, task-accuracy result,
or empirical compression result.

The subject is the alignment interface in [StateBridge v1, Section 3.3 and
Appendices A.2–A.3](https://arxiv.org/html/2608.13317v1), together with the
[pinned authors' implementation](https://github.com/YanwenPneg/StateBridge/blob/3f6bf5442c6e8848555a6132516e6d36f35444fb/methods/state_bridge.py).
The upstream commit is 3f6bf5442c6e8848555a6132516e6d36f35444fb.

The paper's ridge span proposition and its whitened-rotation isometry
proposition are both correct. The observations below concern what follows from
those propositions, the information discarded by a message-dependent
alignment, and how to compute an aligned prefix through smaller matrices.
They do not refute the paper's experiments or establish a new general
communication method.

There are two distinct implementation claims:

1. When the relevant aligned product is unique, a dual formula computes the
   same mathematical prefix as orthogonal Procrustes.
2. When it is not unique, a deterministic completion constructs a specified
   optimal prefix. This is a **canonical variant**, which can differ from the
   authors' SVD completion and therefore needs separate receiver evaluation.

Neither claim implies bitwise equivalence between floating-point programs.

## 1. Conventions and the exact interface

Use row matrices S,R in R^{K×d}, with 2 ≤ K < d. S contains sender states;
R contains reference token embeddings. Unless explicitly stated otherwise,
R is fixed when comparing different S.

The following LaTeX notation is used:

- P = I_K − 11ᵀ/K is the centering projector.
- S_c = PS and R_c = PR.
- μ_R is the row mean of R.
- O(d) contains all real orthogonal matrices, including reflections.
- Row spans refer to subspaces of the d-dimensional feature space.

The paper's pre-calibration interface is

$$
\Sigma_S=S_c^\top S_c/K+\lambda I_d,\qquad
\Sigma_R=R_c^\top R_c/K+\lambda I_d,\qquad \lambda>0,
$$

$$
S_w=S_c\Sigma_S^{-1/2},\qquad R_w=R_c\Sigma_R^{-1/2},
$$

$$
Q_\lambda\in\arg\min_{Q\in O(d)}\|S_wQ-R_w\|_F^2,
\qquad
Z_\lambda=S_wQ_\lambda\Sigma_R^{1/2}+\mathbf1\mu_R^\top.
$$

The subsequent operations normalize each row to the average vocabulary
embedding norm and blend it with its nearest vocabulary embedding under
cosine similarity. The exact published definitions are in
[Equations 3–9](https://arxiv.org/html/2608.13317v1#S3.SS3).

The pinned code additionally:

- casts alignment inputs to float32;
- flattens batch and sequence axes before centering;
- adds regularization, then floors covariance eigenvalues at 10^{-6};
- forces determinant +1 by flipping the last SVD column when necessary;
- floors row norms at 10^{-6} during norm calibration;
- performs vocabulary argmax and then casts the result to model dtype.

Thus K in the formulas must mean the **total number of aligned rows** in that
call. A per-message interpretation assumes batch size one. The code bypasses
alignment for fewer than two rows; this note does not apply that bypass.

## 2. Full-row-rank Procrustes has a unique aligned product

### Proposition 1

Let A,B be K×d matrices of full row rank, with K<d. Define

$$
G=AA^\top>0,\qquad H=BB^\top>0.
$$

For the problem

$$
\min_{Q\in O(d)}\|AQ-B\|_F^2,
$$

Q need not be unique, but the product X=AQ is unique. It equals

$$
X=M(G,H)B,
$$

where

$$
M(G,H)
=
G^{1/2}
\left(G^{1/2}HG^{1/2}\right)^{-1/2}
G^{1/2}.
\tag{1}
$$

M is positive definite and satisfies MHM=G. In particular,

$$
\operatorname{rowspan}(X)=\operatorname{rowspan}(B).
\tag{2}
$$

### Proof

Right-orthogonal multiplication preserves row Gram matrices. Conversely,
any K×d matrix X satisfying XXᵀ=G can be obtained as AQ: the isometry
between the two row configurations extends to an orthogonal transformation
of the ambient feature space.

Write X=G^{1/2}W with WWᵀ=I_K. The squared objective has fixed first two
terms, so minimizing it is equivalent to maximizing

$$
\langle W,G^{1/2}B\rangle_F.
$$

The full-row-rank matrix C=G^{1/2}B has a unique row polar factor,

$$
W_*=(CC^\top)^{-1/2}C.
$$

Substitution gives Equation (1). It follows directly that MHM=G. Since M
is invertible, multiplication on the left preserves B's row span. The
nonuniqueness of Q concerns only how the transformation is extended beyond
the observed row span, and does not change AQ. ∎

This is standard polar/Procrustes geometry, closely related to the
positive-definite geometric mean and Bures–Wasserstein formulas
[Schönemann, 1966](https://engineering.purdue.edu/~bethel/schonemann_orth_proc.pdf);
[Bhatia, Jain and Lim, 2017](https://arxiv.org/html/1712.01504v1).

### Consequence: orientation can be discarded

For every O∈O(d),

$$
(AO)(AO)^\top=AA^\top.
$$

Therefore, with B fixed, independently aligning A and AO produces the
same unique aligned product. This statement concerns a newly fitted
alignment for each message. A single fixed invertible Q applied to every
message would be a different situation.

Preserving each message's internal Gram matrix does not establish an
injective map across messages. An entire orthogonal orbit can map to the
same prefix. Centering also discards the sender's common row offset.

The qualification “with B fixed” matters: rotating hidden states may change
decoded tokens in an actual model, in which case its reference matrix may
also change. Section 6 gives a construction in which the decoded tokens
remain identical.

### Relation to the ridge span statement

Appendix A.2 correctly derives a ridge output of the form H_γR. Equation
(2) shows that full-row-rank Procrustes output can also lie entirely in the
reference span. Span confinement alone therefore does not distinguish the
two procedures in this regime.

Their coefficients and geometrical constraints differ. Continuous
coefficients within a token-defined span can still carry information beyond
token identity. Consequently, span confinement alone is also not a proof
that either method conveys only text. Appendix A.3's narrower isometry
claim remains valid.

## 3. Exact dual formula for the centered, whitened interface

Let r=K−1, and choose a fixed K×r matrix U satisfying

$$
U^\top U=I_r,\qquad UU^\top=P.
$$

Define reduced matrices

$$
A=U^\top S_c,\qquad B=U^\top R_c,
\qquad G_A=AA^\top,\quad G_B=BB^\top.
$$

Assume A and B both have full row rank r. Their regularized whitened
Gram matrices are

$$
F_\lambda(G)
=
K\,G(G+K\lambda I_r)^{-1}.
\tag{3}
$$

This follows from the push-through identity or directly from a thin SVD.
Set

$$
G_{Aw}=F_\lambda(G_A),\qquad G_{Bw}=F_\lambda(G_B),
\qquad M_\lambda=M(G_{Aw},G_{Bw}).
$$

Applying Proposition 1 to the reduced whitened matrices gives

$$
\boxed{
Z_\lambda
=
U M_\lambda B+\mathbf1\mu_R^\top
=
U M_\lambda U^\top R_c+\mathbf1\mu_R^\top .
}
\tag{4}
$$

The final cancellation is exact: the receiver whitening inverse square root
and its dewhitening square root multiply to the identity.

Equation (4) uses the sender only through its centered row Gram matrix.
Its reference quantities can be reconstructed from the token identities
and the shared embedding table. No d×d covariance, rotation, or SVD matrix
is required.

Because M_λ is invertible on the centered subspace and the mean direction
is retained, Equation (4) is an invertible left mixing of R. The
pre-calibration prefix has the same row span as R. Rowwise norm scaling
does not add directions. Vocabulary anchoring can add an embedding outside
that span, but it is still a deterministic function of the same aligned
prefix and vocabulary; it cannot recover an already discarded sender
orientation.

### Proper rotations

The upstream code restricts to determinant +1. Any same-Gram row map can
first be extended to Q∈O(d). If det(Q)=−1, compose it with a reflection in
an unused sender direction. That reflection leaves the observed sender
rows unchanged and changes the determinant.

A spare direction is guaranteed by r<d here. Thus the attainable aligned
products and optimum are unchanged by the proper-rotation constraint.
This argument must not be extended to arbitrary uncentered, full-rank
K=d problems.

## 4. The weak-regularization limit

### Full centered rank on both sides

If G_A,G_B are positive definite, then

$$
F_\lambda(G_A)\to K I_r,\qquad
F_\lambda(G_B)\to K I_r,\qquad
M_\lambda\to I_r
\quad\text{as }\lambda\downarrow0.
$$

Consequently,

$$
Z_\lambda\longrightarrow R.
\tag{5}
$$

All full-rank centered messages approach the same whitened row Gram
matrix. For fixed positive spectral gaps, the first-order expansion is

$$
M_\lambda
=
I_r+\frac{K\lambda}{2}
\left(G_B^{-1}-G_A^{-1}\right)+O(\lambda^2).
\tag{6}
$$

This specifies a regime in which the pre-calibration interface approaches
a function of the reference tokens. It is not an assertion that the
regularization used in the authors' experiments lies in that regime.

### Rank-deficient reference, full centered sender rank

Retain full row rank of A, but let rank(B)=t≤r. The finite-λ product may
now be nonunique. Nevertheless, **every exact optimal dewhitened product
still converges to R in the unfloored equations**.

To see this, write

$$
A_w=A(A^\top A/K+\lambda I_d)^{-1/2},\quad
B_w=B(B^\top B/K+\lambda I_d)^{-1/2},\quad
Y_\lambda=A_wQ_\lambda.
$$

Let Π_B project onto B's row span. The target-space component
Y_λΠ_B is the uniquely determined partial polar alignment. Since
A_wA_wᵀ→KI_r, this component approaches the limiting whitened B.

On the orthogonal complement of B's row span, the receiver covariance
square root is exactly √λ I. Hence

$$
\left\|
Y_\lambda(I-\Pi_B)\Sigma_R^{1/2}
\right\|_F
=
\sqrt{\lambda}\,
\|Y_\lambda(I-\Pi_B)\|_F
\leq \sqrt{Kr\lambda},
\tag{7}
$$

because tr(A_wA_wᵀ)≤Kr. Dewhitening the active component approaches B,
and Equation (7) makes every unmatched component vanish, uniformly over
the choice of an exact optimal completion. Restoring the mean gives
Equation (5).

Full centered sender rank is a substantive assumption. In general,
right multiplication cannot create sample-space directions absent from
the columns of S_c. For example, if S_c and R_c have different orthogonal
one-dimensional column spaces inside 1-perp, convergence to R_c is
impossible. A necessary condition is that the receiver's centered column
space be contained in the sender's.

Repeated reference tokens are an exact source of rank deficiency. If
there are m distinct embeddings among K rows, rank(R_c)≤m−1. Distinct
token IDs do not by themselves guarantee good numerical conditioning.

### What calibration does to the limit

Suppose every reference row r_i is nonzero. Let n̄ be the mean vocabulary
embedding norm. Before anchoring, row calibration approaches

$$
\hat z_i\to \bar n\,r_i/\|r_i\|_2.
$$

If the locally stable nearest-vocabulary choice is the original reference
embedding, the final anchored limit is

$$
\bar z_i\to
\left[(1-\alpha)\frac{\bar n}{\|r_i\|_2}+\alpha\right]r_i.
\tag{8}
$$

This need not equal raw R. It is, however, determined entirely by R and
the fixed vocabulary. Ties, zero rows, and floating-point argmax behavior
need separate treatment.

## 5. A constructive canonical optimum at every rank

This section specifies a mathematical variant, rather than identifying
the upstream SVD's unspecified null-space completion.

Let

$$
G=S_wS_w^\top,\qquad Y=R_w,\qquad r=\operatorname{rank}(G),
$$

and take the compact nonzero SVD

$$
G^{1/2}Y=U_q\Sigma_q V_q^\top,\qquad
q=\operatorname{rank}(G^{1/2}Y).
$$

Here q is not generally the rank of Y. Define

$$
X_\parallel=G^{1/2}U_qV_q^\top,\qquad
D=G-X_\parallel X_\parallel^\top
=G^{1/2}(I-U_qU_q^\top)G^{1/2}.
\tag{9}
$$

D is positive semidefinite with rank r−q. Choose a deterministic factor
F∈R^{K×(r−q)} such that FFᵀ=D, and a deterministic matrix
W∈R^{d×(r−q)} satisfying

$$
W^\top W=I,\qquad V_q^\top W=0.
$$

Then set

$$
\boxed{X=X_\parallel+FW^\top.}
\tag{10}
$$

Empty factors are permitted when r=q.

### Proof of feasibility and optimality

The cross terms in XXᵀ vanish because V_qᵀW=0. Therefore,

$$
XX^\top=X_\parallel X_\parallel^\top+D=G.
$$

Moreover,

$$
DY=G^{1/2}(I-U_qU_q^\top)G^{1/2}Y=0.
$$

Since D=FFᵀ, this implies FᵀY=0. Thus the completion contributes zero
to the objective:

$$
\langle X,Y\rangle_F
=
\langle X_\parallel,Y\rangle_F
=
\operatorname{tr}\Sigma_q.
$$

This attains the nuclear-norm upper bound for the Procrustes trace
objective. Equal row Gram matrices imply X=S_wQ for some orthogonal Q.
The unused-direction argument gives a proper rotation as well when
r<d. ∎

When q=r, D=0 and the product is unique; the construction agrees with
the unique Procrustes product. When q<r, many optimal prefixes exist.
A deterministic F,W makes the variant a function of G and R, but it
need not reproduce the authors' chosen prefix or its receiver behavior.

A numerical specification must fix rank tolerances, basis construction,
sign conventions and ties. “Use an eigendecomposition” alone does not
fully specify a canonical basis in a repeated eigenspace. Projected
coordinate axes, with a fixed selection and orthogonalization rule, are
one possible construction. The receiver-side projector need not be
materialized: a coordinate vector e_j can be projected using
e_j−V_q(V_qᵀe_j).

## 6. An exact two-token sign-bit collision

This construction demonstrates a possible information loss; it is not
a measurement on hidden states from an existing LLM.

Let K=2, d=3, with vocabulary embeddings e_1 and −e_1. Use the same two
vectors as a simple linear token readout. For b∈{−1,+1}, define

$$
S_b=
\begin{bmatrix}
1&b&0\\
-1&-b&0
\end{bmatrix},
\qquad
R=
\begin{bmatrix}
1&0&0\\
-1&0&0
\end{bmatrix}.
\tag{11}
$$

The first row always decodes to the positive token and the second to the
negative token: their readout scores depend on the first coordinate,
not b. Therefore both messages have identical reference tokens and R.
The raw sender states reveal b exactly.

The row means are zero and

$$
S_bS_b^\top
=
2\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

is independent of b. In the one-dimensional centered subspace,
Procrustes maps the sender direction to e_1. Direct evaluation of the
whiten–rotate–dewhiten equations gives

$$
Z_b=
\begin{bmatrix}
c_\lambda&0&0\\
-c_\lambda&0&0
\end{bmatrix},
\qquad
c_\lambda=
\sqrt{\frac{2(1+\lambda)}{2+\lambda}}.
\tag{12}
$$

The sign bit has already disappeared. Since 1≤c_λ<√2 for λ>0 and the
mean vocabulary norm is one, norm calibration gives exactly R.
Each row's nearest vocabulary embedding is itself, so any anchor blend
α∈[0,1] also gives exactly R.

The construction uses A=B=1 in the more general vector (A,bB,0). This
choice avoids the active covariance and row-norm floors in the pinned
implementation. A proper rotation can realize the alignment; the unused
third dimension removes any determinant obstacle.

The equalities are statements about real arithmetic. Float32
eigendecomposition and SVD can introduce small residual artifacts; no
claim of observed bitwise output equality is made before the numerical
study. Arbitrarily tiny A,B would also require explicit consideration
of the row-norm clamp.

## 7. Dual computation, compact transport, and costs

Let G_S=S_cS_cᵀ and G_R=R_cR_cᵀ, now using full K×K row matrices.
The ideal whitened sender Gram and target can be computed as

$$
G=K G_S(G_S+K\lambda I_K)^{-1},
\qquad
Y=(G_R/K+\lambda I_K)^{-1/2}R_c.
\tag{13}
$$

For any constructed whitened prefix X,

$$
X\Sigma_R^{1/2}
=
\sqrt{\lambda}\,X+(XR_c^\top)h_\lambda(G_R)R_c,
\tag{14}
$$

where

$$
h_\lambda(t)=
\frac{1}{K\left(\sqrt{t/K+\lambda}+\sqrt{\lambda}\right)},
\qquad t\geq0.
$$

Equation (14) follows by rationalizing
sqrt(t/K+λ)−sqrt(λ). It is valid at t=0 and for rank-deficient R_c.

These identities permit whitening, Procrustes product construction and
dewhitening in O(K²d+K³) arithmetic with O(Kd+K²) working storage,
without a d×d covariance or rotation. The vocabulary search remains
O(KVd). No end-to-end speedup follows merely from the reduced alignment
complexity.

For a shared receiver embedding table, the sender-centered Gram matrix
and reference token IDs are algebraically sufficient to reconstruct:

- the original mathematical product in the unique-product regime;
- the explicitly chosen canonical variant at every rank.

This changes the nominal representation from Kd real coordinates to a
symmetric K×K Gram matrix plus K token IDs. A centered basis can reduce
the independent symmetric coordinates to (K−1)K/2. Protocol metadata,
precision, transfer direction and reconstruction cost still need to be
charged in an actual transport study.

This is a statement of sufficient statistics in exact arithmetic.
Serializing a floating-point Gram matrix is not automatically a lossless
encoding of the authors' finite-precision output. In particular, a
Gram-only canonical packet does not reproduce an arbitrary
rank-deficient SVD null-space completion.

## 8. Eigenvalue floors and numerical qualifications

The covariance eigenvalue floor is distinct from regularization.
For a row-Gram eigenvalue g, the real-arithmetic counterpart of the
authors' floored whitening uses

$$
F_{\lambda,\varepsilon}(g)
=
\frac{g}{\max(g/K+\lambda,\varepsilon)},
\qquad \varepsilon=10^{-6}.
\tag{15}
$$

Replacing Equation (3) by this spectral function gives the corresponding
full-rank dual formula. The paired floored inverse square root and square
root still cancel exactly in real arithmetic.

For the all-rank dewhitening formula, put

$$
\beta=\sqrt{\max(\lambda,\varepsilon)},\qquad
f(t)=\sqrt{\max(t/K+\lambda,\varepsilon)},
$$

$$
h_{\lambda,\varepsilon}(t)=\frac{f(t)-\beta}{t}\quad(t>0).
$$

Use its continuous value at zero: zero if λ<ε, and
1/(2K√λ) if λ≥ε. Then

$$
X\,\operatorname{floor}_\varepsilon(\Sigma_R)^{1/2}
=
\beta X+(XR_c^\top)h_{\lambda,\varepsilon}(G_R)R_c.
\tag{16}
$$

With a fixed positive floor, unmatched target-null directions receive
√ε rather than a factor tending to zero as λ→0. Thus the unfloored
rank-deficient limit is not a literal limit theorem for the pinned code.
If all nonzero active covariance eigenvalues stay above the floor, the
full-rank limiting argument still applies to those active directions.

Additional numerical boundaries include:

- float32 formation of d×d low-rank covariances can introduce small
  negative or spurious eigenvalues;
- an exact-rank test and a numerical-rank threshold are different;
- eigenspace and SVD null-space bases can change under perturbations;
- nearest-vocabulary argmax is sensitive to ties and small margins;
- a determinant correction can change a nonunique residual completion;
- normalization is undefined at zero in the paper formula and is
  explicitly clamped in the implementation;
- final casting to model dtype can mask or amplify reported equivalence
  depending on the chosen error metric.

The proposed numerical study must report failure cases and conditioning,
rather than quietly dropping them or claiming equality after a permissive
rank truncation.

## 9. Prior art and the research boundary

The underlying mathematics is established:

- [Schönemann (1966), A generalized solution of the orthogonal Procrustes
  problem](https://engineering.purdue.edu/~bethel/schonemann_orth_proc.pdf)
  treats the orthogonal least-squares problem, including singular cases.
- [Bhatia, Jain and Lim (2017), On the Bures-Wasserstein distance between
  positive definite matrices](https://arxiv.org/html/1712.01504v1)
  connects factor alignment, polar decompositions and positive-definite
  geometric means. Equation (1) is an instance of this machinery.
- [Pumir, Singer and Boumal (2019), The generalized orthogonal Procrustes
  problem in the high noise regime](https://web.math.princeton.edu/~amits/publications/1907.01145.pdf)
  uses Gram invariants to characterize configurations up to orthogonal
  transformations.
- [Kornblith et al. (ICML 2019), Similarity of neural network representations
  revisited](https://proceedings.mlr.press/v97/kornblith19a.html)
  identifies a related high-width degeneracy for representation
  comparisons invariant to invertible transformations. It is context for
  the whitening issue, not a direct theorem about this communication
  interface.
- [Meng, Tang and Wong (2026), FLoRG: Federated fine-tuning with low-rank
  Gram matrices and Procrustes alignment](https://arxiv.org/html/2602.17095v1)
  combines Gram-based parameterization, Procrustes alignment and reduced
  communication in federated fine-tuning. Its Section 3.1 sends low-rank
  factors and aggregates their Gram matrices; that mechanism differs from
  reconstructing an activation prefix from its row Gram and token IDs.
  The broader combination of Gram representations, Procrustes and
  communication efficiency is therefore not a new claim here.

The bounded lead is a StateBridge-specific sufficient-statistic analysis,
an exact dual implementation in its uniqueness regime, a specified
canonical variant for other ranks, and a transparent numerical evaluation.
Any later claim about receiver quality, communication savings or runtime
requires its own measured evidence. No result in this note establishes a
breakthrough, a new general theorem, or a refutation of the paper.
