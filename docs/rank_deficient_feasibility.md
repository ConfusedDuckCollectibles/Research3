# Rank-deficient alignment: preserving the chosen output

**Status: analytical feasibility note, not an executed study.** No new model, codec, numerical fixture or timing result is reported here. The question is whether a receiver can reconstruct the *particular* dense SO alignment already chosen by the sender, including its otherwise arbitrary completion.

**Conclusion:** source Gram plus actual residual orientation is sufficient in exact arithmetic under a spare-direction assumption. For the main full-source-rank case, however, a simpler representation is smaller: transmit a symmetric coefficient on the column space of the centered reference matrix and the actual remaining coordinates. Its scalar count is

$$
\frac{q(q+1)}{2}+(r-q)d,\qquad r=K-1,\quad q=\operatorname{rank}(R_c).
$$

This is a feasibility result and a stronger comparison baseline. It establishes neither a new general compression principle nor bitwise reproduction of floating-point legacy code.

## 1. Assumptions and the chosen SO output

Let $S_c,R_c\in\mathbb R^{K\times d}$ be centered source and reference rows. Assume $2\le K<d$ and that the paired covariance whitening/dewhitening matrices are invertible, as with positive regularization or the specified positive eigenvalue floor. Write

$$
X=S_cW_SQ,\qquad Y=R_cW_R,\qquad
Z=XW_R^{-1}+\mathbf1\mu_R^\top,
$$

where $Q\in SO(d)$ is the *actual selected optimum* and $Z$ is the pre-calibration prefix. Let $p=\operatorname{rank}(X)$.

The standard Procrustes optimality relation is

$$
Y^\top X\succeq0.
$$

To see why the proper-rotation restriction does not invalidate it here, factor
$(S_cW_S)^\top Y=U\Sigma V^\top$. An optimal orthogonal map gives
$Y^\top X=V\Sigma V^\top$. If a determinant correction is needed, it can act on a zero singular direction because $p\le K-1<d$; the product stays PSD. This argument does **not** cover a full-feature-rank case with no spare direction and a necessary nonzero-cost determinant correction. Deficient-rank Procrustes and proper-rotation distinctions are longstanding; see [Schönemann (1966)](https://doi.org/10.1007/BF02289451) and the [existing geometry note](alignment_geometry.md).

## 2. Source Gram plus actual orientation is sufficient

Let $G=XX^\top$ and use a fixed $r$-column Helmert basis to serialize its reduced $r\times r$ Gram. Define

$$
G^{1/2}Y=U_c\Sigma_cV_c^\top,\qquad
c=\operatorname{rank}(G^{1/2}Y).
$$

For any optimum, equality in the Procrustes trace bound fixes

$$
X_\parallel=G^{1/2}U_cV_c^\top.
$$

The *actual* residual $E=X-X_\parallel$ obeys

$$
EE^\top=D_G=G-X_\parallel X_\parallel^\top,\qquad
\operatorname{rank}(E)=p-c,\qquad EV_c=0.
$$

Choose a deterministic full-column factor $F$ with $FF^\top=D_G$. The sender can extract

$$
W=E^\top(F^\dagger)^\top,\qquad
E=FW^\top,\qquad W^\top W=I.
$$

The receiver, given the Gram, reference rows and the transmitted $W$, reconstructs the sender's selected completion. It never substitutes a newly chosen canonical completion. The general rank is $p-c$, not $p-\operatorname{rank}(Y)$; these coincide when the centered source has full row rank.

A straightforward packet sends the reduced Gram triangle and $(p-c)d$ orientation entries. Ignoring shared framing/IDs, that is

$$
\frac{r(r+1)}2+(p-c)d
$$

scalars. When $p=c$, the residual vanishes and a coefficient-only unique-product mode is possible. But a coefficient alone does not generally identify the covariance or orientation of a nonzero residual.

This construction has avoidable liabilities: a Gram squares singular-value conditioning; recovering $F^\dagger$ can amplify small directions; and nearly degenerate factorizations need compatible basis conventions. A Gram determines an orthogonal orbit, not a particular orientation—an established distinction in [Pumir, Singer and Boumal](https://web.math.princeton.edu/~amits/publications/1907.01145.pdf).

## 3. A smaller reference-space representation

Choose $L\in\mathbb R^{K\times q}$ with orthonormal columns spanning the columns of $R_c$, and $N\in\mathbb R^{K\times(r-q)}$ completing $L$ to an orthonormal basis of $\mathbf1^\perp$. These depend on receiver information, not the private source. Define

$$
C=L^\top R_c=L^\top R,\qquad C_Y=CW_R.
$$

$C_Y$ has full row rank. Set $T=L^\top X$. Optimality says $C_Y^\top T=Y^\top X$ is symmetric PSD. Consequently $T$ annihilates the null space of $C_Y$, so there is a unique matrix $A$ with $T=AC_Y$. Multiplying by a right inverse shows that $A$ is symmetric PSD.

Paired dewhitening now cancels:

$$
L^\top(Z-\mathbf1\mu_R^\top)=A C_YW_R^{-1}=AC.
$$

Define the actual remaining coordinates

$$
D=N^\top(Z-\mathbf1\mu_R^\top).
$$

Then the desired legacy pre-prefix is exactly

$$
\boxed{
Z=L A(L^\top R)+ND+\mathbf1\mu_R^\top.
}
$$

The sender can extract $A=[L^\top(Z-\mathbf1\mu_R^\top)]C^\dagger$ and $D$ from its already-computed output. The receiver needs only the triangle of $A$, all entries of $D$, the IDs and the declared shared context. It needs no private source, source Gram, new null completion or receiver dewhitening.

This identity does not require full source rank. However, $\operatorname{rank}(D)$ need not equal $p-c$ for a general deficient source: $D$ can contain both determined and undetermined components. The simple layout sends its full $(r-q)\times d$ array.

### Repeated IDs

If there are $m$ distinct IDs and their distinct embeddings are affinely independent, then $q=m-1$. Deterministic weighted group contrasts give $L$; within-group Helmert contrasts give $N$, with $K-m$ columns. Both can be constructed from the ID sequence and group counts alone. The receiver must still check the affine-rank assumption. If distinct embeddings are dependent or numerically ambiguous, merely counting IDs is insufficient.

The extremes are explicit:

- **$q=r$:** $N$ is empty and only the symmetric coefficient remains, matching the coefficient-only representation up to the declared sample-basis change.
- **$q=0$:** the coefficient is empty and $D$ contains the full centered prefix. The only simple saving is the known mean row; no hidden orientation disappears.
- **Full source rank:** $p=r$, $c=q$, and $\operatorname{rank}(D)=r-q$. Compared with Gram plus orientation, this layout saves exactly $[r(r+1)-q(q+1)]/2$ scalar entries.

## 4. Wire cost and strong generic baselines

For a proposed group-contrast mode using the existing 56-byte header, $K$ uint32 IDs, $b$ bytes per scalar, and a basis/version contract bound by the shared session digest, the calculated size is

$$
56+4K+b\left[\frac{q(q+1)}2+(r-q)d\right].
$$

Here $q=m-1$ is derivable from the IDs; an arbitrary-rank mode needing explicit rank or pivot metadata must charge it separately. These are proposed serialized sizes, not measurements or TCP totals.

For $K=64,d=1024$, float64:

| Distinct IDs $m$ | Reference rank $q$ under the stated assumption | Proposed bytes |
|---:|---:|---:|
| 1 | 0 | 516,408 |
| 16 | 15 | 394,488 |
| 32 | 31 | 266,424 |
| 64 | 63 | 16,440 |

The direct final-prefix packet is 524,344 bytes; the token-only reference is 312 bytes and does not reconstruct the same output.

A strong generic baseline is essential. The centered pre-prefix in a fixed $r$-row basis has rank $p$. A row-skeleton/interpolative factorization can send $p$ independent rows and interpolation weights for the others, using

$$
p\,d+p(r-p)
$$

scalars plus pivot/rank metadata. This is stronger than naively charging both redundant factors of a thin SVD. [Interpolative decomposition](https://arxiv.org/abs/2002.01387) is established numerical linear algebra.

For $p=r$, generic centered coding needs $rd$ scalars; the proposed layout saves $qd-q(q+1)/2$ of those. For small $p$, generic rank-$p$ coding can be much smaller than the proposed full-$D$ layout. It must remain an explicit comparator or fallback. A symmetric coefficient and a residual are structured matrix coding, not inherently a new communication method.

The flat-$W$ comparison does not exploit its Stiefel constraints. For full source rank, put $t=r-q$. An orientation in the reference-feature complement has $t(d-q)-t(t+1)/2$ local degrees of freedom. Adding the Gram parameters gives

$$
\frac{r(r+1)}2+t(d-q)-\frac{t(t+1)}2
=\frac{q(q+1)}2+td.
$$

Thus the simpler layout removes redundant coordinates; it is not a new information bound over an optimally parameterized orientation.

No information-theoretic lower bound follows from these wire counts. The space of all admissible optimal pre-prefixes, the image of a particular deterministic SVD implementation, and the post-normalized/anchored outputs are different sets. Normalization is many-to-one. Parameter counts alone establish neither the minimum bits for finite-precision data nor optimality under lossy, entropy-coded or discontinuous encodings.

## 5. Costs and finite-precision limits

The sender has **already paid for the legacy dense alignment** to obtain its particular $Z$. Extracting the coefficients, constructing/verifying bases, serializing, checking the actual rounded packet and any fallback are additional costs. A fair complete latency account includes that legacy solve on every compared route; a separate incremental codec measurement must clearly state the prefix is already available.

The receiver pays basis work, reconstruction and the original normalization/nearest-vocabulary anchoring. The embedding table, its order, numerical parameters and basis contract are shared side information. Cold transfer costs include the table and session metadata; a direct final-prefix recipient does not need that table. Generic pre-prefix coding with receiver postprocessing needs the same legitimate context.

For extraction, a QR/SVD solve for $C^\dagger$ is preferable to treating an ill-conditioned $CC^\top$ inverse as harmless. Small singular values, rounding, changes in numerical rank, tiny pre-norms and vocabulary ties all need explicit treatment.

Most importantly, floating-point legacy output may violate exact centering, PSD/symmetry and support identities slightly. Its numerical residual need not have the ideal exact rank. Symmetrizing an extracted coefficient or dropping those directions therefore does not prove bitwise reproduction. Preserving every final float bit can require an explicit numerical correction, potentially as large as the original prefix. A practical approximate codec must check full-vector fidelity against the *actual selected output* and retain raw fallback. Matching nearest token IDs alone is inadequate; the [original alignment results](alignment_results.md) demonstrate that limitation.

## 6. Prior art and one bounded next check

The relevant prior art is substantial:

- [Schönemann (1966)](https://doi.org/10.1007/BF02289451) treats rank-deficient orthogonal Procrustes.
- [Pumir, Singer and Boumal](https://web.math.princeton.edu/~amits/publications/1907.01145.pdf) study Gram invariants, orthogonal orbits and factor sensitivity.
- [Kernel Hyperalignment](https://proceedings.neurips.cc/paper_files/paper/2012/file/170c944978496731ba71f34c25826a34-Paper.pdf) moves a large-feature alignment problem into sample-space calculations.
- [FLoRG](https://arxiv.org/html/2602.17095v1) combines Gram parameterization, Procrustes and communication-efficient federated adaptation.
- [Palu](https://arxiv.org/html/2407.21118v2) uses compressed latent states with reconstruction matrices and explicitly addresses reconstruction overhead.

These sources do not by themselves establish whether this precise legacy-prefix specialization has appeared before. They rule out treating factorized state communication, Gram invariance or sample-space Procrustes as novel on their own.

**Recommendation:** do not implement naive Gram-plus-orientation or launch another latency sweep yet. One defensible next step is a small, explicitly labeled regression assay on the already-retained 27 suite-B inputs and chosen dense pre-prefixes, including the repeated-ID geometry. Freeze the codec/basis rules first, then compare actual receiver-only reconstruction with direct transmission and generic rank-$p$ coding at matched precision. Preserve every case, actual byte count, conditioning/rank decision, full pre/final-vector error and fallback. Do not choose a new canonical completion or reselect a favorable dense output.

That check would answer whether the exact algebra survives the stored legacy floating-point outputs. It would establish no new-data generalization or speed result. Only successful fidelity and a material advantage over the strongest representation baseline would justify a later fresh workload study.
