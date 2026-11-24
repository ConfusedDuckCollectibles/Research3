# What source information does StateBridge retain?

**Status:** analytical characterization in exact real arithmetic. This note does not report a new experiment, a verified novelty claim, a bit-exact codec, or a model-quality result.

The useful conclusion has two levels:

1. **Any exact orthogonal alignment preserves enough information in its pre-calibration output to recover the centered source Gram matrix, at positive regularization and a known reference.** This holds even for a rank-deficient reference and a nonunique Procrustes completion.
2. **In the unique-product regime, that output contains exactly the centered Gram information.** If the actual aligned rows have full row rank, the same statement survives row calibration and vocabulary anchoring. A full-row-rank uncentered reference is a sufficient condition in this regime.

Neither conclusion recovers the source mean or its absolute feature-space orientation in the unique-product case. Neither identifies which information a pretrained receiver can use.

## 1. Conventions and scope

Let $S,R\in\mathbb R^{K\times d}$, with $2\le K<d$, and let

$$
P=I_K-\frac1K\mathbf1\mathbf1^\top,\qquad
S_c=PS,\qquad R_c=PR,\qquad
\mu=\frac1K R^\top\mathbf1.
$$

The reference $R$, its ordered token IDs, the vocabulary table, and all algorithm constants are held fixed and known when discussing inversion. This is a **conditional** statement: the reference tokens are not necessarily recoverable from the transmitted prefix alone. The paper explicitly says in [Section 3.2](https://arxiv.org/html/2608.13317v1#S3.SS2) that these reference token IDs are not transmitted. Thus this inverse is not, by itself, an operational decoder for the paper's original prefix-only handoff; a test must provide or fix the reference side information explicitly.

For $\lambda>0$ and a covariance eigenvalue floor $\epsilon\ge0$, define

$$
D_S=\left[\max_{\rm spectral}
 \left(S_c^\top S_c/K+\lambda I_d,\epsilon I_d\right)\right]^{1/2},
\quad W_S=D_S^{-1},
$$

and define $D_R,W_R$ analogously. The exact pre-calibration output is

$$
Z=S_cW_SQD_R+\mathbf1\mu^\top,\qquad Q^\top Q=I_d.
\tag{1}
$$

The paper uses the unfloored equations. The [pinned authors' implementation](https://github.com/YanwenPneg/StateBridge/blob/3f6bf5442c6e8848555a6132516e6d36f35444fb/methods/state_bridge.py) floors covariance eigenvalues, enforces a proper rotation, clamps row norms, computes in float32, and casts the result to the model's dtype. The theorems below describe the corresponding exact operations, not those finite-precision casts. This note stops at the alignment/postprocessing function; optional later pooling, noise, or other prefix processing is outside scope. The code also flattens batch and sequence before centering: $K$ must denote the total aligned row count, or batch size must be one.

## 2. A broad Gram-recovery theorem

Let $\Gamma=S_cS_c^\top\succeq0$. Define the scalar function, including its value at zero,

$$
f_{\lambda,\epsilon}(g)=
 \frac{g}{\max(g/K+\lambda,\epsilon)},\qquad g\ge0.
\tag{2}
$$

An SVD of $S_c$ gives

$$
S_cW_S^2S_c^\top=f_{\lambda,\epsilon}(\Gamma),
\tag{3}
$$

where a scalar function of a symmetric matrix is applied spectrally.

Given $Z$ and the reference, form

$$
T=(Z-\mathbf1\mu^\top)W_R.
$$

Equation (1) implies $T=S_cW_SQ$, and therefore

$$
TT^\top=f_{\lambda,\epsilon}(\Gamma).
\tag{4}
$$

For every $\lambda>0$, $f_{\lambda,\epsilon}$ is strictly increasing from $[0,\infty)$ onto $[0,K)$. It consequently has a unique spectral inverse, including zero eigenvalues.

If $\lambda\ge\epsilon$, or no floor is used,

$$
\psi(x)=\frac{K\lambda x}{K-x},\qquad 0\le x<K.
\tag{5}
$$

If $0<\lambda<\epsilon$, set $x_0=K(1-\lambda/\epsilon)$. Then

$$
\psi(x)=
\begin{cases}
\epsilon x,&0\le x\le x_0,\\
K\lambda x/(K-x),&x_0<x<K.
\end{cases}
\tag{6}
$$

Thus

$$
\boxed{\Gamma=\psi(TT^\top).}
\tag{7}
$$

This does not require $Q$ to be uniquely determined, or even Procrustes-optimal: exact orthogonality is sufficient. It does not require a full-rank reference. A selected null-space completion can carry **additional** information, so this broad result establishes recoverability of **at least** the centered Gram, not equivalence to it.

## 3. Exactly Gram information in the unique-product regime

Let $r=K-1$, and choose a fixed $H\in\mathbb R^{K\times r}$ with

$$
H^\top H=I_r,\qquad HH^\top=P.
$$

Write $A=H^\top S_c$, $B=H^\top R_c$, and assume that both have full row rank $r$. Put

$$
G=AA^\top\succ0,\quad G_R=BB^\top\succ0,\quad
X=f_{\lambda,\epsilon}(G),\quad Y=f_{\lambda,\epsilon}(G_R).
$$

The unique aligned product, although the ambient rotation need not be unique, is

$$
Z=HMB+\mathbf1\mu^\top,
\tag{8}
$$

where $M\succ0$ is the unique positive-definite solution of

$$
MYM=X.
\tag{9}
$$

For example,

$$
M=Y^{-1/2}(Y^{1/2}XY^{1/2})^{1/2}Y^{-1/2}.
$$

This is the classical positive-definite Riccati/Bures map. Since the centered rank is below $d$, an unused ambient direction allows a determinant correction without changing the aligned product.

An explicit inverse is

$$
M=(H^\top Z)B^\top G_R^{-1},\qquad
X=MYM,\qquad G=\psi(X).
\tag{10}
$$

The exact finite-image conditions for a proposed $Z$ are:

- its row mean is $\mu$;
- $H^\top Z=MB$, with $M$ computed by (10);
- $M$ is symmetric positive definite;
- $0\prec MYM\prec KI_r$.

These conditions are necessary and sufficient. They avoid mistaking an arbitrary matrix with the right dimensions for an attainable prefix.

It follows that, with the reference fixed,

$$
Z(S_1)=Z(S_2)
\quad\Longleftrightarrow\quad
S_{1c}S_{1c}^\top=S_{2c}S_{2c}^\top.
\tag{11}
$$

For arbitrary real source matrices in this regime, the fibers are precisely changes of source mean and right orthogonal orientation. This is a complete invariant of centered Euclidean geometry, not recovery of the original hidden coordinates.

## 4. Dimension and Jacobian rank

The space $G\succ0$ has

$$
n=\frac{r(r+1)}2=\frac{K(K-1)}2
$$

real degrees of freedom. The pre-calibration map has Jacobian rank exactly $n$ throughout the unfloored positive-regularization domain.

One direct proof differentiates the unfloored map:

$$
Df_G[E]=K^2\lambda(G+K\lambda I)^{-1}
 E(G+K\lambda I)^{-1}.
$$

This is invertible on symmetric matrices. Differentiating (9) gives

$$
dX=(dM)YM+MY(dM).
$$

After congruence by $Y^{1/2}$, this is a Sylvester equation with positive-definite coefficient $Y^{1/2}MY^{1/2}$, hence an invertible map on symmetric perturbations. Finally, $dM\mapsto H(dM)B$ is injective.

With a positive covariance floor the same rank holds on each smooth stratum away from eigenvalues at the floor transition. The map remains injective at the transition, but a classical Jacobian need not exist there.

For the unrestricted source matrix $S$, the map $S\mapsto G$ is a submersion at full centered rank. Its Jacobian has rank $n$ and kernel dimension $Kd-n$. This is a dimension statement over real matrices. It is not a bit-capacity estimate or a statement about the dimension of the attainable states of a particular transformer.

## 5. Row normalization can be inverted using the known mean

Let the row calibration be

$$
C_i=\frac{\tau Z_i}{\max(\|Z_i\|,\delta)},\qquad
\tau>0,\quad\delta>0.
\tag{12}
$$

Every row is multiplied by a positive scalar, including when the clamp is active, so $\operatorname{rank}C=\operatorname{rank}Z$.

**If the actual $Z$ has row rank $K$,** solve

$$
C^\top s=K\mu,\qquad
s=(CC^\top)^{-1}C(K\mu).
\tag{13}
$$

The solution is unique, and for an actual calibrated prefix it satisfies

$$
s_i=\max(\|Z_i\|,\delta)/\tau>0,\qquad
\boxed{Z=\operatorname{diag}(s)C.}
\tag{14}
$$

No assumption that the norm clamp is inactive is needed. Full row rank excludes zero rows. It also forces full centered source rank: equation (1) gives $\operatorname{rank}Z\le\operatorname{rank}S_c+1$, so $\operatorname{rank}Z=K$ requires $\operatorname{rank}S_c=K-1$. The broad pre-calibration inverse covers deficient sources, but this sufficient postprocessing condition does not.

In the unique-product regime,

$$
Z=\left(HMH^\top+\mathbf1\mathbf1^\top/K\right)R.
$$

The left factor is invertible, hence $\operatorname{rank}Z=\operatorname{rank}R$. Therefore full row rank of the **uncentered** reference is a convenient sufficient condition. It is stronger than full centered rank.

Outside the unique regime, the actual aligned $Z$ can still have rank $K$, even with repeated reference IDs. The scale-recovery theorem then still applies. A rank-deficient reference alone is not proof of final information loss.

## 6. Vocabulary anchoring is injective at a mixing ratio below one

Let $v_j$ be a vocabulary row and select

$$
j(C)=\arg\max_j C^\top w_j,\qquad
F(C)=(1-\alpha)C+\alpha v_{j(C)},\qquad 0\le\alpha<1.
\tag{15}
$$

Use a fixed, shared first-index tie rule. Normalizing the query $C$ does not change these comparisons, because its normalization is a common positive divisor.

### Ordinary non-tiny vocabulary rows

If every nonzero vocabulary row is normalized to unit direction,
$w_j=v_j/\|v_j\|$, then the selected nonzero anchor remains a maximizer after blending:

$$
F(C)^\top(w_j-w_k)
=(1-\alpha)C^\top(w_j-w_k)
+\alpha\|v_j\|(1-w_j^\top w_k)\ge0.
$$

For $\alpha>0$, distinct-direction ties become strict. Duplicate directions retain the same deterministic tie choice. A selected zero anchor merely scales $C$ and preserves its ranking.

Thus reselecting the anchor from $F$ and computing

$$
C=(F-\alpha v_j)/(1-\alpha)
\tag{16}
$$

recovers the calibrated row. At $\alpha=0$, simply take $C=F$.

### General lemma including vocabulary normalization floors

The implementation uses $w_j=v_j/\max(\|v_j\|,\eta)$, with $\eta>0$. Tiny nonzero rows can invalidate the simple reselect-from-$F$ rule. For example, take vocabulary rows $0.1\eta e_1,\eta e_1$, calibrated row $C=-0.55\eta e_1$, and $\alpha=0.9$. The first anchor is selected, but $F=0.035\eta e_1$ is positive and reselecting from $F$ picks the second. This is an exact algebraic example, not an empirical observation. Tiny rows do **not** invalidate injectivity.

Indeed, $w(v)$ is projection of $v/\eta$ onto the unit ball. The standard firm-nonexpansive projection inequality ([Bauschke, Moffat and Wang, Fact 1.1](https://arxiv.org/html/1101.4688v1)) gives

$$
(v_j-v_k)^\top(w_j-w_k)\ge
\eta\|w_j-w_k\|^2.
\tag{17}
$$

Suppose two calibrated rows $C,D$, selecting $j,k$, give the same $F$. Selection implies

$$
(C-D)^\top(w_j-w_k)\ge0.
$$

Equality of the outputs gives
$C-D=-\alpha(v_j-v_k)/(1-\alpha)$. For $0<\alpha<1$, (17) forces $w_j=w_k$. The shared deterministic tie rule then forces the same selected index, and hence $C=D$.

An explicit finite inverse exists by checking candidates
$C_j=(F-\alpha v_j)/(1-\alpha)$ against their selection rule. A practical audit need not perform a full-vocabulary enumeration: it can restrict the fast inverse to supported non-tiny rows and report the remaining cases as unsupported by that procedure. Such rejection is not evidence of information destruction.

## 7. Combined statement

For a known fixed reference, positive regularization, positive target norm, and $0\le\alpha<1$:

- If the **actual** pre-calibration $Z$ has rank $K$, the final postprocessed prefix determines $Z$, and therefore determines the centered source Gram by (7), regardless of reference rank or completion uniqueness.
- If both centered matrices have full rank $K-1$ and the uncentered reference has rank $K$, the final prefix and the centered source Gram determine each other exactly.
- On smooth strata with stable anchor choices, the final map from $G$ has local Jacobian rank $n=K(K-1)/2$. Row calibration is injective on mean-preserving tangents when the rows of $Z$ are independent; anchoring then multiplies the derivative by $1-\alpha$.
- At anchor boundaries the map can be discontinuous. Global exact injectivity does not imply a globally smooth or numerically stable inverse.

Under the stated equivalence conditions, for any task variable $L$, the exact conditional information satisfies
$I(L;F\mid R)=I(L;G\mid R)$ whenever those quantities are defined. In the broader recoverability case, the corresponding relation is an inequality in favor of $F$, not equality. Continuous mutual information can be infinite; these identities are not useful numerical capacity estimates.

## 8. Failure cases and conditioning

**Unknown reference.** Fixed reference side information is substantive, even if every candidate reference has full row rank. For $K=2,d=3,\lambda=1$, let

$$
R_1=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix},\qquad
R_2=\begin{bmatrix}3/2&-1/2&0\\-1/2&3/2&0\end{bmatrix}.
$$

Use one shared vocabulary containing all four rows. Both references have mean $(e_1+e_2)/2$. Their reduced reference Grams are $1$ and $4$. Reduced source Grams $G_1=1$ and $G_2=2/5$ give whitened source Grams $2/3$ and $1/3$, while the corresponding reference values are $2/3$ and $4/3$. Equation (9) therefore gives $M_1=1$ and $M_2=1/2$. Both pre-prefixes equal $R_1$, so the complete calibrated and anchored outputs also coincide under the same vocabulary and parameters, despite different source Grams. Sources realizing these Grams exist, for example $S_i=H\sqrt{G_i}e_3^\top$. This is an exact arbitrary-matrix counterexample; it is not an observed pair of attainable transformer states. It shows why the conditional inverse must not be promoted to unconditional recovery from the original prefix alone.

**Zero regularization.** Without a floor, exact whitening sends every positive source eigenvalue to $K$, losing its magnitude. With a fixed positive floor and $\lambda=0$, $f(g)=\min(g/\epsilon,K)$ saturates. Neither case supports the global inverse above.

**Complete anchoring.** At $\alpha=1$, only selected vocabulary rows remain, so arbitrary continuous calibrated rows cannot be recovered.

**Rank deficiency after alignment.** If $\operatorname{rank}Z<K$, the mean equation need not determine row scales. This does not prove loss in every such case, but it permits it. For $K=2$, reference $R=(e_1,-e_1)^\top$, and a range of positive scalar source Grams, unique pre-prefixes have the form $Z=(m e_1,-m e_1)^\top$. Once both rows are above the norm clamp, calibration removes $m$, and the final prefix is constant. Here the centered reference is full rank but its uncentered row rank is only one.

**Positive regularization can still be poorly conditioned.** In the unfloored scalar case,

$$
f'(g)=\frac{K^2\lambda}{(g+K\lambda)^2},\qquad
\frac{x\psi'(x)}{\psi(x)}
=\frac{K}{K-x}=1+\frac{g}{K\lambda}.
\tag{18}
$$

Large source eigenvalues relative to $K\lambda$ can therefore amplify inverse error substantially. Mean-based scale recovery also deteriorates as the smallest singular value of $C$ approaches zero. Anchor subtraction amplifies perturbations by $1/(1-\alpha)$; near-tied decisions and weak reference singular values add further sensitivity.

In the unfloored full-rank regime, outputs can converge to the same reference as $\lambda\to0$ while remaining mathematically distinct at every positive $\lambda$. This is exact injectivity with a collapsing numerical separation, not a contradiction. A fixed active covariance floor changes that limiting argument.

**Finite precision.** Rounding, approximate orthogonality, eigenvalue thresholding, and the final model-dtype cast can destroy distinctions. An empirical inverse must retain domain failures, residuals, ranks, margins, and condition estimates rather than silently clip eigenvalues into $[0,K)$. Failure of an inverse algorithm does not by itself establish a collision or unusable task information.

## 9. Arbitrary matrices are not attainable transformer states

The algebra treats $S$ as an arbitrary real matrix while $R$ is fixed. A transformer's hidden states obey architectural and data constraints; changing their mean or rotating their feature coordinates need not produce another attainable state, and can change the decoded reference tokens.

Consequently, the lost mathematical degrees of freedom do not by themselves demonstrate lost task information on a real distribution. Conversely, an exact inverse does not show that a frozen receiver can learn or exploit it. A task-level claim requires evidence on actual states and relevant labels, with finite precision and conditioning accounted for.

## 10. Prior art and value of this result

The ingredients have substantial primary-source precedent:

- [Schönemann (1966), *A generalized solution of the orthogonal Procrustes problem*](https://web.stanford.edu/class/cs273/refs/procrustes.pdf): the classical orthogonal alignment solution.
- [Bhatia, Jain and Lim, *On the Bures-Wasserstein distance between positive definite matrices*](https://arxiv.org/abs/1712.01504): positive-definite matrix geometry and the unique Riccati/geometric-mean map underlying (9).
- [Pumir, Singer and Boumal, *The generalized orthogonal Procrustes problem in the high noise regime*](https://web.math.princeton.edu/~amits/publications/1907.01145.pdf): Gram matrices characterize orthogonal equivalence classes.
- [Wadia et al., ICML 2021](https://proceedings.mlr.press/v139/wadia21a.html): sample-Gram information, whitening, and regularization in neural-network training. Its model and training assumptions differ from this fixed communication map.
- [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html): representation similarity and the degeneracy of broad transformation invariances in high dimension.
- [FLoRG, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85ce42cbc4842572cfe7a9a9b8fc95ec-Abstract-Conference.html): Gram aggregation and Procrustes alignment already appear in communication-efficient federated fine-tuning; this is a different application.
- [StateBridge, Section 3.3 and Appendix A](https://arxiv.org/html/2608.13317v1): the actual alignment and postprocessing sequence characterized here.

The contribution of this note is an **application-specific information audit**: explicit inverse and range conditions; a separation between exact Gram equivalence and broader Gram recoverability; and the effect of the known mean on postprocessing identifiability. Novelty has not been established. The result is worth retaining because it prevents two inaccurate conclusions: that positive-regularization alignment necessarily erases the Gram, and that exact real-arithmetic recoverability establishes practical semantic preservation.

No new measurement was performed for this note.
