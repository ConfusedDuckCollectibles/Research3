"""Small-matrix alignment algorithms for the frozen numerical study.

These are float64 mathematical implementations, not a copy of the upstream
float32 program.  Rank-deficient canonical completion is a separate variant.
"""
from dataclasses import dataclass, field
import numpy as np

ATOL = 1e-10
RTOL = 1e-8
RANK_TOL = 1e-10
AMBIGUITY_LO = 1e-12
AMBIGUITY_HI = 1e-8
AXIS_TOL = 1e-10


class Unsupported(ValueError):
    """The declared numerical/dimensional guarantee does not cover an input."""
    def __init__(self, reason, diagnostics=None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics or {}


@dataclass
class Alignment:
    pre: np.ndarray
    white: np.ndarray | None = None
    sender_gram: np.ndarray | None = None
    target_white: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    bypass: bool = False


def helmert(k):
    if k < 1:
        raise ValueError("K must be positive")
    u = np.zeros((k, max(0, k - 1)), dtype=np.float64)
    for j in range(k - 1):
        denom = np.sqrt((j + 1) * (j + 2))
        u[:j + 1, j] = 1.0 / denom
        u[j + 1, j] = -(j + 1) / denom
    return u


def close_error(actual, reference):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    error = float(np.linalg.norm(a - b))
    norm = float(np.linalg.norm(b))
    return {"absolute": error, "relative": error / max(norm, 1e-300),
            "reference_norm": norm, "passed": bool(error <= ATOL + RTOL * norm)}


def _eigh(a, trace):
    trace.append({"operation": "eigh", "shape": list(a.shape)})
    return np.linalg.eigh((a + a.T) * 0.5)


def _svd(a, trace):
    trace.append({"operation": "svd", "shape": list(a.shape)})
    return np.linalg.svd(a, full_matrices=False)


def spectrum_info(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise Unsupported("nonfinite_spectrum")
    scale = max(1.0, float(np.max(values, initial=0.0)))
    z = values / scale
    ambiguous = (np.abs(z) >= AMBIGUITY_LO) & (np.abs(z) <= AMBIGUITY_HI)
    return {
        "rank": int(np.count_nonzero(z > RANK_TOL)),
        "scale": scale,
        "minimum": float(np.min(values)) if len(values) else 0.0,
        "maximum": float(np.max(values, initial=0.0)),
        "ambiguity_count": int(np.count_nonzero(ambiguous)),
        "negative_invalid": bool(np.any(z < -RANK_TOL)),
        "negative_clipped_mass": float(-np.sum(values[values < 0])),
        "positive_truncated_mass": float(np.sum(values[(values > 0) &
                                                       (z <= RANK_TOL)])),
        "eigenvalues": values.tolist(),
    }


def _supported_values(values, label, full=None):
    info = spectrum_info(values)
    if info["negative_invalid"]:
        raise Unsupported(label + "_not_psd", info)
    if info["ambiguity_count"]:
        raise Unsupported(label + "_rank_ambiguous", info)
    if full is not None and info["rank"] != full:
        raise Unsupported(label + "_not_full_rank", info)
    clean = np.where(values > RANK_TOL * info["scale"], values, 0.0)
    return clean, info


def _psd(a, label, trace, full=None):
    values, vectors = _eigh(a, trace)
    clean, info = _supported_values(values, label, full)
    return clean, vectors, info


def _parameters(lam, eps):
    if not np.isfinite([lam, eps]).all() or lam < 0 or eps < 0:
        raise ValueError("lambda and epsilon must be finite and nonnegative")
    if max(lam, eps) <= 0:
        raise Unsupported("unregularized_zero_floor_not_supported")


def _pair(s, r, fast):
    s = np.asarray(s, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if s.ndim != 2 or s.shape != r.shape or not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError("finite, equally shaped two-dimensional inputs required")
    k, d = s.shape
    if d < 1 or k < 1:
        raise ValueError("nonempty matrices required")
    if fast and not (2 <= k < d):
        raise Unsupported("outside_2_le_K_lt_d", {"K": k, "d": d})
    return s, r, k, d


def _from_values(vectors, values):
    return (vectors * values[None, :]) @ vectors.T


def dense_reference(s, r, lam=1e-3, eps=1e-6, details=True):
    """Dense, properly rotated reference with within-call eigenbasis reuse."""
    _parameters(lam, eps)
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError("finite matrix required")
    if r is None or np.asarray(r).size == 0:
        return Alignment(s.copy(), diagnostics={"method": "dense", "reason": "missing_tokens"},
                         bypass=True)
    s, r, k, d = _pair(s, r, False)
    if k < 2:
        return Alignment(s.copy(), diagnostics={"method": "dense", "reason": "fewer_than_two_rows"},
                         bypass=True)
    trace = []
    sc, rc = s - s.mean(axis=0), r - r.mean(axis=0)
    cs = sc.T @ sc / k + lam * np.eye(d)
    cr = rc.T @ rc / k + lam * np.eye(d)
    es, vs = _eigh(cs, trace)
    er, vr = _eigh(cr, trace)
    es, er = np.maximum(es, eps), np.maximum(er, eps)
    if np.min(es) <= 0 or np.min(er) <= 0:
        raise Unsupported("dense_nonpositive_covariance")
    sw = sc @ _from_values(vs, es ** -0.5)
    rw = rc @ _from_values(vr, er ** -0.5)
    left, singular, right = _svd(sw.T @ rw, trace)
    rotation = left @ right
    if np.linalg.slogdet(rotation)[0] < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    x = sw @ rotation
    pre = x @ _from_values(vr, np.sqrt(er)) + r.mean(axis=0)
    diag = {"method": "dense", "operation_shapes": trace,
            "cross_nuclear_norm": float(np.sum(singular))}
    if details:
        diag["rotation_orthogonality"] = close_error(rotation.T @ rotation, np.eye(d))
        diag["rotation_determinant_sign"] = float(np.linalg.slogdet(rotation)[0])
    return Alignment(pre, x if details else None, sw @ sw.T if details else None,
                     rw if details else None, diag)


def thin_reference(s, r, lam=1e-3, eps=1e-6, details=True):
    _parameters(lam, eps)
    s, r, k, _ = _pair(s, r, True)
    u, trace = helmert(k), []
    a0, b0 = u.T @ s, u.T @ r
    ls, sv, _ = _svd(a0, trace)
    lr, tv, vtr = _svd(b0, trace)
    _, si = _supported_values(sv ** 2, "source", k - 1)
    _, ri = _supported_values(tv ** 2, "receiver", k - 1)
    sa = np.sqrt(np.maximum(sv ** 2 / k + lam, eps))
    tb = np.sqrt(np.maximum(tv ** 2 / k + lam, eps))
    a, b = sv / sa, tv / tb
    small = (a[:, None] * ls.T) @ lr * b[None, :]
    p, cross, qt = _svd(small, trace)
    _, qi = _supported_values(cross ** 2, "cross", k - 1)
    # No reduced determinant correction: an unused ambient axis supplies it.
    rot = p @ qt
    left = u @ (ls * a[None, :])
    pre = (left @ (rot * tb[None, :])) @ vtr + r.mean(axis=0)
    diag = {"method": "thin", "source": si, "receiver": ri, "cross": qi,
            "operation_shapes": trace}
    return Alignment(pre, left @ rot @ vtr if details else None,
                     left @ left.T if details else None,
                     (u @ (lr * b[None, :])) @ vtr if details else None, diag)


def _receiver(r, lam, eps, trace):
    k = len(r)
    rc = r - r.mean(axis=0)
    gr = rc @ rc.T
    vals, vecs, info = _psd(gr, "receiver", trace)
    denom = np.maximum(vals / k + lam, eps)
    yw = _from_values(vecs, denom ** -0.5) @ rc
    return rc, vals, vecs, yw, info


def _dewhite(x, rc, values, vectors, lam, eps):
    k = len(rc)
    base = max(lam, eps)
    beta = np.sqrt(base)
    shifted = np.maximum(values / k + lam, eps)
    h = np.zeros_like(values)
    if lam >= eps:
        # No subtractive cancellation, including at t=0.
        h[:] = 1.0 / (k * (np.sqrt(shifted) + np.sqrt(lam)))
    else:
        nz = values > 0
        # Below the floor the numerator is exactly zero.
        h[nz] = (shifted[nz] - base) / (
            values[nz] * (np.sqrt(shifted[nz]) + beta))
    return beta * x + (x @ rc.T) @ _from_values(vectors, h) @ rc


def _residual_factor(dmat, count):
    k = len(dmat)
    f = np.zeros((k, count), dtype=np.float64)
    if count == 0:
        check = close_error(dmat, np.zeros_like(dmat))
        if not check["passed"]:
            raise Unsupported("unexpected_nonzero_completion_residual", check)
        return f
    for j in range(count):
        diag = np.diag(dmat) - np.sum(f[:, :j] ** 2, axis=1)
        p = int(np.argmax(diag))  # first index on an exact tie
        scale = max(1.0, float(np.max(np.diag(dmat), initial=0.0)))
        if diag[p] <= RANK_TOL * scale:
            raise Unsupported("completion_pivot_too_small", {"pivot": float(diag[p]), "step": j})
        f[:, j] = (dmat[:, p] - f[:, :j] @ f[p, :j]) / np.sqrt(diag[p])
    check = close_error(f @ f.T, dmat)
    if not check["passed"]:
        raise Unsupported("completion_factor_residual", check)
    return f


def _receiver_axes(vq, sender_rank, count):
    d = vq.shape[0]
    if count == 0:
        return np.empty((d, 0), dtype=np.float64)
    accepted = []
    for index in range(sender_rank):
        axis = np.zeros(d, dtype=np.float64)
        axis[index] = 1.0
        axis -= vq @ vq[index, :]
        for _ in range(2):
            for w in accepted:
                axis -= w * np.dot(w, axis)
        norm = float(np.linalg.norm(axis))
        if norm <= AXIS_TOL:
            continue
        axis /= norm
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1.0
        accepted.append(axis)
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise Unsupported("receiver_axis_completion_failed",
                          {"needed": count, "found": len(accepted)})
    w = np.column_stack(accepted)
    if not close_error(w.T @ w, np.eye(count))["passed"]:
        raise Unsupported("receiver_axes_not_orthonormal")
    if not close_error(vq.T @ w, np.zeros((vq.shape[1], count)))["passed"]:
        raise Unsupported("receiver_axes_not_orthogonal_to_active_span")
    return w


def align_from_gram(ga, r, method="dual", lam=1e-3, eps=1e-6, details=True):
    """Decode from reduced Gram and receiver rows; there is no sender input."""
    _parameters(lam, eps)
    r = np.asarray(r, dtype=np.float64)
    ga = np.asarray(ga, dtype=np.float64)
    if r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError("finite reference rows required")
    k, d = r.shape
    if not (2 <= k < d):
        raise Unsupported("outside_2_le_K_lt_d", {"K": k, "d": d})
    if ga.shape != (k - 1, k - 1) or not np.isfinite(ga).all():
        raise ValueError("finite reduced Gram of shape (K-1,K-1) required")
    if method not in {"dual", "canonical"}:
        raise ValueError("unknown Gram alignment method")
    u, trace = helmert(k), []
    gs, vs, si = _psd(ga, "source", trace, k - 1 if method == "dual" else None)
    fvals = gs / np.maximum(gs / k + lam, eps)
    gw_reduced = _from_values(vs, fvals)
    if method == "dual":
        b = u.T @ r
        gr, vr, ri = _psd(b @ b.T, "receiver", trace, k - 1)
        den = np.maximum(gr / k + lam, eps)
        target_gram = _from_values(vr, gr / den)
        xs = _from_values(vs, np.sqrt(fvals))
        mid = xs @ target_gram @ xs
        mv, me, qi = _psd(mid, "cross", trace, k - 1)
        m = xs @ _from_values(me, mv ** -0.5) @ xs
        pre = u @ m @ b + r.mean(axis=0)
        yw = u @ _from_values(vr, den ** -0.5) @ b if details else None
        diag = {"method": method, "source": si, "receiver": ri, "cross": qi,
                "operation_shapes": trace, "unique": True}
        return Alignment(pre, u @ m @ (u.T @ yw) if details else None,
                         u @ gw_reduced @ u.T if details else None, yw, diag)

    g = u @ gw_reduced @ u.T
    sqrtg = u @ _from_values(vs, np.sqrt(fvals)) @ u.T
    rc, rv, re, yw, ri = _receiver(r, lam, eps, trace)
    cu, cs, cvt = _svd(sqrtg @ yw, trace)
    _, qi = _supported_values(cs ** 2, "cross")
    rank, q = si["rank"], qi["rank"]
    if q > rank:
        raise Unsupported("cross_rank_exceeds_sender_rank", {"rank": rank, "q": q})
    uq, vq = cu[:, :q], cvt[:q, :].T
    parallel = sqrtg @ uq @ vq.T
    residual = (g - parallel @ parallel.T)
    residual = (residual + residual.T) * 0.5
    f = _residual_factor(residual, rank - q)
    w = _receiver_axes(vq, rank, rank - q)
    x = parallel + f @ w.T
    pre = _dewhite(x, rc, rv, re, lam, eps) + r.mean(axis=0)
    diag = {"method": method, "source": si, "receiver": ri, "cross": qi,
            "operation_shapes": trace, "unique": bool(q == rank),
            "completion_rank": rank - q,
            "completion_factor_error": close_error(f @ f.T, residual),
            "cross_nuclear_norm": float(np.sum(cs[:q]))}
    if details:
        diag["completion_target_orthogonality"] = close_error(f.T @ yw,
                                                            np.zeros((rank - q, d)))
    return Alignment(pre, x if details else None, g if details else None,
                     yw if details else None, diag)


def dual_alignment(s, r, lam=1e-3, eps=1e-6, details=True):
    s, r, k, _ = _pair(s, r, True)
    a = helmert(k).T @ s
    return align_from_gram(a @ a.T, r, "dual", lam, eps, details)


def canonical_alignment(s, r, lam=1e-3, eps=1e-6, details=True):
    s, r, k, _ = _pair(s, r, True)
    a = helmert(k).T @ s
    return align_from_gram(a @ a.T, r, "canonical", lam, eps, details)


def postprocess(pre, vocab, alpha=0.3):
    pre = np.asarray(pre, dtype=np.float64)
    vocab = np.asarray(vocab, dtype=np.float64)
    if (pre.ndim != 2 or vocab.ndim != 2 or pre.shape[1] != vocab.shape[1]
            or len(vocab) < 1 or not np.isfinite(pre).all()
            or not np.isfinite(vocab).all() or not 0 <= alpha <= 1):
        raise ValueError("invalid finite postprocessing inputs")
    target_norm = float(np.linalg.norm(vocab, axis=1).mean())
    norms = np.linalg.norm(pre, axis=1)
    calibrated = pre * (target_norm / np.maximum(norms, 1e-6))[:, None]
    qnorm = np.linalg.norm(calibrated, axis=1)
    vnorm = np.linalg.norm(vocab, axis=1)
    scores = (calibrated / np.maximum(qnorm, 1e-12)[:, None]) @ (
        vocab / np.maximum(vnorm, 1e-12)[:, None]).T
    ids = np.argmax(scores, axis=1)
    best = scores[np.arange(len(pre)), ids]
    if len(vocab) > 1:
        second = np.partition(scores, -2, axis=1)[:, -2]
        margins = best - second
    else:
        margins = np.full(len(pre), np.inf)
    final = (1.0 - alpha) * calibrated + alpha * vocab[ids]
    return {"final": final, "calibrated": calibrated, "ids": ids,
            "margins": margins, "pre_norms": norms, "query_norms": qnorm,
            "vocab_norms": vnorm, "target_norm": target_norm}


def final_nearest_ids(prefix, vocab):
    """Diagnostic lookup, not a requirement of a direct-prefix recipient."""
    p = np.asarray(prefix, dtype=np.float64)
    v = np.asarray(vocab, dtype=np.float64)
    pn = p / np.maximum(np.linalg.norm(p, axis=1), 1e-12)[:, None]
    vn = v / np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
    return np.argmax(pn @ vn.T, axis=1)


ALGORITHMS = {"dense": dense_reference, "thin": thin_reference,
              "dual": dual_alignment, "canonical": canonical_alignment}
