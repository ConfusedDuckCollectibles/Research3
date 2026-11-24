"""Fixed, source-free inverse for the contextual-information assay.

Only recover_final(final, public) performs recovery. Its input boundary
contains the received prefix and declared public quantities, never evaluation
truth or captured forward locals. Other functions compute explicitly separate
evaluation targets or exact pair witnesses. This module has no model, Torch,
filesystem, network, training, or timing dependencies.

The fixed assay uses lambda >= covariance_floor. It rejects unsupported floor
branches, out-of-domain spectra and numerical failures without clipping or
trying another reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np


U64 = np.finfo(np.float64).eps
SCHEMA_VERSION = 1


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _matrix64(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape, default=0) < 1:
        raise ValueError(name + " must be a nonempty matrix")
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _readonly(value, dtype=np.float64):
    """An owned immutable byte backing, not a view of a caller's mutable array."""
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _ids(value, vocabulary_size):
    result = np.asarray(value)
    if result.ndim != 1 or len(result) < 2 or result.dtype.kind not in "iu":
        raise ValueError("probe_ids must contain at least two integer IDs")
    if np.any(result < 0) or np.any(result >= vocabulary_size):
        raise ValueError("probe ID is outside the public vocabulary")
    return _readonly(result, np.int64)


def _json_number(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _json_vector(value):
    return [_json_number(x) for x in np.asarray(value).reshape(-1)]


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def helmert(k):
    """The exact fixed construction, evaluated in binary64."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("K must be an integer at least two")
    result = np.zeros((int(k), int(k)-1), dtype=np.float64)
    for j in range(int(k)-1):
        denominator = np.sqrt((j+1)*(j+2))
        result[:j+1, j] = 1.0 / denominator
        result[j+1, j] = -(j+1) / denominator
    return result


def _rank_from_singular(values, shape, solver_rank=None):
    values = np.asarray(values, dtype=np.float64)
    rcond = float(U64 * max(shape))
    largest = float(values[0]) if values.size else 0.0
    cutoff = rcond * largest
    rank = int(np.count_nonzero(values > cutoff))
    smallest = float(values[-1]) if values.size else 0.0
    condition = largest / smallest if smallest > 0 else math.inf
    result = {
        "status": "computed",
        "shape": [int(x) for x in shape],
        "singular_values": _json_vector(values),
        "rcond": rcond,
        "cutoff": _json_number(cutoff),
        "rank": rank,
        "condition_number": _json_number(condition),
        "condition_finite": math.isfinite(condition),
        "strict_rank_rule": "singular_value > rcond * largest_singular_value",
    }
    if solver_rank is not None:
        result["solver_rank"] = int(solver_rank)
    return result


def rank_diagnostics(matrix):
    """Fixed binary64 SVD rank convention, with JSON-safe failure diagnostics."""
    try:
        array = _matrix64(matrix, "rank input")
        values = np.linalg.svd(array, compute_uv=False)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite singular values")
        return _rank_from_singular(values, array.shape)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "failed", "rank": None, "condition_number": None,
                "condition_finite": False, "reason": type(exc).__name__ + ": " + str(exc)}


def array_errors(actual, reference):
    """Evaluation-only metrics; no tolerance or outcome-dependent decision.

    Undefined relative error is None, never zero. Zero/zero has relative
    error zero only when the actual difference is exactly zero.
    """
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    base = {
        "status": "invalid",
        "reason": None,
        "finite": False,
        "absolute_frobenius": None,
        "relative_frobenius": None,
        "maximum_absolute_coordinate": None,
        "reference_norm": None,
        "actual_norm": None,
        "norm_change": None,
    }
    if a.shape != b.shape:
        return {**base, "reason": "shape_mismatch"}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {**base, "reason": "nonfinite_input"}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = a-b
        absolute = float(np.linalg.norm(difference))
        bn = float(np.linalg.norm(b))
        an = float(np.linalg.norm(a))
        maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
        relative = absolute/bn if bn > 0 else (0.0 if np.all(difference == 0) else math.inf)
        change = an-bn
    finite = all(math.isfinite(x) for x in (absolute, bn, an, maximum, relative, change))
    return {
        **base,
        "status": "computed" if finite else "nonfinite_metric",
        "reason": None if finite else ("zero_reference_norm" if bn == 0 else "nonfinite_metric"),
        "finite": finite,
        "absolute_frobenius": _json_number(absolute),
        "relative_frobenius": _json_number(relative),
        "maximum_absolute_coordinate": _json_number(maximum),
        "reference_norm": _json_number(bn),
        "actual_norm": _json_number(an),
        "norm_change": _json_number(change),
    }


@dataclass(frozen=True, eq=False)
class PublicContext:
    """Declared public data, promoted to binary64 without recomputing its values.

    In public32, callers independently construct mean/root/target_norm/scalars
    in the producing Torch float32 convention before private extraction. The
    vocabulary directions are always recomputed here in float64. No captured
    private local, source matrix, answer, or true anchor has a field.
    """

    vocab: np.ndarray = field(repr=False)
    probe_ids: np.ndarray = field(repr=False)
    mean: np.ndarray = field(repr=False)
    root: np.ndarray = field(repr=False)
    target_norm: float
    lam: float
    anchor_gain: float
    complement: float
    covariance_floor: float = 1e-6
    row_floor: float = 1e-6
    cosine_floor: float = 1e-12
    convention: str = "public64"
    reference: np.ndarray = field(init=False, repr=False)
    vocab_norms: np.ndarray = field(init=False, repr=False)
    directions: np.ndarray = field(init=False, repr=False)
    basis: np.ndarray = field(init=False, repr=False)
    _metadata: dict = field(init=False, repr=False)

    def __post_init__(self):
        vocab = _readonly(_matrix64(self.vocab, "vocabulary"))
        ids = _ids(self.probe_ids, len(vocab))
        d = vocab.shape[1]
        mean = np.asarray(self.mean, dtype=np.float64)
        if mean.shape == (1, d):
            mean = mean[0]
        if mean.shape != (d,) or not np.isfinite(mean).all():
            raise ValueError("public mean must be a finite vector of width d")
        root = _matrix64(self.root, "public root")
        if root.shape != (d, d):
            raise ValueError("public root must have shape (d,d)")
        params = {}
        for name in ("target_norm", "lam", "anchor_gain", "complement",
                     "covariance_floor", "row_floor", "cosine_floor"):
            params[name] = _finite_number(getattr(self, name), name)
        if params["target_norm"] <= 0 or params["lam"] <= 0:
            raise ValueError("target norm and lambda must be positive")
        if not 0 <= params["anchor_gain"] < 1 or params["complement"] <= 0:
            raise ValueError("anchor gain must be in [0,1) and complement positive")
        if params["covariance_floor"] < 0 or params["row_floor"] <= 0 or params["cosine_floor"] <= 0:
            raise ValueError("invalid public floors")
        if not isinstance(self.convention, str) or not self.convention:
            raise ValueError("public convention must be a nonempty string")
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(vocab, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError("public vocabulary norms are not finite")
        directions = vocab / np.maximum(norms, params["cosine_floor"])[:, None]
        reference = vocab[ids]
        centered_reference = reference-reference.mean(axis=0)
        for name, value in params.items():
            object.__setattr__(self, name, value)
        for name, value, dtype in (
            ("vocab", vocab, np.float64), ("probe_ids", ids, np.int64),
            ("mean", mean, np.float64), ("root", root, np.float64),
            ("reference", reference, np.float64), ("vocab_norms", norms, np.float64),
            ("directions", directions, np.float64), ("basis", helmert(len(ids)), np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype))
        tiny = (norms > 0) & (norms < params["cosine_floor"])
        object.__setattr__(self, "_metadata", {
            "schema_version": SCHEMA_VERSION,
            "convention": self.convention,
            "K": len(ids), "d": d, "vocabulary_size": len(vocab),
            "probe_ids": ids.tolist(),
            **params,
            "affine_gain_sum": params["anchor_gain"] + params["complement"],
            "vocabulary_zero_rows": int(np.count_nonzero(norms == 0)),
            "vocabulary_nonzero_below_floor_rows": int(np.count_nonzero(tiny)),
            "anchor_geometry_supported": not bool(np.any(tiny)),
            "vocabulary_direction_dtype": "float64_recomputed_from_public_values",
            "root": rank_diagnostics(root),
            "reference": rank_diagnostics(reference),
            "centered_reference": rank_diagnostics(centered_reference),
        })

    @property
    def k(self):
        return len(self.probe_ids)

    @property
    def d(self):
        return self.vocab.shape[1]

    def metadata(self):
        return _json_copy(self._metadata)


def public64(vocab, probe_ids, lam=1e-3, covariance_floor=1e-6,
             row_floor=1e-6, cosine_floor=1e-12, alpha=0.3):
    """Public-only NumPy path matching the frozen dense reference grouping."""
    table = _matrix64(vocab, "vocabulary")
    ids = _ids(probe_ids, len(table))
    lam = _finite_number(lam, "lambda")
    covariance_floor = _finite_number(covariance_floor, "covariance floor")
    alpha = _finite_number(alpha, "alpha")
    rows = table[ids]
    mean = rows.mean(axis=0)
    centered = rows-mean
    covariance = centered.T @ centered / len(ids) + lam*np.eye(table.shape[1])
    values, vectors = np.linalg.eigh((covariance+covariance.T)*0.5)
    values = np.maximum(values, covariance_floor)
    root = (vectors*np.sqrt(values)[None, :]) @ vectors.T
    target = float(np.linalg.norm(table, axis=1).mean())
    return PublicContext(table, ids, mean, root, target, lam, alpha, 1.0-alpha,
                         covariance_floor, row_floor, cosine_floor, "public64")


def inverse_spectrum(values, k, lam, covariance_floor=1e-6):
    """Return (recovered_values, None), or (None, fixed_reason).

    No input eigenvalue is clipped, truncated, or changed. The fixed assay
    supports only the unfloored lambda >= covariance-floor branch.
    """
    try:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 1:
            return None, "invalid_spectrum_shape"
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2:
            return None, "invalid_spectral_parameters"
        lam = _finite_number(lam, "lambda")
        floor = _finite_number(covariance_floor, "covariance floor")
        if lam <= 0 or floor < 0:
            return None, "invalid_spectral_parameters"
        if lam < floor:
            return None, "unsupported_covariance_floor_branch"
        if not np.isfinite(x).all():
            return None, "nonfinite_spectrum"
        if np.any(x < 0) or np.any(x >= k):
            return None, "spectral_domain"
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values_out = k*lam*x/(k-x)
        if not np.isfinite(values_out).all():
            return None, "nonfinite_reconstruction"
        return values_out, None
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_spectral_parameters"


def _select_anchors(rows, public):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(rows, axis=1)
        normalized = rows/np.maximum(norms, public.cosine_floor)[:, None]
        scores = normalized @ public.directions.T
    if not np.isfinite(norms).all() or not np.isfinite(scores).all():
        raise ValueError("nonfinite anchor norms or scores")
    ids = np.argmax(scores, axis=1)  # first index on an exact tie
    best = scores[np.arange(len(rows)), ids]
    if scores.shape[1] > 1:
        second = np.partition(scores, -2, axis=1)[:, -2]
        margins = best-second
    else:
        margins = np.full(len(rows), math.inf, dtype=np.float64)
    return ids.astype(np.int64), margins, best, norms


def recover_final(final, public):
    """The sole source-free recovery entry point; no truth arguments exist.

    Status 'estimated' means a finite estimate is available. It does NOT mean
    recovery accuracy, semantic preservation, or a successful scientific gate.
    Arrays and diagnostics are retained through the first rejecting stage.
    """
    arrays = {}
    diagnostics = {"schema_version": SCHEMA_VERSION, "stage": "input",
                   "not_a_fidelity_claim": True}

    def finish(status, reason=None, gram=None):
        return {"status": status, "reason": reason, "gram": gram,
                "arrays": arrays, "diagnostics": diagnostics}

    if not isinstance(public, PublicContext):
        return finish("rejected", "invalid_public_context")
    diagnostics["public"] = public.metadata()
    try:
        received = _matrix64(final, "final prefix")
    except (ValueError, TypeError, OverflowError) as exc:
        diagnostics["error"] = type(exc).__name__ + ": " + str(exc)
        return finish("rejected", "invalid_final")
    if received.shape != (public.k, public.d):
        diagnostics["received_shape"] = list(received.shape)
        return finish("rejected", "final_shape")
    diagnostics["received_dtype"] = str(np.asarray(final).dtype)
    if not public._metadata["anchor_geometry_supported"]:
        return finish("unsupported", "tiny_vocabulary_rows")
    if public.lam < public.covariance_floor:
        return finish("unsupported", "unsupported_covariance_floor_branch")

    diagnostics["stage"] = "anchor"
    try:
        ids, margins, best, norms = _select_anchors(received, public)
        arrays["anchor_ids"] = ids
        arrays["anchor_margins"] = margins
        diagnostics["anchor_ids"] = ids.tolist()
        diagnostics["anchor_margins"] = _json_vector(margins)
        diagnostics["anchor_best_scores"] = _json_vector(best)
        diagnostics["received_row_norms"] = _json_vector(norms)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            calibrated = (received-public.anchor_gain*public.vocab[ids])/public.complement
        arrays["calibrated"] = calibrated
        if not np.isfinite(calibrated).all():
            return finish("rejected", "nonfinite_calibrated")
        check_ids, check_margins, check_best, calibrated_norms = _select_anchors(calibrated, public)
        arrays["calibrated_anchor_ids"] = check_ids
        arrays["calibrated_anchor_margins"] = check_margins
        diagnostics["calibrated_anchor_ids"] = check_ids.tolist()
        diagnostics["calibrated_anchor_margins"] = _json_vector(check_margins)
        diagnostics["calibrated_anchor_best_scores"] = _json_vector(check_best)
        diagnostics["calibrated_row_norms"] = _json_vector(calibrated_norms)
        diagnostics["anchor_self_consistent"] = bool(np.array_equal(ids, check_ids))
        if not diagnostics["anchor_self_consistent"]:
            return finish("rejected", "anchor_self_consistency")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics["error"] = type(exc).__name__ + ": " + str(exc)
        return finish("rejected", "anchor_computation")

    diagnostics["stage"] = "scale"
    design = calibrated.T
    target = public.k*public.mean
    rcond = float(U64*max(design.shape))
    try:
        scales, _, solver_rank, singular = np.linalg.lstsq(design, target, rcond=rcond)
        arrays["scales"] = scales
        arrays["scale_singular_values"] = singular
        scale_rank = _rank_from_singular(singular, design.shape, solver_rank)
        diagnostics["scale_solve"] = scale_rank
        with np.errstate(over="ignore", invalid="ignore"):
            residual = design @ scales-target
            residual_norm = float(np.linalg.norm(residual))
            denominator = float(singular[0])*float(np.linalg.norm(scales)) + float(np.linalg.norm(target))
        arrays["scale_mean_residual"] = residual
        diagnostics["mean_residual_norm"] = _json_number(residual_norm)
        diagnostics["mean_backward_error"] = _json_number(
            residual_norm/denominator if denominator > 0 else (0.0 if residual_norm == 0 else math.inf))
        diagnostics["scales"] = _json_vector(scales)
        diagnostics["minimum_scale"] = _json_number(np.min(scales))
        if not np.isfinite(scales).all() or not np.isfinite(singular).all():
            return finish("rejected", "nonfinite_scales")
        if scale_rank["rank"] != public.k or int(solver_rank) != public.k:
            return finish("rejected", "scale_rank")
        if np.any(scales <= 0):
            return finish("rejected", "nonpositive_scales")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics["error"] = type(exc).__name__ + ": " + str(exc)
        return finish("rejected", "scale_solve")

    diagnostics["stage"] = "rewhiten"
    with np.errstate(over="ignore", invalid="ignore"):
        pre = scales[:, None]*calibrated
        centered = pre-public.mean
        mean_residual = pre.mean(axis=0)-public.mean
    arrays["pre"] = pre
    arrays["centered_pre"] = centered
    arrays["pre_mean_residual"] = mean_residual
    diagnostics["pre_mean_residual_norm"] = _json_number(np.linalg.norm(mean_residual))
    if not np.isfinite(pre).all() or not np.isfinite(centered).all():
        return finish("rejected", "nonfinite_pre")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pre_norms = np.linalg.norm(pre, axis=1)
        recalibrated = pre*(public.target_norm/np.maximum(pre_norms, public.row_floor))[:, None]
    diagnostics["calibration_residual"] = array_errors(calibrated, recalibrated)
    diagnostics["reconstructed_pre_norms"] = _json_vector(pre_norms)
    try:
        # Deliberately generic: do not symmetrize or replace the actual public root.
        rewhitened = np.linalg.solve(public.root.T, centered.T).T
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics["error"] = type(exc).__name__ + ": " + str(exc)
        return finish("rejected", "root_solve")
    arrays["rewhitened"] = rewhitened
    if not np.isfinite(rewhitened).all():
        return finish("rejected", "nonfinite_rewhitening")
    rewhitened_mean = rewhitened.mean(axis=0)
    arrays["rewhitened_mean_residual"] = rewhitened_mean
    diagnostics["rewhitened_centering_residual_norm"] = _json_number(np.linalg.norm(rewhitened_mean))
    diagnostics["root_solve_residual"] = array_errors(rewhitened @ public.root, centered)

    diagnostics["stage"] = "spectrum"
    with np.errstate(over="ignore", invalid="ignore"):
        reduced = public.basis.T @ rewhitened
        white_gram = reduced @ reduced.T
        white_gram = (white_gram+white_gram.T)*0.5
    arrays["reduced_white"] = reduced
    arrays["white_gram"] = white_gram
    if not np.isfinite(white_gram).all():
        return finish("rejected", "nonfinite_white_gram")
    try:
        values, vectors = np.linalg.eigh(white_gram)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics["error"] = type(exc).__name__ + ": " + str(exc)
        return finish("rejected", "spectral_solve")
    arrays["white_eigenvalues"] = values
    arrays["white_eigenvectors"] = vectors
    diagnostics["white_eigenvalues"] = _json_vector(values)
    diagnostics["minimum_white_eigenvalue"] = _json_number(np.min(values))
    diagnostics["minimum_upper_domain_gap"] = _json_number(np.min(public.k-values))
    recovered, reason = inverse_spectrum(values, public.k, public.lam, public.covariance_floor)
    if reason is not None:
        return finish("rejected", reason)
    arrays["recovered_eigenvalues"] = recovered
    diagnostics["recovered_eigenvalues"] = _json_vector(recovered)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        diagnostics["inverse_relative_condition_factors"] = _json_vector(public.k/(public.k-values))
        gram = (vectors*recovered[None, :]) @ vectors.T
    if not np.isfinite(gram).all():
        return finish("rejected", "nonfinite_reconstruction")
    diagnostics["stage"] = "complete"
    return finish("estimated", gram=gram)


def source_grams(source):
    """Evaluation truth only: explicitly center stored values in float64."""
    array = _matrix64(source, "source")
    if len(array) < 2:
        raise ValueError("source must have at least two rows")
    centered = array-array.mean(axis=0)
    reduced_source = helmert(len(array)).T @ centered
    full = centered @ centered.T
    full = (full+full.T)*0.5
    reduced = reduced_source @ reduced_source.T
    reduced = (reduced+reduced.T)*0.5
    if not np.isfinite(full).all() or not np.isfinite(reduced).all():
        raise ValueError("source Gram is not finite")
    eigenvalues = np.linalg.eigvalsh(reduced)
    return {
        "reduced": reduced, "full": full, "centered": centered,
        "diagnostics": {
            "source": rank_diagnostics(array),
            "centered": rank_diagnostics(centered),
            "reduced_source": rank_diagnostics(reduced_source),
            "reduced_eigenvalues": _json_vector(eigenvalues),
            "reduced_gram_norm": _json_number(np.linalg.norm(reduced)),
            "full_gram_norm": _json_number(np.linalg.norm(full)),
            "source_mean": _json_vector(array.mean(axis=0)),
        },
    }


def _binary32(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(name + " must be a nonempty matrix")
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError(name + " must contain binary32 values; no implicit downcast")
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return np.ascontiguousarray(array, dtype="<f4")


def _dyadic_components(array):
    """Exact signed significand and power for every finite IEEE binary32 value."""
    words = array.view("<u4").reshape(-1)
    result = []
    for word0 in words:
        word = int(word0)
        exponent = (word >> 23) & 255
        fraction = word & 0x7fffff
        sign = -1 if word >> 31 else 1
        if exponent == 0:
            significand, power = fraction, -149
        else:
            significand, power = (1 << 23) | fraction, exponent-127-23
        result.append((sign*significand, power))
    return result


def _integer_matrix(components, shape, common_exponent):
    flat = [int(significand) << (power-common_exponent) if significand else 0
            for significand, power in components]
    width = shape[1]
    return [flat[i*width:(i+1)*width] for i in range(shape[0])]


def _center_integer(matrix):
    k, d = len(matrix), len(matrix[0])
    column_sums = [sum(matrix[i][j] for i in range(k)) for j in range(d)]
    return [[k*matrix[i][j]-column_sums[j] for j in range(d)] for i in range(k)]


def _integer_gram(matrix):
    return [[sum(x*y for x, y in zip(row, other)) for other in matrix] for row in matrix]


def _integer_hash(matrix, common_exponent, kind):
    """Canonical ASCII JSON; arbitrary integers are encoded as decimal strings."""
    payload = {
        "schema": "exact-binary32-centered-gram-v1",
        "kind": kind,
        "shape": [len(matrix), len(matrix[0])],
        "common_exponent": common_exponent,
        "integers": [[str(value) for value in row] for row in matrix],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def exact_centered_gram_pair(left_f32, right_f32):
    """Compare exact centered Grams of two finite stored binary32 matrices.

    Let e be the minimum nonzero IEEE significand power across both arrays,
    capped above by zero (or zero for an all-zero pair). Each source is A*2**e
    with integral A. Its exactly centered Gram is (T@T.T)*2**(2*e)/K**2,
    where T=K*A-column_sums(A). All arithmetic in T and its Gram is Python
    arbitrary-precision integer arithmetic. Signed zero affects raw_bits_equal
    but not numeric_values_equal or gram_equal.

    The hashes identify canonical integer representations for this common
    exponent; they are not claimed invariant across differently scaled pairs.
    Invalid dtype, shape or nonfinite values raise ValueError.
    """
    left = _binary32(left_f32, "left")
    right = _binary32(right_f32, "right")
    if left.shape != right.shape:
        raise ValueError("paired source shapes differ")
    left_components = _dyadic_components(left)
    right_components = _dyadic_components(right)
    powers = [power for significand, power in left_components+right_components if significand]
    common = min([0]+powers)
    li = _integer_matrix(left_components, left.shape, common)
    ri = _integer_matrix(right_components, right.shape, common)
    lt, rt = _center_integer(li), _center_integer(ri)
    lg, rg = _integer_gram(lt), _integer_gram(rt)
    return {
        "schema_version": SCHEMA_VERSION,
        "shape": list(left.shape),
        "numeric_values_equal": bool(np.array_equal(left, right)),
        "raw_bits_equal": left.tobytes(order="C") == right.tobytes(order="C"),
        "gram_equal": lg == rg,
        "common_exponent": common,
        "denominator_k_squared": int(left.shape[0]**2),
        "denominator_power_of_two": -2*common,
        "integer_encoding": "sorted compact ASCII JSON; decimal-string integers; exact-binary32-centered-gram-v1",
        "left_source_integer_sha256": _integer_hash(li, common, "source"),
        "right_source_integer_sha256": _integer_hash(ri, common, "source"),
        "left_centered_integer_sha256": _integer_hash(lt, common, "centered_numerator"),
        "right_centered_integer_sha256": _integer_hash(rt, common, "centered_numerator"),
        "left_gram_integer_sha256": _integer_hash(lg, common, "gram_numerator"),
        "right_gram_integer_sha256": _integer_hash(rg, common, "gram_numerator"),
    }
