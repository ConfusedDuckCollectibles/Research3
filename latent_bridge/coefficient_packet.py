"""Coefficient-packet study v2: codecs only, with no transport or fixture oracle.

The fixed design is docs/coefficient_packet_design.md at 49cfba7261326d58e174e51c8ddf5abba6f4a966.
The original alignment modules are imported unchanged.  All encoder decisions use
the actual input and emitted numerical payload; no expected prefix/hash is an input.
"""
from dataclasses import dataclass, field
import hashlib
import json
import struct
import time

import numpy as np

from . import alignment_core as core
from . import alignment_packet as original

ARMS = ("coefficient", "dual_direct", "thin_direct",
        "gram_guard", "gram_original", "token_lookup")
HEADER = original.HEADER
MAGIC = original.MAGIC
VERSION = original.VERSION
COEFFICIENT_MODE = 6
MAX_PAYLOAD = 16 * 1024 * 1024
MAX_SETUP_JSON = 4096
UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
DESIGN_COMMIT = "49cfba7261326d58e174e51c8ddf5abba6f4a966"
CORE_COMMIT = "7aee330f7799751a6a9f4b2a8decb612abe9c266"

PacketError = original.PacketError
Unsupported = core.Unsupported
Encoded = original.Encoded
Decoded = original.Decoded
DirectContext = original.DirectContext


def _arm(arm):
    if type(arm) is not str or arm not in ARMS:
        raise PacketError("unknown study arm")
    return arm


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _immutable(array):
    a = np.ascontiguousarray(array, dtype="<f8")
    return np.frombuffer(a.tobytes(), dtype="<f8").reshape(a.shape)


@dataclass(frozen=True)
class SessionV2(original.Session):
    """Immutable table/configuration; caches affect cost, never session identity."""
    cache_post: bool = False
    normalized_vocab: np.ndarray | None = field(init=False, default=None, repr=False)
    vocab_norms: np.ndarray | None = field(init=False, default=None, repr=False)
    target_norm: float | None = field(init=False, default=None)

    def __post_init__(self):
        if type(self.cache_post) is not bool:
            raise PacketError("cache_post must be bool")
        super().__post_init__()
        manifest = json.loads(self.manifest_bytes)
        manifest.update({
            "algorithm_version": "coefficient_packet_v2",
            "session_version": 2,
            "numerical_core_commit": CORE_COMMIT,
            "coefficient_mode": COEFFICIENT_MODE,
            "basis": "helmert_j0:first_j+1_positive;row_j+1_negative;denominator_sqrt((j+1)*(j+2))",
            "reconstruction_grouping": "(H @ M) @ (H.T @ R) + mean(R)",
            "coefficient_symmetrization": "(M + M.T) * 0.5 before lower_triangle",
            "triangle_order": "numpy.tril_indices(K-1),row_major",
            "wire_precisions": {"float64": "<f8", "float32": "<f4",
                                "token_ids": "<u4", "decoded_compute": "<f8"},
            "wire_modes": {"1": "dual_gram", "3": "raw_fallback",
                           "4": "direct", "5": "token_lookup", "6": "coefficient"},
        })
        raw = _canonical(manifest)
        object.__setattr__(self, "manifest_bytes", raw)
        object.__setattr__(self, "digest", hashlib.sha256(raw).digest())
        if self.cache_post:
            norms = np.linalg.norm(self.vocab, axis=1)
            target = float(norms.mean())
            unit = self.vocab / np.maximum(norms, 1e-12)[:, None]
            if not np.isfinite(norms).all() or not np.isfinite(target) or not np.isfinite(unit).all():
                raise PacketError("nonfinite postprocessing cache")
            object.__setattr__(self, "vocab_norms", _immutable(norms))
            object.__setattr__(self, "normalized_vocab", _immutable(unit))
            object.__setattr__(self, "target_norm", target)


def prepare_sender(vocab, arm, lam=1e-3, eps=1e-6, alpha=0.3):
    _arm(arm)
    return SessionV2(vocab, lam=lam, eps=eps, alpha=alpha,
                     cache_post=arm in {"dual_direct", "thin_direct"})


def prepare_receiver(sender_session, arm):
    _arm(arm)
    if not isinstance(sender_session, SessionV2):
        raise PacketError("sender SessionV2 required")
    if arm in {"dual_direct", "thin_direct"}:
        return sender_session.direct_context
    return SessionV2(sender_session.vocab, lam=sender_session.lam,
                     eps=sender_session.eps, alpha=sender_session.alpha,
                     cache_post=arm in {"coefficient", "gram_guard"})


def postprocess_v2(pre, context):
    """Same formula as frozen postprocess; only invariant table work is cached."""
    if not isinstance(context, SessionV2):
        raise PacketError("postprocessing requires SessionV2")
    if not context.cache_post:
        return core.postprocess(pre, context.vocab, context.alpha)
    pre = np.asarray(pre, dtype=np.float64)
    if (pre.ndim != 2 or pre.shape[1] != context.d or len(pre) < 1 or
            not np.isfinite(pre).all()):
        raise PacketError("invalid finite postprocessing input")
    norms = np.linalg.norm(pre, axis=1)
    calibrated = pre * (context.target_norm / np.maximum(norms, 1e-6))[:, None]
    qnorm = np.linalg.norm(calibrated, axis=1)
    scores = (calibrated / np.maximum(qnorm, 1e-12)[:, None]) @ context.normalized_vocab.T
    if not np.isfinite(scores).all():
        raise PacketError("nonfinite postprocessing scores")
    ids = np.argmax(scores, axis=1)
    best = scores[np.arange(len(pre)), ids]
    margins = best - np.partition(scores, -2, axis=1)[:, -2] if len(context.vocab) > 1 else (
        np.full(len(pre), np.inf))
    final = (1.0 - context.alpha) * calibrated + context.alpha * context.vocab[ids]
    if not np.isfinite(final).all():
        raise PacketError("nonfinite postprocessed output")
    return {"final": final, "calibrated": calibrated, "ids": ids, "margins": margins,
            "pre_norms": norms, "query_norms": qnorm, "vocab_norms": context.vocab_norms,
            "target_norm": context.target_norm}


def _post_metadata(post):
    return {"snap_ids": post["ids"].tolist(), "snap_margins": post["margins"].tolist(),
            "pre_norms": post["pre_norms"].tolist(),
            "query_norms": post["query_norms"].tolist(),
            "target_norm": float(post["target_norm"])}


def _timed(phases, name, function):
    started = time.perf_counter()
    try:
        return function()
    finally:
        phases[name] = phases.get(name, 0.0) + time.perf_counter() - started


def _precision(precision):
    if type(precision) is not str or precision not in original.PRECISIONS:
        raise PacketError("unknown floating-point precision")
    code = original.PRECISIONS[precision]
    return code, original.DTYPES[code]


def _float_bytes(values, dtype, reason):
    with np.errstate(over="ignore", invalid="ignore"):
        cast = np.asarray(values, dtype=dtype, order="C")
    if not np.isfinite(cast).all():
        raise Unsupported(reason)
    return cast.tobytes(order="C")


def _matrix(a, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise Unsupported("invalid_guard_matrix")
    return a


def _cholesky(m):
    if not np.isfinite(m).all():
        raise Unsupported("coefficient_nonfinite")
    if m.ndim != 2 or m.shape[0] != m.shape[1] or len(m) < 1 or not np.array_equal(m, m.T):
        raise Unsupported("coefficient_not_symmetric")
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError as exc:
        raise Unsupported("coefficient_not_spd") from exc


def _finite_bound(value):
    if not np.isfinite(value) or value >= 1e100:
        raise Unsupported("prefix_range_bound", {"upper_bound": float(value)})
    return float(value)


def guard_coefficient(m_hat, m_symmetric, x, y, gb, mean, receiver_max,
                      precision="float64"):
    """Pure actual-rounded-M guard; no eigensolve, prefix expansion, or lookup."""
    _precision(precision)
    m_hat = np.asarray(m_hat, dtype=np.float64)
    _cholesky(m_hat)
    r = len(m_hat)
    shape = (r, r)
    m = _matrix(m_symmetric, shape)
    x, y, gb = (_matrix(a, shape) for a in (x, y, gb))
    mean = np.asarray(mean, dtype=np.float64)
    if mean.ndim != 1 or not np.isfinite(mean).all() or not np.array_equal(m, m.T):
        raise Unsupported("invalid_guard_reference")
    receiver_max = float(receiver_max)
    if not np.isfinite(receiver_max) or receiver_max < 0:
        raise Unsupported("invalid_receiver_maximum")
    residual = float(np.linalg.norm(m_hat @ y @ m_hat - x) /
                     max(1.0, float(np.linalg.norm(x))))
    residual_limit = 1e-8 if precision == "float64" else 1e-5
    diagnostic = {"decoded_cholesky": True, "normalized_residual": residual,
                  "residual_limit": residual_limit}
    if not np.isfinite(residual) or residual > residual_limit:
        raise Unsupported("coefficient_residual", diagnostic)

    # H's columns are orthogonal to the mean term: no full prefix is formed.
    terms = (m @ gb) * m
    mean_term = float((r + 1) * np.dot(mean, mean))
    squared = float(terms.sum()) + mean_term
    absolute_terms = float(np.abs(terms).sum()) + abs(mean_term)
    clip_tolerance = 64.0 * r * UNIT_ROUNDOFF * max(1.0, absolute_terms)
    if not np.isfinite([squared, absolute_terms]).all() or squared < -clip_tolerance:
        raise Unsupported("coefficient_scale_invalid",
                          {**diagnostic, "squared_scale": squared,
                           "negative_clip_tolerance": clip_tolerance})
    clipped = squared < 0
    z_norm = float(np.sqrt(max(0.0, squared)))
    delta = float(np.linalg.norm(m_hat - m) * np.sqrt(receiver_max))
    threshold = (1e-10 + 1e-8 * z_norm if precision == "float64"
                 else 1e-7 + 1e-5 * z_norm)
    trace_gb = float(np.trace(gb))
    if trace_gb < 0 or not np.isfinite(trace_gb):
        raise Unsupported("coefficient_scale_invalid", diagnostic)
    bound = _finite_bound(float(np.linalg.norm(m_hat)) * np.sqrt(trace_gb) +
                          np.sqrt(r + 1) * float(np.linalg.norm(mean)))
    diagnostic.update({"quantization_bound": delta, "quantization_limit": threshold,
                       "reference_prefix_frobenius_scale": z_norm,
                       "negative_scale_clipped": clipped,
                       "negative_clip_tolerance": clip_tolerance,
                       "prefix_norm_upper_bound": bound})
    if not np.isfinite(delta) or delta > threshold:
        raise Unsupported("coefficient_quantization_bound", diagnostic)
    return diagnostic


def guard_gram_spectrum(x_values, y_values, gb, mean):
    """Two-eigensolve caller supplies extrema; this helper never solves for M."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if (x.ndim != 1 or x.shape != y.shape or len(x) < 1 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(x <= 0) or np.any(y <= 0)):
        raise Unsupported("gram_invalid_spectral_values")
    r = len(x)
    gb = _matrix(gb, (r, r))
    mean = np.asarray(mean, dtype=np.float64)
    if mean.ndim != 1 or not np.isfinite(mean).all():
        raise Unsupported("gram_invalid_mean")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    lower, upper = xmin * ymin, xmax * ymax
    threshold = (core.AMBIGUITY_HI + 64.0 * r * UNIT_ROUNDOFF) * max(1.0, upper)
    diagnostic = {"cross_lower_bound": lower, "cross_upper_bound": upper,
                  "cross_required_lower": threshold, "sender_eigensolves": 2,
                  "sender_computed_coefficient": False}
    if not np.isfinite([lower, upper, threshold]).all() or not lower > threshold:
        raise Unsupported("gram_cross_bound", diagnostic)
    trace_gb = float(np.trace(gb))
    if not np.isfinite(trace_gb) or trace_gb < 0:
        raise Unsupported("gram_invalid_scale", diagnostic)
    m_norm_bound = float(np.sqrt(xmax / ymin))
    bound = _finite_bound(m_norm_bound * np.sqrt(trace_gb) +
                          np.sqrt(r + 1) * float(np.linalg.norm(mean)))
    diagnostic.update({"coefficient_spectral_norm_upper": m_norm_bound,
                       "prefix_norm_upper_bound": bound})
    return diagnostic


@dataclass
class _GramParts:
    h: np.ndarray
    b: np.ndarray
    mean: np.ndarray
    gb: np.ndarray
    source_values: np.ndarray
    source_vectors: np.ndarray
    receiver_values: np.ndarray
    receiver_vectors: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray
    diagnostics: dict
    trace: list


def _source_gram(s):
    h = core.helmert(len(s))
    a = h.T @ s
    ga = a @ a.T
    return h, (ga + ga.T) * 0.5


def _gram_parts(ga, r, context, h=None):
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2 or not np.isfinite(r).all():
        raise Unsupported("invalid_reference_rows")
    k, d = r.shape
    if not (2 <= k < d):
        raise Unsupported("outside_2_le_K_lt_d")
    if np.asarray(ga).shape != (k - 1, k - 1) or not np.isfinite(ga).all():
        raise Unsupported("source_not_finite_reduced_gram")
    h = core.helmert(k) if h is None else h
    trace = []
    gs, vs, si = core._psd(ga, "source", trace, k - 1)
    b = h.T @ r
    gb = b @ b.T
    gr, vr, ri = core._psd(gb, "receiver", trace, k - 1)
    xv = gs / np.maximum(gs / k + context.lam, context.eps)
    yv = gr / np.maximum(gr / k + context.lam, context.eps)
    if not np.isfinite(xv).all() or not np.isfinite(yv).all():
        raise Unsupported("nonfinite_whitened_spectrum")
    return _GramParts(h, b, r.mean(axis=0), gb, gs, vs, gr, vr, xv, yv,
                      {"source": si, "receiver": ri}, trace)


def _coefficient(parts):
    xs = core._from_values(parts.source_vectors, np.sqrt(parts.x_values))
    y = core._from_values(parts.receiver_vectors, parts.y_values)
    mid = xs @ y @ xs
    values, vectors, info = core._psd(mid, "cross", parts.trace, len(parts.x_values))
    m = xs @ core._from_values(vectors, values ** -0.5) @ xs
    parts.diagnostics["cross"] = info
    return m, y


def _expanded(m, parts):
    return (parts.h @ m) @ parts.b + parts.mean


def _float_packet(mode, code, k, d, payload, context):
    packet = original._frame(mode, code, k, d, payload, context)
    if len(packet) > MAX_PAYLOAD:
        raise PacketError("packet exceeds fixed maximum")
    return packet


def _input(s, token_ids, context):
    if not isinstance(context, SessionV2):
        raise PacketError("encoder requires SessionV2")
    s = np.asarray(s, dtype=np.float64)
    if (s.ndim != 2 or len(s) < 1 or s.shape[1] != context.d or
            not np.isfinite(s).all()):
        raise PacketError("invalid finite sender matrix")
    missing = token_ids is None or np.asarray(token_ids).size == 0
    ids = None if missing else original._ids(token_ids, len(s), context)
    return s, ids, None if missing else context.vocab[ids]


def _fallback(s, r, context, dtype, phases, metadata):
    result = _timed(phases, "dense_fallback",
                    lambda: core.dense_reference(s, r, context.lam, context.eps, details=False))
    if result.bypass:
        final = result.pre
        metadata["output_source"] = "raw_hidden_early_return"
    else:
        post = _timed(phases, "fallback_postprocess",
                      lambda: postprocess_v2(result.pre, context))
        final = post["final"]
        metadata.update(_post_metadata(post))
        metadata["output_source"] = "dense_reference"
    return _timed(phases, "fallback_serialization",
                  lambda: original._float_payload(final, dtype))


def encode_arm(s, token_ids, context, arm, precision="float64"):
    """Return actual bytes and small diagnostics, never an expected prefix/hash."""
    started = time.perf_counter()
    _arm(arm)
    if not isinstance(context, SessionV2):
        raise PacketError("encoder requires SessionV2")
    phases = {}
    metadata = {"arm": arm, "precision": precision, "fallback": False,
                "fallback_reason": None, "phase_seconds": phases,
                "sender_cache_post": context.cache_post}
    if arm == "token_lookup":
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or len(ids) < 1:
            raise PacketError("nonempty token IDs required")
        result = _timed(phases, "token_encode",
                        lambda: original.encode_token_only(ids, context))
        if len(result.packet) > MAX_PAYLOAD:
            raise PacketError("packet exceeds fixed maximum")
        metadata.update(result.metadata)
        metadata.update({"precision": "uint32", "route": "native",
                         "wire_mode": 5, "packet_bytes": len(result.packet),
                         "sender_total_encode_seconds": time.perf_counter() - started})
        return Encoded(result.packet, metadata)
    code, dtype = _precision(precision)
    if arm == "gram_original":
        result = _timed(phases, "frozen_gram_encode",
                        lambda: original.encode_message(s, token_ids, context, "dual", precision))
        if len(result.packet) > MAX_PAYLOAD:
            raise PacketError("packet exceeds fixed maximum")
        metadata.update(result.metadata)
        metadata.update({"arm": arm, "phase_seconds": phases,
                         "fallback_reason": result.metadata.get("fallback_reason"),
                         "wire_mode": 3 if result.metadata["fallback"] else 1,
                         "route": "raw_fallback" if result.metadata["fallback"] else "native",
                         "packet_bytes": len(result.packet),
                         "sender_total_encode_seconds": time.perf_counter() - started})
        return Encoded(result.packet, metadata)

    s, ids, r = _timed(phases, "input_and_reference",
                       lambda: _input(s, token_ids, context))
    k, d = s.shape
    metadata.update({"K": k, "d": d})
    try:
        if r is None:
            raise Unsupported("missing_tokens")
        if not (2 <= k < d):
            raise Unsupported("outside_2_le_K_lt_d")
        if arm == "thin_direct":
            aligned = _timed(phases, "thin_alignment",
                             lambda: core.thin_reference(s, r, context.lam,
                                                         context.eps, details=False))
            metadata["alignment"] = aligned.diagnostics
            post = _timed(phases, "postprocess",
                          lambda: postprocess_v2(aligned.pre, context))
            metadata.update(_post_metadata(post))
            payload = _timed(phases, "direct_serialization",
                             lambda: original._float_payload(post["final"], dtype))
            mode = 4
        else:
            h, ga = _timed(phases, "source_gram", lambda: _source_gram(s))
            if arm == "gram_guard":
                tri = np.tril_indices(k - 1)
                gram_bytes = _timed(phases, "gram_serialization",
                                   lambda: _float_bytes(ga[tri], dtype, "rounded_gram_nonfinite"))
                rounded = _timed(phases, "rounded_gram_materialization",
                                 lambda: original._reduced_from_triangle(
                                     np.frombuffer(gram_bytes, dtype=dtype).astype(np.float64), k))
                parts = _timed(phases, "source_target_eigensolves",
                               lambda: _gram_parts(rounded, r, context, h))
                metadata["guard"] = _timed(phases, "gram_spectral_guard",
                    lambda: guard_gram_spectrum(parts.x_values, parts.y_values,
                                                parts.gb, parts.mean))
                metadata["alignment"] = {**parts.diagnostics,
                                         "operation_shapes": parts.trace}
                payload, mode = ids.tobytes() + gram_bytes, 1
            else:
                parts = _timed(phases, "source_target_eigensolves",
                               lambda: _gram_parts(ga, r, context, h))
                m, y = _timed(phases, "coefficient_solve", lambda: _coefficient(parts))
                metadata["alignment"] = {**parts.diagnostics, "operation_shapes": parts.trace}
                if arm == "dual_direct":
                    # Reuse B, mean, H and the solve. No candidate-only guard.
                    pre = _timed(phases, "expansion", lambda: _expanded(m, parts))
                    post = _timed(phases, "postprocess", lambda: postprocess_v2(pre, context))
                    metadata.update(_post_metadata(post))
                    payload = _timed(phases, "direct_serialization",
                                     lambda: original._float_payload(post["final"], dtype))
                    mode = 4
                else:
                    sym = _timed(phases, "coefficient_symmetrization",
                                 lambda: (m + m.T) * 0.5)
                    tri = np.tril_indices(k - 1)
                    coefficient_bytes = _timed(phases, "coefficient_serialization",
                        lambda: _float_bytes(sym[tri], dtype, "coefficient_rounding_nonfinite"))
                    m_hat = _timed(phases, "rounded_coefficient_materialization",
                        lambda: original._reduced_from_triangle(
                            np.frombuffer(coefficient_bytes, dtype=dtype).astype(np.float64), k))
                    x = _timed(phases, "guard_source_gram",
                               lambda: core._from_values(parts.source_vectors, parts.x_values))
                    metadata["guard"] = _timed(phases, "coefficient_guard",
                        lambda: guard_coefficient(m_hat, sym, x, y, parts.gb, parts.mean,
                                                  float(parts.receiver_values.max()), precision))
                    payload, mode = ids.tobytes() + coefficient_bytes, COEFFICIENT_MODE
    except (Unsupported, np.linalg.LinAlgError) as exc:
        metadata.update({"fallback": True,
                         "fallback_reason": exc.reason if isinstance(exc, Unsupported)
                         else "numerical_factorization_failure",
                         "fallback_diagnostics": getattr(exc, "diagnostics", {})})
        payload = _fallback(s, r, context, dtype, phases, metadata)
        mode = 3

    packet = _timed(phases, "framing",
                    lambda: _float_packet(mode, code, k, d, payload, context))
    metadata.update({"route": "raw_fallback" if mode == 3 else "native",
                     "wire_mode": mode, "mode": "coefficient" if mode == 6 else original.MODES[mode],
                     "header_bytes": HEADER.size, "payload_bytes": len(payload),
                     "packet_bytes": len(packet),
                     "sender_total_encode_seconds": time.perf_counter() - started})
    return Encoded(packet, metadata)


def _header(packet, context, arm):
    if not isinstance(context, (SessionV2, DirectContext)):
        raise PacketError("unknown receiver context")
    if type(packet) is not bytes or not HEADER.size <= len(packet) <= MAX_PAYLOAD:
        raise PacketError("invalid packet bytes/length")
    magic, version, mode, code, flags, k, d, length, digest = HEADER.unpack_from(packet)
    expected_modes = {"coefficient": {3, 6}, "dual_direct": {3, 4},
                      "thin_direct": {3, 4}, "gram_guard": {1, 3},
                      "gram_original": {1, 3}, "token_lookup": {5}}
    if (magic != MAGIC or version != VERSION or flags != 0 or
            mode not in expected_modes[arm] or digest != context.digest):
        raise PacketError("packet format/mode/session mismatch")
    if k < 1 or d != context.d or length != len(packet) - HEADER.size:
        raise PacketError("packet dimensions/length mismatch")
    return mode, code, k, d, packet[HEADER.size:]


def decode_arm(packet, context, arm):
    """Receiver-only decoding; no private input, oracle, or expected output."""
    started = time.perf_counter()
    _arm(arm)
    mode, code, k, d, payload = _header(packet, context, arm)
    phases = {}
    metadata = {"arm": arm, "wire_mode": mode, "packet_bytes": len(packet),
                "header_bytes": HEADER.size, "payload_bytes": len(payload),
                "phase_seconds": phases, "receiver_accesses_sender": False,
                "fallback": mode == 3,
                "precision": "uint32" if mode == 5 else ("float64" if code == 1 else "float32")}
    if mode in {3, 4, 5} or arm == "gram_original":
        result = _timed(phases, "frozen_decode",
                        lambda: original.decode_packet(packet, context))
        metadata.update(result.metadata)
        metadata.update({"arm": arm, "wire_mode": mode, "phase_seconds": phases,
                         "decode_seconds": time.perf_counter() - started})
        return Decoded(result.final, result.mode, result.pre, metadata)

    if not isinstance(context, SessionV2):
        raise PacketError("coefficient/Gram decoding requires a shared table")
    if code not in original.DTYPES or not (2 <= k < d):
        raise PacketError("invalid numerical mode/dimensions")
    dtype = original.DTYPES[code]
    if len(payload) != 4 * k + dtype.itemsize * k * (k - 1) // 2:
        raise PacketError("numerical payload length mismatch")
    ids = np.frombuffer(payload[:4 * k], dtype="<u4")
    if np.any(ids >= len(context.vocab)):
        raise PacketError("unknown token ID")
    values = np.frombuffer(payload[4 * k:], dtype=dtype).astype(np.float64)
    if not np.isfinite(values).all():
        raise PacketError("nonfinite numerical payload")
    matrix = original._reduced_from_triangle(values, k)
    try:
        r = _timed(phases, "reference_lookup", lambda: context.vocab[ids])
        if mode == COEFFICIENT_MODE:
            _timed(phases, "coefficient_cholesky", lambda: _cholesky(matrix))
            def expand():
                h = core.helmert(k)
                b = h.T @ r
                return (h @ matrix) @ b + r.mean(axis=0)
            pre = _timed(phases, "expansion", expand)
        else:
            parts = _timed(phases, "source_target_eigensolves",
                           lambda: _gram_parts(matrix, r, context))
            m, _ = _timed(phases, "coefficient_solve", lambda: _coefficient(parts))
            metadata["alignment"] = {**parts.diagnostics, "operation_shapes": parts.trace}
            pre = _timed(phases, "expansion", lambda: _expanded(m, parts))
        if not np.isfinite(pre).all():
            raise Unsupported("nonfinite_decoder_prefix")
        post = _timed(phases, "postprocess", lambda: postprocess_v2(pre, context))
    except (Unsupported, ValueError, np.linalg.LinAlgError) as exc:
        raise PacketError("decoder guard rejection: " + str(exc)) from exc
    metadata.update(_post_metadata(post))
    metadata.update({"precision": "float64" if code == 1 else "float32",
                     "decode_seconds": time.perf_counter() - started})
    return Decoded(post["final"], "coefficient" if mode == 6 else "gram_guard", pre, metadata)


def encode_setup(sender_session, arm):
    """Exact cold setup bytes; the harness owns framing and SET2 ACK hashing."""
    _arm(arm)
    if not isinstance(sender_session, SessionV2):
        raise PacketError("setup requires sender SessionV2")
    common = {"version": 2, "arm": arm, "d": sender_session.d,
              "digest": sender_session.digest.hex()}
    if arm in {"dual_direct", "thin_direct"}:
        metadata = {**common, "kind": "direct_context", "table_bytes": 0}
        table = b""
    else:
        table = sender_session.vocab.tobytes(order="C")
        metadata = {**common, "kind": "table", "table_bytes": len(table),
                    "manifest": json.loads(sender_session.manifest_bytes)}
    raw = _canonical(metadata)
    if len(raw) > MAX_SETUP_JSON or 4 + len(raw) + len(table) > MAX_PAYLOAD:
        raise PacketError("setup exceeds fixed limits")
    return struct.pack("<I", len(raw)) + raw + table


def _bad_json_constant(value):
    raise PacketError("nonfinite JSON constant")


def decode_setup(payload, arm):
    """Build the receiver solely from the actual received setup payload."""
    _arm(arm)
    if type(payload) is not bytes or not 4 <= len(payload) <= MAX_PAYLOAD:
        raise PacketError("invalid setup bytes/length")
    json_length = struct.unpack_from("<I", payload)[0]
    if not 1 <= json_length <= MAX_SETUP_JSON or 4 + json_length > len(payload):
        raise PacketError("invalid setup JSON length")
    raw = payload[4:4 + json_length]
    try:
        meta = json.loads(raw.decode("utf-8"), parse_constant=_bad_json_constant)
        if type(meta) is not dict or _canonical(meta) != raw:
            raise PacketError("noncanonical setup JSON")
    except (ValueError, UnicodeError, TypeError) as exc:
        raise PacketError("invalid setup JSON") from exc
    direct = arm in {"dual_direct", "thin_direct"}
    keys = {"version", "arm", "d", "digest", "kind", "table_bytes"}
    if not direct:
        keys.add("manifest")
    if set(meta) != keys or meta["version"] != 2 or type(meta["version"]) is not int:
        raise PacketError("unknown setup schema/version")
    if meta["arm"] != arm or type(meta["d"]) is not int or meta["d"] < 1:
        raise PacketError("setup arm/dimension mismatch")
    if type(meta["digest"]) is not str or len(meta["digest"]) != 64:
        raise PacketError("invalid setup digest")
    try:
        digest = bytes.fromhex(meta["digest"])
    except ValueError as exc:
        raise PacketError("invalid setup digest") from exc
    if digest.hex() != meta["digest"]:
        raise PacketError("noncanonical setup digest")
    table = payload[4 + json_length:]
    if type(meta["table_bytes"]) is not int or meta["table_bytes"] != len(table):
        raise PacketError("setup table length mismatch")
    if direct:
        if meta["kind"] != "direct_context" or len(table) != 0:
            raise PacketError("direct setup must not supply a table")
        return DirectContext(meta["d"], digest)
    if meta["kind"] != "table" or type(meta["manifest"]) is not dict:
        raise PacketError("table setup required")
    try:
        manifest = meta["manifest"]
        shape = manifest["vocabulary"]["shape"]
        if (type(shape) is not list or len(shape) != 2 or
                any(type(v) is not int or v < 1 for v in shape) or
                shape[1] != meta["d"] or len(table) != shape[0] * shape[1] * 8 or
                manifest["vocabulary"]["dtype"] != "<f8"):
            raise PacketError("invalid setup table shape/dtype")
        vocab = np.frombuffer(table, dtype="<f8").reshape(tuple(shape))
        context = SessionV2(vocab, lam=manifest["lambda"], eps=manifest["epsilon"],
                            alpha=manifest["alpha"],
                            cache_post=arm in {"coefficient", "gram_guard"})
        if context.digest != digest or context.manifest_bytes != _canonical(manifest):
            raise PacketError("setup table/manifest digest mismatch")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PacketError("invalid table setup: " + str(exc)) from exc
    return context
