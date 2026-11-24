"""A fixed, explicitly costed Gram packet for the numerical study."""
from dataclasses import dataclass, field
import hashlib
import json
import struct
import time
import numpy as np
from .alignment_core import (Unsupported, helmert, align_from_gram,
                             dense_reference, postprocess)

HEADER = struct.Struct("<4sBBBBIIQ32s")
MAGIC = b"SBG1"
VERSION = 1
MODES = {1: "dual", 2: "canonical", 3: "raw_fallback", 4: "direct", 5: "token_only"}
DTYPES = {1: np.dtype("<f8"), 2: np.dtype("<f4")}
PRECISIONS = {"float64": 1, "float32": 2}
UPSTREAM_SHA256 = "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d"


class PacketError(ValueError):
    pass


@dataclass(frozen=True)
class DirectContext:
    """Only the shared fingerprint and width; no embedding table is retained."""
    d: int
    digest: bytes

    def __post_init__(self):
        if (not isinstance(self.d, int) or self.d < 1 or
                not isinstance(self.digest, bytes) or len(self.digest) != 32):
            raise PacketError("invalid direct context")


@dataclass(frozen=True)
class Session:
    vocab: np.ndarray
    lam: float = 1e-3
    eps: float = 1e-6
    alpha: float = 0.3
    manifest_bytes: bytes = field(init=False, repr=False)
    digest: bytes = field(init=False, repr=False)
    table_bytes: int = field(init=False)

    def __post_init__(self):
        vocab = np.array(self.vocab, dtype="<f8", order="C", copy=True)
        if (vocab.ndim != 2 or min(vocab.shape) < 1 or
                not np.isfinite(vocab).all() or
                not np.isfinite([self.lam, self.eps, self.alpha]).all() or
                self.lam < 0 or self.eps < 0 or max(self.lam, self.eps) <= 0 or
                not 0 <= self.alpha <= 1):
            raise PacketError("invalid session")
        raw_vocab = vocab.tobytes(order="C")
        vocab = np.frombuffer(raw_vocab, dtype="<f8").reshape(vocab.shape)
        object.__setattr__(self, "vocab", vocab)
        manifest = {
            "format": "SBG1", "version": 1,
            "algorithm_version": "alignment_numerical_v1",
            "rank_policy": {"threshold": 1e-10, "ambiguity": [1e-12, 1e-8],
                            "scale": "max(1,largest_eigenvalue)"},
            "lambda": float(self.lam), "epsilon": float(self.eps),
            "alpha": float(self.alpha),
            "row_norm_floor": 1e-6, "cosine_norm_floor": 1e-12,
            "reference_source_sha256": UPSTREAM_SHA256,
            "vocabulary": {"sha256": hashlib.sha256(vocab.tobytes()).hexdigest(),
                           "shape": list(vocab.shape), "dtype": "<f8"},
        }
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        object.__setattr__(self, "manifest_bytes", raw)
        object.__setattr__(self, "digest", hashlib.sha256(raw).digest())
        object.__setattr__(self, "table_bytes", int(vocab.nbytes))

    @property
    def d(self):
        return int(self.vocab.shape[1])

    @property
    def direct_context(self):
        return DirectContext(self.d, self.digest)

    @property
    def cold_setup(self):
        return {"session_json_bytes": len(self.manifest_bytes),
                "vocabulary_bytes": self.table_bytes,
                "gram_decoder_total_bytes": len(self.manifest_bytes) + self.table_bytes,
                "token_lookup_total_bytes": len(self.manifest_bytes) + self.table_bytes,
                "direct_prefix_requires_vocabulary": False}


@dataclass
class Encoded:
    packet: bytes
    metadata: dict


@dataclass
class Decoded:
    final: np.ndarray
    mode: str
    pre: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def _ids(token_ids, k, session):
    values = np.asarray(token_ids)
    if values.ndim != 1 or len(values) != k or values.dtype.kind not in "ui":
        raise PacketError("integer token IDs of exactly K entries required")
    if np.any(values < 0) or np.any(values >= len(session.vocab)) or np.any(values > 0xffffffff):
        raise PacketError("token ID outside vocabulary/uint32 range")
    return values.astype("<u4", copy=True)


def _precision(precision):
    if precision not in PRECISIONS:
        raise PacketError("unknown precision")
    code = PRECISIONS[precision]
    return code, DTYPES[code]


def _frame(mode, code, k, d, payload, session):
    if not (1 <= k <= 0xffffffff and 1 <= d <= 0xffffffff):
        raise PacketError("unsupported frame dimensions")
    return HEADER.pack(MAGIC, VERSION, mode, code, 0, k, d,
                       len(payload), session.digest) + payload


def _float_payload(values, dtype):
    cast = np.asarray(values, dtype=dtype, order="C")
    if not np.isfinite(cast).all():
        raise PacketError("nonfinite rounded payload")
    return cast.tobytes(order="C")


def _reduced_from_triangle(values, k):
    reduced = np.zeros((k - 1, k - 1), dtype=np.float64)
    index = np.tril_indices(k - 1)
    reduced[index] = values
    reduced[(index[1], index[0])] = values
    return reduced


def encode_message(s, token_ids, session, method="dual", precision="float64"):
    """Preflight rounded bytes on the sender; receiver never sees this s."""
    started = time.perf_counter()
    if method not in {"dual", "canonical"}:
        raise PacketError("unknown Gram variant")
    s = np.asarray(s, dtype=np.float64)
    if (s.ndim != 2 or len(s) < 1 or s.shape[1] != session.vocab.shape[1]
            or not np.isfinite(s).all()):
        raise PacketError("invalid sender shape/values")
    k, d = s.shape
    code, dtype = _precision(precision)
    missing = token_ids is None or np.asarray(token_ids).size == 0
    ids = None if missing else _ids(token_ids, k, session)
    r = None if missing else session.vocab[ids]
    metadata = {"variant": method, "precision": precision, "K": k, "d": d,
                "header_bytes": HEADER.size, "sender_gram_seconds": 0.0,
                "sender_preflight_seconds": 0.0, "dense_fallback_seconds": 0.0,
                "receiver_accesses_sender": False}
    try:
        if missing:
            raise Unsupported("missing_tokens")
        if not (2 <= k < d):
            raise Unsupported("outside_2_le_K_lt_d")
        phase = time.perf_counter()
        a = helmert(k).T @ s
        ga = a @ a.T
        # Match the symmetric statistic used by the core eigensolver.
        ga = (ga + ga.T) * 0.5
        tri = np.tril_indices(k - 1)
        gram_bytes = _float_payload(ga[tri], dtype)
        rounded = _reduced_from_triangle(np.frombuffer(gram_bytes, dtype=dtype).astype(np.float64), k)
        metadata["sender_gram_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        try:
            check = align_from_gram(rounded, r, method, session.lam, session.eps, details=False)
            final = postprocess(check.pre, session.vocab, session.alpha)["final"]
            if not np.isfinite(final).all():
                raise Unsupported("nonfinite_preflight_output")
        finally:
            metadata["sender_preflight_seconds"] = time.perf_counter() - phase
        payload = ids.tobytes() + gram_bytes
        mode = 1 if method == "dual" else 2
        metadata.update({"fallback": False, "rounded_preflight_supported": True,
                         "mode": MODES[mode], "token_id_bytes": int(ids.nbytes),
                         "gram_bytes": len(gram_bytes), "raw_prefix_bytes": 0})
    except Unsupported as exc:
        phase = time.perf_counter()
        result = dense_reference(s, r, session.lam, session.eps, details=False)
        final = result.pre if result.bypass else postprocess(
            result.pre, session.vocab, session.alpha)["final"]
        payload = _float_payload(final, dtype)
        metadata["dense_fallback_seconds"] = time.perf_counter() - phase
        mode = 3
        metadata.update({"fallback": True, "rounded_preflight_supported": False,
                         "fallback_reason": exc.reason,
                         "fallback_diagnostics": exc.diagnostics,
                         "mode": MODES[mode], "output_source": (
                             "raw_hidden_early_return" if result.bypass else "dense_reference"),
                         "token_id_bytes": 0, "gram_bytes": 0,
                         "raw_prefix_bytes": len(payload)})
    packet = _frame(mode, code, k, d, payload, session)
    metadata.update({"payload_bytes": len(payload), "packet_bytes": len(packet),
                     "sender_total_encode_seconds": time.perf_counter() - started,
                     "cold_setup": session.cold_setup})
    return Encoded(packet, metadata)


def encode_direct(prefix, session, precision="float64"):
    """The prefix is already postprocessed; no decoder vocabulary is needed."""
    started = time.perf_counter()
    prefix = np.asarray(prefix, dtype=np.float64)
    if prefix.ndim != 2 or len(prefix) < 1 or prefix.shape[1] != session.d:
        raise PacketError("invalid direct prefix shape")
    code, dtype = _precision(precision)
    payload = _float_payload(prefix, dtype)
    k, d = prefix.shape
    packet = _frame(4, code, k, d, payload, session)
    return Encoded(packet, {"mode": "direct", "precision": precision,
                            "header_bytes": HEADER.size, "payload_bytes": len(payload),
                            "packet_bytes": len(packet), "receiver_requires_vocabulary": False,
                            "sender_total_encode_seconds": time.perf_counter() - started})


def encode_token_only(token_ids, session):
    values = np.asarray(token_ids)
    if values.ndim != 1 or len(values) < 1:
        raise PacketError("nonempty token IDs required")
    ids = _ids(values, len(values), session)
    payload = ids.tobytes()
    packet = _frame(5, 0, len(ids), session.vocab.shape[1], payload, session)
    return Encoded(packet, {"mode": "token_only", "header_bytes": HEADER.size,
                            "payload_bytes": len(payload), "packet_bytes": len(packet),
                            "receiver_requires_embedding_lookup": True,
                            "cold_setup": session.cold_setup})


def decode_packet(packet, session):
    """Pure receiver interface: immutable bytes plus the shared session only."""
    if not isinstance(session, (Session, DirectContext)):
        raise PacketError("unknown receiver context")
    if not isinstance(packet, bytes) or len(packet) < HEADER.size:
        raise PacketError("truncated/nonbytes frame")
    magic, version, mode, code, flags, k, d, length, digest = HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION or flags != 0 or mode not in MODES:
        raise PacketError("unknown frame header")
    if digest != session.digest:
        raise PacketError("session digest mismatch")
    if k < 1 or d != session.d or length != len(packet) - HEADER.size:
        raise PacketError("dimension/payload length mismatch")
    payload = packet[HEADER.size:]
    metadata = {"mode": MODES[mode], "header_bytes": HEADER.size,
                "payload_bytes": len(payload), "packet_bytes": len(packet),
                "receiver_accesses_sender": False}
    if mode in {1, 2, 5} and not isinstance(session, Session):
        raise PacketError("Gram/token decoding requires the full shared table")
    if mode == 5:
        if code != 0 or len(payload) != 4 * k:
            raise PacketError("token-only payload mismatch")
        ids = np.frombuffer(payload, dtype="<u4")
        if np.any(ids >= len(session.vocab)):
            raise PacketError("unknown token ID")
        return Decoded(session.vocab[ids].copy(), MODES[mode], metadata=metadata)
    if code not in DTYPES:
        raise PacketError("unknown floating-point code")
    dtype = DTYPES[code]
    if mode in {3, 4}:
        if len(payload) != k * d * dtype.itemsize:
            raise PacketError("raw prefix payload mismatch")
        final = np.frombuffer(payload, dtype=dtype).astype(np.float64).reshape(k, d)
        if not np.isfinite(final).all():
            raise PacketError("nonfinite raw prefix")
        return Decoded(final, MODES[mode], metadata=metadata)
    expected = 4 * k + dtype.itemsize * k * (k - 1) // 2
    if len(payload) != expected:
        raise PacketError("Gram payload mismatch")
    ids = np.frombuffer(payload[:4 * k], dtype="<u4")
    if np.any(ids >= len(session.vocab)):
        raise PacketError("unknown token ID")
    values = np.frombuffer(payload[4 * k:], dtype=dtype).astype(np.float64)
    if not np.isfinite(values).all():
        raise PacketError("nonfinite Gram")
    gram = _reduced_from_triangle(values, k)
    try:
        result = align_from_gram(gram, session.vocab[ids], MODES[mode],
                                 session.lam, session.eps, details=False)
        processed = postprocess(result.pre, session.vocab, session.alpha)
    except (Unsupported, ValueError, np.linalg.LinAlgError) as exc:
        raise PacketError("decoder guard rejection: " + str(exc)) from exc
    metadata.update({"snap_ids": processed["ids"].tolist(),
                     "snap_margins": processed["margins"].tolist()})
    return Decoded(processed["final"], MODES[mode], result.pre, metadata)
