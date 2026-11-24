"""Independent numerical and wire contracts for the coefficient v2 codec.

These are small fixed unit fixtures, not the preregistered study grid.
The extreme rounded-matrix controls exercise the pure predicate; they do not
claim that the upstream full-rank admission policy accepts those scenes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import unittest
from unittest.mock import patch

import numpy as np

from latent_bridge import alignment_core as original_core
from latent_bridge import alignment_packet as original_packet
from latent_bridge import coefficient_packet as codec

WIRE = struct.Struct("<4sBBBBIIQ32s")
ARMS = ("coefficient", "dual_direct", "thin_direct",
        "gram_guard", "gram_original", "token_lookup")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def manual_packet(context, mode, code, k, d, payload):
    return WIRE.pack(b"SBG1", 1, mode, code, 0, k, d, len(payload),
                     context.digest) + payload


def replace_header(packet, index, value):
    fields = list(WIRE.unpack_from(packet))
    fields[index] = value
    return WIRE.pack(*fields) + packet[WIRE.size:]


def split_setup(payload):
    n = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4:4+n]), payload[4+n:]


def rebuild_setup(meta, table):
    raw = canonical_json(meta)
    return struct.pack("<I", len(raw)) + raw + table


def fixed_helmert(k):
    result = np.zeros((k, k - 1))
    for j in range(k - 1):
        denominator = np.sqrt((j + 1) * (j + 2))
        result[:j + 1, j] = 1.0 / denominator
        result[j + 1, j] = -(j + 1) / denominator
    return result


class CoefficientPacketTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[1., 1., 0.], [-1., -1., 0.]])
        self.vocab = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        self.ids = np.array([0, 1], dtype=np.uint32)

    def native(self, arm="coefficient", precision="float64"):
        sender = codec.prepare_sender(self.vocab, arm)
        packet = codec.encode_arm(self.source, self.ids, sender, arm, precision)
        receiver = codec.prepare_receiver(sender, arm)
        return sender, receiver, packet

    def test_session_identity_binds_v2_semantics_but_not_cache_placement(self):
        plain = codec.SessionV2(self.vocab, cache_post=False)
        cached = codec.SessionV2(self.vocab, cache_post=True)
        old = original_packet.Session(self.vocab)
        self.assertEqual(plain.digest, cached.digest)
        self.assertEqual(plain.manifest_bytes, cached.manifest_bytes)
        self.assertEqual(hashlib.sha256(plain.manifest_bytes).digest(), plain.digest)
        self.assertNotEqual(plain.digest, old.digest)
        manifest = json.loads(plain.manifest_bytes)
        self.assertEqual(manifest["coefficient_mode"], 6)
        self.assertEqual(manifest["session_version"], 2)
        self.assertIn("basis", manifest)
        self.assertIn("reconstruction_grouping", manifest)
        self.assertEqual(manifest["wire_precisions"]["float32"], "<f4")
        self.assertNotEqual(plain.digest, codec.SessionV2(self.vocab, alpha=0.2).digest)
        for arm in ARMS:
            with self.subTest(arm=arm):
                sender = codec.prepare_sender(self.vocab, arm)
                receiver = codec.prepare_receiver(sender, arm)
                self.assertEqual(sender.cache_post, arm in {"dual_direct", "thin_direct"})
                self.assertEqual(sender.digest, receiver.digest)
                if arm in {"dual_direct", "thin_direct"}:
                    self.assertIsInstance(receiver, original_packet.DirectContext)
                    self.assertFalse(hasattr(receiver, "vocab"))
                else:
                    self.assertEqual(receiver.cache_post, arm in {"coefficient", "gram_guard"})

    def test_native_scalar_coefficient_matches_closed_form_and_wire_layout(self):
        sender, receiver, encoded = self.native()
        magic, version, mode, code, flags, k, d, length, fingerprint = WIRE.unpack_from(encoded.packet)
        self.assertEqual((magic, version, mode, code, flags, k, d),
                         (b"SBG1", 1, 6, 1, 0, 2, 3))
        self.assertEqual(length, 4 * k + 8)
        self.assertEqual(len(encoded.packet), 56 + 4 * k + 8)
        self.assertEqual(fingerprint, sender.digest)
        np.testing.assert_array_equal(np.frombuffer(encoded.packet[56:64], dtype="<u4"), self.ids)
        coefficient = struct.unpack_from("<d", encoded.packet, 64)[0]
        expected = np.sqrt(2 * (1 + sender.lam) / (2 + sender.lam))
        self.assertAlmostEqual(coefficient, expected, places=12)
        self.assertFalse(encoded.metadata["fallback"])
        decoded = codec.decode_arm(encoded.packet, receiver, "coefficient")
        np.testing.assert_allclose(decoded.pre, expected * self.vocab[self.ids],
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(decoded.final, self.vocab[self.ids],
                                   atol=1e-12, rtol=1e-12)

    def test_all_cold_contexts_reconstruct_from_actual_setup_bytes(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                sender, warm_receiver, encoded = self.native(arm)
                setup = codec.encode_setup(sender, arm)
                meta, table = split_setup(setup)
                self.assertLessEqual(struct.unpack_from("<I", setup)[0], 4096)
                self.assertEqual(meta["table_bytes"], len(table))
                cold_receiver = codec.decode_setup(setup, arm)
                self.assertEqual(cold_receiver.digest, sender.digest)
                if arm in {"dual_direct", "thin_direct"}:
                    self.assertEqual(table, b"")
                    self.assertFalse(hasattr(cold_receiver, "vocab"))
                else:
                    self.assertEqual(table, np.asarray(self.vocab, dtype="<f8").tobytes())
                    np.testing.assert_array_equal(cold_receiver.vocab, self.vocab)
                warm = codec.decode_arm(encoded.packet, warm_receiver, arm)
                cold = codec.decode_arm(encoded.packet, cold_receiver, arm)
                self.assertEqual(warm.final.tobytes(), cold.final.tobytes())

    def test_direct_receiver_preserves_wire_bits_without_a_vocabulary(self):
        values = np.array([[0.0, -0.0, 1.25], [-2.5, 3.0, -0.0]])
        session = codec.prepare_sender(self.vocab, "dual_direct")
        context = codec.prepare_receiver(session, "dual_direct")
        for code, dtype in ((1, "<f8"), (2, "<f4")):
            with self.subTest(dtype=dtype):
                payload = np.asarray(values, dtype=dtype).tobytes()
                packet = manual_packet(context, 4, code, 2, 3, payload)
                decoded = codec.decode_arm(packet, context, "dual_direct")
                self.assertEqual(np.asarray(decoded.final, dtype=dtype).tobytes(), payload)
                self.assertFalse(hasattr(context, "vocab"))

    def test_native_header_truncation_and_trailing_bytes_are_rejected(self):
        _, receiver, encoded = self.native()
        packet = encoded.packet
        bad = {
            "magic": replace_header(packet, 0, b"BAD!"),
            "version": replace_header(packet, 1, 255),
            "mode": replace_header(packet, 2, 255),
            "precision": replace_header(packet, 3, 255),
            "reserved_flags": replace_header(packet, 4, 1),
            "zero_rows": replace_header(packet, 5, 0),
            "wrong_width": replace_header(packet, 6, 4),
            "wrong_length": replace_header(packet, 7, 1),
            "wrong_session": replace_header(packet, 8, bytes(32)),
            "truncated": packet[:-1],
            "trailing": packet + b"x",
            "mutable": bytearray(packet),
        }
        for name, broken in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                codec.decode_arm(broken, receiver, "coefficient")

    def test_actual_wire_coefficient_and_ids_are_validated(self):
        _, receiver, encoded = self.native()
        for coefficient in (-1.0, float("nan")):
            payload = self.ids.astype("<u4").tobytes() + struct.pack("<d", coefficient)
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                codec.decode_arm(manual_packet(receiver, 6, 1, 2, 3, payload),
                                 receiver, "coefficient")
        for bad_id in (len(self.vocab), 0xffffffff):
            raw = bytearray(encoded.packet)
            struct.pack_into("<I", raw, 56, bad_id)
            with self.subTest(id=bad_id), self.assertRaises(ValueError):
                codec.decode_arm(bytes(raw), receiver, "coefficient")

    def test_wrong_mode_or_session_cannot_supply_receiver_state(self):
        sender, receiver, encoded = self.native()
        with self.assertRaises(ValueError):
            codec.decode_arm(encoded.packet, receiver, "gram_guard")
        with self.assertRaises(ValueError):
            codec.decode_arm(encoded.packet, sender.direct_context, "coefficient")
        other = codec.prepare_sender(self.vocab[::-1].copy(), "coefficient")
        with self.assertRaises(ValueError):
            codec.decode_arm(encoded.packet, codec.prepare_receiver(other, "coefficient"),
                             "coefficient")

    def test_setup_limits_canonical_json_and_table_digest_are_enforced(self):
        sender, _, _ = self.native()
        setup = codec.encode_setup(sender, "coefficient")
        meta, table = split_setup(setup)
        wrong_digest = copy.deepcopy(meta)
        wrong_digest["digest"] = "0" * 64
        wrong_dtype = copy.deepcopy(meta)
        wrong_dtype["manifest"]["vocabulary"]["dtype"] = "<f4"
        unknown = copy.deepcopy(meta)
        unknown["extra"] = 1
        raw = canonical_json(meta)
        spaced = b" " + raw
        bad = {
            "short": setup[:3],
            "zero_json": struct.pack("<I", 0) + setup[4:],
            "oversized_json": struct.pack("<I", 4097) + setup[4:],
            "truncated_table": setup[:-1],
            "trailing_table": setup + b"x",
            "digest": rebuild_setup(wrong_digest, table),
            "dtype": rebuild_setup(wrong_dtype, table),
            "unknown_key": rebuild_setup(unknown, table),
            "noncanonical": struct.pack("<I", len(spaced)) + spaced + table,
        }
        for name, broken in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                codec.decode_setup(broken, "coefficient")
        with self.assertRaises(ValueError):
            codec.decode_setup(setup, "dual_direct")
        direct = codec.encode_setup(codec.prepare_sender(self.vocab, "dual_direct"),
                                    "dual_direct")
        with self.assertRaises(ValueError):
            codec.decode_setup(direct + bytes(8), "dual_direct")

    def test_actual_float32_rounding_can_destroy_spd_and_is_rejected(self):
        m = np.array([[1., 1. - 2.0**-26], [1. - 2.0**-26, 1.]])
        rounded = m.astype("<f4").astype(np.float64)
        np.testing.assert_array_equal(rounded, np.ones((2, 2)))
        x, y, gb, mean = m @ m, np.eye(2), np.eye(2), np.zeros(3)
        codec.guard_coefficient(m, m, x, y, gb, mean, 1., "float64")
        with self.assertRaises(codec.Unsupported) as caught:
            codec.guard_coefficient(rounded, m, x, y, gb, mean, 1., "float32")
        self.assertEqual(caught.exception.reason, "coefficient_not_spd")

    def test_float32_output_bound_is_not_replaced_by_covariance_residual(self):
        m = np.diag([10000.0001, 1.])
        rounded = m.astype("<f4").astype(np.float64)
        y_values = np.array([1e-8, 1.])
        y = np.diag(y_values)
        x = m @ y @ m
        gb_values = 1e-3 * y_values / (1 - y_values / 3.)
        gb = np.diag(gb_values)
        mean = np.zeros(3)
        residual = np.linalg.norm(rounded @ y @ rounded - x) / max(1., np.linalg.norm(x))
        self.assertLess(residual, 1e-5)
        bound = np.linalg.norm(rounded - m) * np.sqrt(gb_values.max())
        scale = np.sqrt(np.trace(m @ gb @ m.T))
        self.assertGreater(bound, 1e-7 + 1e-5 * scale)
        with self.assertRaises(codec.Unsupported) as caught:
            codec.guard_coefficient(rounded, m, x, y, gb, mean,
                                    float(gb_values.max()), "float32")
        self.assertEqual(caught.exception.reason, "coefficient_quantization_bound")
        self.assertAlmostEqual(caught.exception.diagnostics["quantization_bound"], bound, places=14)

    def test_positive_definiteness_alone_does_not_pass_source_residual_guard(self):
        with self.assertRaises(codec.Unsupported) as caught:
            codec.guard_coefficient(2 * np.eye(2), 2 * np.eye(2),
                                    np.eye(2), np.eye(2), np.eye(2), np.zeros(3), 1.)
        self.assertEqual(caught.exception.reason, "coefficient_residual")

    def test_optimized_gram_sender_avoids_coefficient_expansion_and_search(self):
        sender = codec.prepare_sender(self.vocab, "gram_guard")
        eigen = np.linalg.eigh
        with patch.object(np.linalg, "eigh", wraps=eigen) as calls:
            with patch.object(codec, "_coefficient", side_effect=AssertionError("extra solve")):
                with patch.object(codec, "_expanded", side_effect=AssertionError("expansion")):
                    with patch.object(codec, "postprocess_v2", side_effect=AssertionError("search")):
                        encoded = codec.encode_arm(self.source, self.ids, sender, "gram_guard")
        self.assertFalse(encoded.metadata["fallback"])
        self.assertEqual(calls.call_count, 2)
        self.assertFalse(encoded.metadata["guard"]["sender_computed_coefficient"])

    def test_coefficient_sender_never_expands_or_searches_on_native_route(self):
        sender = codec.prepare_sender(self.vocab, "coefficient")
        with patch.object(codec, "_expanded", side_effect=AssertionError("expansion")):
            with patch.object(codec, "postprocess_v2", side_effect=AssertionError("search")):
                encoded = codec.encode_arm(self.source, self.ids, sender, "coefficient")
        self.assertFalse(encoded.metadata["fallback"])
        decoded = codec.decode_arm(encoded.packet, codec.prepare_receiver(sender, "coefficient"),
                                   "coefficient")
        np.testing.assert_allclose(decoded.final, self.vocab[self.ids], atol=1e-12, rtol=1e-12)

    def test_rank_fallback_sends_its_declared_dense_output_byte_for_byte(self):
        source = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [-1., -1., 0., 0.]])
        vocab = np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]])
        ids = np.zeros(3, dtype=np.uint32)
        reference = original_core.dense_reference(source, vocab[ids], details=False)
        final = original_core.postprocess(reference.pre, vocab)["final"]
        for arm in ARMS[:-1]:
            for precision, dtype in (("float64", "<f8"), ("float32", "<f4")):
                with self.subTest(arm=arm, precision=precision):
                    sender = codec.prepare_sender(vocab, arm)
                    encoded = codec.encode_arm(source, ids, sender, arm, precision)
                    self.assertTrue(encoded.metadata["fallback"])
                    self.assertEqual(WIRE.unpack_from(encoded.packet)[2], 3)
                    expected = np.asarray(final, dtype=dtype).tobytes()
                    self.assertEqual(encoded.packet[56:], expected)
                    decoded = codec.decode_arm(encoded.packet, codec.prepare_receiver(sender, arm), arm)
                    self.assertEqual(np.asarray(decoded.final, dtype=dtype).tobytes(), expected)

    def test_conservative_cross_bound_falls_back_when_full_solve_is_supported(self):
        k, d = 6, 24
        h = fixed_helmert(k)
        ga = np.array([1., 1., 1., 1., 4e-8])
        gb = np.array([4e-8, 1., 1., 1., 1.])
        basis = np.eye(d)[:k - 1]
        source = h @ np.diag(np.sqrt(ga)) @ basis
        vocab = h @ np.diag(np.sqrt(gb)) @ basis
        ids = np.arange(k, dtype=np.uint32)
        coefficient = codec.encode_arm(source, ids, codec.prepare_sender(vocab, "coefficient"),
                                       "coefficient")
        guarded = codec.encode_arm(source, ids, codec.prepare_sender(vocab, "gram_guard"),
                                   "gram_guard")
        self.assertFalse(coefficient.metadata["fallback"])
        self.assertTrue(guarded.metadata["fallback"])
        self.assertEqual(guarded.metadata["fallback_reason"], "gram_cross_bound")
        x, y = ga / (ga / k + 1e-3), gb / (gb / k + 1e-3)
        self.assertGreater(np.min(x * y), 1e-8 * max(1., float(np.max(x * y))))
        self.assertLess(np.min(x) * np.min(y), 1e-8 * max(1., np.max(x) * np.max(y)))

    def test_cached_postprocessing_matches_frozen_clamps_and_ties(self):
        vocab = np.array([[1., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 0.]])
        pre = np.array([[0., 0., 0.], [1e-12, 0., 0.], [1., 0., 0.], [0., 2., 0.]])
        actual = codec.postprocess_v2(pre, codec.SessionV2(vocab, cache_post=True))
        expected = original_core.postprocess(pre, vocab)
        for key in ("ids", "calibrated", "final", "margins"):
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(actual["ids"].tolist()[:3], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
