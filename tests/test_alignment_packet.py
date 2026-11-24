"""Independent packet-contract tests for the frozen alignment study.

Fixtures follow the public 56-byte SBG1 wire specification and elementary
closed-form cases, not the study generator. The sqrt(5) rank-one fixture checks
that sender preflight tests the serialized Gram, not an unrounded substitute.
These are small functional tests; they run no study grid or language model.
"""
import hashlib
import json
import math
import struct
import unittest
from unittest import mock

import numpy as np

from latent_bridge import alignment_packet as packet
from latent_bridge.alignment_packet import (
    DirectContext, PacketError, Session, decode_packet, encode_direct,
    encode_message, encode_token_only,
)

WIRE = struct.Struct("<4sBBBBIIQ32s")


def changed_header(data, index, value):
    fields = list(WIRE.unpack_from(data))
    fields[index] = value
    return WIRE.pack(*fields) + data[56:]


def replaced_payload(data, payload):
    fields = list(WIRE.unpack_from(data))
    fields[7] = len(payload)
    return WIRE.pack(*fields) + payload


class AlignmentPacketTests(unittest.TestCase):
    def setUp(self):
        self.vocab = np.array([[1., 0., 0.], [-1., 0., 0.]])
        self.session = Session(self.vocab)
        self.sender = np.array([[1., 1., 0.], [-1., -1., 0.]])
        self.ids = np.array([0, 1], dtype=np.int64)

    def assert_accounting(self, encoded):
        fields = WIRE.unpack_from(encoded.packet)
        self.assertEqual(fields[7], len(encoded.packet) - 56)
        self.assertEqual(encoded.metadata["header_bytes"], 56)
        self.assertEqual(encoded.metadata["payload_bytes"], len(encoded.packet) - 56)
        self.assertEqual(encoded.metadata["packet_bytes"], len(encoded.packet))

    def test_wire_header_and_direct_payload_have_independent_exact_bytes(self):
        self.assertEqual(WIRE.size, 56)
        identity = bytes(range(32))
        context = DirectContext(3, identity)
        values = [[0., -0., 1.5], [-2.25, 3., 4.5]]
        encoded = encode_direct(values, context, "float64")
        expected = (
            struct.pack("<4sBBBBIIQ32s", b"SBG1", 1, 4, 1, 0, 2, 3, 48, identity)
            + struct.pack("<6d", 0., -0., 1.5, -2.25, 3., 4.5)
        )
        self.assertEqual(encoded.packet, expected)
        self.assert_accounting(encoded)
        decoded = decode_packet(encoded.packet, context)
        self.assertEqual(decoded.mode, "direct")
        self.assertIsNone(decoded.pre)
        self.assertEqual(decoded.final.astype("<f8").tobytes(), expected[56:])

    def test_direct_context_needs_no_vocabulary_for_either_precision(self):
        context = self.session.direct_context
        self.assertFalse(hasattr(context, "vocab"))
        prefix = np.array([[1. + 2. ** -25, -0., math.pi],
                           [2. ** -149, -2. ** 20, -0.125]])
        for precision, dtype, code in (
                ("float64", "<f8", 1), ("float32", "<f4", 2)):
            with self.subTest(precision=precision):
                encoded = encode_direct(prefix, context, precision)
                decoded = decode_packet(encoded.packet, context)
                expected = prefix.astype(dtype).astype(np.float64)
                np.testing.assert_array_equal(decoded.final, expected)
                self.assertEqual(decoded.final.astype("<f8").tobytes(),
                                 expected.astype("<f8").tobytes())
                self.assertEqual(WIRE.unpack_from(encoded.packet)[3], code)
                self.assertFalse(encoded.metadata["receiver_requires_vocabulary"])
                self.assertEqual(len(encoded.packet),
                                 56 + prefix.size * np.dtype(dtype).itemsize)
                self.assert_accounting(encoded)

    def test_closed_form_sign_bit_roundtrip_for_both_gram_modes(self):
        scalar = math.sqrt(2. * (1. + self.session.lam)
                           / (2. + self.session.lam))
        for method, mode in (("dual", 1), ("canonical", 2)):
            for precision, size, code in (
                    ("float64", 8, 1), ("float32", 4, 2)):
                with self.subTest(method=method, precision=precision):
                    encoded = encode_message(
                        self.sender, self.ids, self.session, method, precision)
                    self.assertFalse(encoded.metadata["fallback"])
                    decoded = decode_packet(encoded.packet, self.session)
                    np.testing.assert_allclose(decoded.pre, scalar * self.vocab,
                                               rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(decoded.final, self.vocab,
                                               rtol=1e-8, atol=1e-10)
                    self.assertEqual(decoded.metadata["snap_ids"], [0, 1])
                    header = WIRE.unpack_from(encoded.packet)
                    self.assertEqual(header[:8],
                                     (b"SBG1", 1, mode, code, 0, 2, 3, 8 + size))
                    self.assertEqual(header[8], self.session.digest)
                    self.assertEqual(encoded.packet[56:64], struct.pack("<2I", 0, 1))
                    self.assertEqual(len(encoded.packet), 56 + 4 * 2 + size)
                    self.assert_accounting(encoded)

    def test_opposite_raw_signs_produce_same_gram_packet(self):
        other = self.sender.copy()
        other[:, 1] *= -1.
        self.assertFalse(np.array_equal(other, self.sender))
        for method in ("dual", "canonical"):
            first = encode_message(self.sender, self.ids, self.session, method)
            second = encode_message(other, self.ids, self.session, method)
            self.assertFalse(first.metadata["fallback"])
            self.assertEqual(first.packet, second.packet)

    def test_decoder_uses_packet_after_caller_mutates_private_inputs(self):
        s = self.sender.copy()
        ids = self.ids.copy()
        encoded = encode_message(s, ids, self.session)
        s[:] = 912.
        ids[:] = 1
        with mock.patch.object(packet, "dense_reference",
                               side_effect=AssertionError("receiver called sender fallback")):
            decoded = decode_packet(encoded.packet, self.session)
        np.testing.assert_allclose(decoded.final, self.vocab, rtol=1e-8, atol=1e-10)
        self.assertFalse(decoded.metadata["receiver_accesses_sender"])

    def test_token_only_is_exact_lookup_with_uint32_payload(self):
        ids = np.array([1, 0, 1], dtype=np.int64)
        encoded = encode_token_only(ids, self.session)
        header = WIRE.unpack_from(encoded.packet)
        self.assertEqual(header[:8], (b"SBG1", 1, 5, 0, 0, 3, 3, 12))
        self.assertEqual(encoded.packet[56:], struct.pack("<3I", 1, 0, 1))
        decoded = decode_packet(encoded.packet, self.session)
        np.testing.assert_array_equal(decoded.final, self.vocab[[1, 0, 1]])
        self.assertIsNone(decoded.pre)
        self.assert_accounting(encoded)

    def test_direct_context_is_rejected_for_gram_and_token_packets(self):
        grams = [encode_message(self.sender, self.ids, self.session, method).packet
                 for method in ("dual", "canonical")]
        token = encode_token_only(self.ids, self.session).packet
        for data in grams + [token]:
            with self.subTest(mode=WIRE.unpack_from(data)[2]):
                with self.assertRaises(PacketError):
                    decode_packet(data, self.session.direct_context)

    def test_missing_tokens_fallback_preserves_raw_values(self):
        raw = np.array([[2., -0., 5.], [-7., 3.5, -11.]])
        for missing in (None, []):
            with self.subTest(missing=missing):
                encoded = encode_message(raw, missing, self.session)
                self.assertTrue(encoded.metadata["fallback"])
                self.assertEqual(encoded.metadata["output_source"], "raw_hidden_early_return")
                decoded = decode_packet(encoded.packet, self.session.direct_context)
                self.assertEqual(decoded.mode, "raw_fallback")
                self.assertEqual(decoded.final.astype("<f8").tobytes(), raw.tobytes())
                self.assertEqual(len(encoded.packet), 56 + raw.size * 8)
                self.assert_accounting(encoded)

    def test_one_row_early_return_is_not_calibrated_or_anchored(self):
        raw = np.array([[3., 4., -9.]])
        encoded = encode_message(raw, [0], self.session, precision="float32")
        self.assertEqual(encoded.metadata["output_source"], "raw_hidden_early_return")
        decoded = decode_packet(encoded.packet, self.session.direct_context)
        np.testing.assert_array_equal(decoded.final, raw)
        self.assertEqual(WIRE.unpack_from(encoded.packet)[2:4], (3, 2))

    def test_dimension_guard_uses_sender_dense_fallback(self):
        reference = np.array([[1., 0.], [-1., 0.]])
        session = Session(reference)
        sender = np.array([[1., 1.], [-1., -1.]])
        encoded = encode_message(sender, [0, 1], session)
        self.assertTrue(encoded.metadata["fallback"])
        self.assertEqual(encoded.metadata["fallback_reason"], "outside_2_le_K_lt_d")
        self.assertEqual(encoded.metadata["output_source"], "dense_reference")
        decoded = decode_packet(encoded.packet, session.direct_context)
        np.testing.assert_allclose(decoded.final, reference, rtol=1e-8, atol=1e-10)
        self.assert_accounting(encoded)

    def test_rank_deficient_dual_fallback_and_zero_sender_output(self):
        zero = np.zeros((2, 3))
        encoded = encode_message(zero, self.ids, self.session, method="dual")
        self.assertTrue(encoded.metadata["fallback"])
        self.assertEqual(encoded.metadata["output_source"], "dense_reference")
        decoded = decode_packet(encoded.packet, self.session.direct_context)
        expected = np.tile(0.3 * self.vocab[0], (2, 1))
        np.testing.assert_allclose(decoded.final, expected, rtol=0., atol=1e-12)
        self.assert_accounting(encoded)

    def test_canonical_handles_exact_zero_sender_without_dense_fallback(self):
        with mock.patch.object(packet, "dense_reference",
                               side_effect=AssertionError("unnecessary dense fallback")):
            encoded = encode_message(np.zeros((2, 3)), self.ids,
                                     self.session, method="canonical")
            decoded = decode_packet(encoded.packet, self.session)
        self.assertFalse(encoded.metadata["fallback"])
        self.assertEqual(decoded.mode, "canonical")
        np.testing.assert_array_equal(decoded.pre, np.zeros((2, 3)))
        np.testing.assert_allclose(decoded.final, np.tile(0.3 * self.vocab[0], (2, 1)),
                                   rtol=0., atol=1e-12)

    def test_float32_rounded_indefinite_gram_falls_back_at_sender(self):
        # The exact rank-one reduced Gram has entries 1, sqrt(5), 5.
        # float32 rounds sqrt(5) upward, making that rounded matrix indefinite.
        u = np.array([[1. / math.sqrt(2.), 1. / math.sqrt(6.)],
                      [-1. / math.sqrt(2.), 1. / math.sqrt(6.)],
                      [0., -2. / math.sqrt(6.)]])
        a = np.array([[1., 0., 0., 0.], [math.sqrt(5.), 0., 0., 0.]])
        sender = u @ a
        session = Session(sender)
        exact = encode_message(sender, [0, 1, 2], session, "canonical", "float64")
        rounded = encode_message(sender, [0, 1, 2], session, "canonical", "float32")
        self.assertFalse(exact.metadata["fallback"])
        self.assertTrue(rounded.metadata["fallback"])
        self.assertFalse(rounded.metadata["rounded_preflight_supported"])
        self.assertEqual(rounded.metadata["fallback_reason"], "source_not_psd")
        self.assertEqual(rounded.metadata["output_source"], "dense_reference")
        self.assertEqual(rounded.metadata["gram_bytes"], 0)
        self.assertEqual(rounded.metadata["raw_prefix_bytes"], 3 * 4 * 4)
        decoded = decode_packet(rounded.packet, session.direct_context)
        self.assertEqual(decoded.mode, "raw_fallback")
        self.assertEqual(decoded.final.shape, (3, 4))
        self.assertTrue(np.isfinite(decoded.final).all())
        self.assert_accounting(rounded)

    def test_phase_costs_are_present_and_inside_total_encode_time(self):
        for encoded in (
                encode_message(self.sender, self.ids, self.session),
                encode_message(self.sender[:1], [0], self.session)):
            names = ("sender_gram_seconds", "sender_preflight_seconds",
                     "dense_fallback_seconds")
            values = [encoded.metadata[name] for name in names]
            self.assertTrue(all(math.isfinite(x) and x >= 0. for x in values))
            self.assertGreaterEqual(
                encoded.metadata["sender_total_encode_seconds"] + 1e-12, sum(values))

    def test_session_table_is_copied_and_cannot_be_made_writable(self):
        original = self.vocab.copy()
        session = Session(original)
        digest = session.digest
        original[0, 0] = 100.
        np.testing.assert_array_equal(session.vocab, self.vocab)
        with self.assertRaises(ValueError):
            session.vocab[0, 0] = 9.
        with self.assertRaises(ValueError):
            session.vocab.setflags(write=True)
        self.assertEqual(session.digest, digest)

    def test_manifest_hash_and_cold_setup_include_actual_table(self):
        session = self.session
        manifest = json.loads(session.manifest_bytes)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
        self.assertEqual(canonical, session.manifest_bytes)
        self.assertEqual(hashlib.sha256(canonical).digest(), session.digest)
        self.assertEqual(manifest["vocabulary"]["sha256"],
                         hashlib.sha256(self.vocab.astype("<f8").tobytes()).hexdigest())
        self.assertEqual(manifest["vocabulary"]["shape"], [2, 3])
        self.assertEqual(manifest["vocabulary"]["dtype"], "<f8")
        self.assertEqual(session.table_bytes, 48)
        self.assertEqual(session.cold_setup["vocabulary_bytes"], 48)
        self.assertEqual(session.cold_setup["session_json_bytes"], len(canonical))
        self.assertEqual(session.cold_setup["gram_decoder_total_bytes"],
                         48 + len(canonical))
        self.assertFalse(session.cold_setup["direct_prefix_requires_vocabulary"])

    def test_session_fingerprint_binds_parameters_and_vocabulary(self):
        encoded = encode_message(self.sender, self.ids, self.session)
        alternatives = (
            Session(self.vocab, lam=0.01), Session(self.vocab, eps=1e-5),
            Session(self.vocab, alpha=0.0), Session(2. * self.vocab),
        )
        for session in alternatives:
            with self.subTest(digest=session.digest.hex()):
                self.assertNotEqual(self.session.digest, session.digest)
                with self.assertRaises(PacketError):
                    decode_packet(encoded.packet, session)

    def test_short_nonbyte_truncated_and_trailing_frames_are_rejected(self):
        valid = encode_message(self.sender, self.ids, self.session).packet
        for data in (None, "", bytearray(valid), b"", valid[:55], valid[:-1],
                     valid + b"\x00"):
            with self.subTest(kind=type(data).__name__, length=(
                    len(data) if data is not None else None)):
                with self.assertRaises(PacketError):
                    decode_packet(data, self.session)

    def test_unknown_or_inconsistent_header_fields_are_rejected(self):
        valid = encode_message(self.sender, self.ids, self.session).packet
        cases = ((0, b"BAD!"), (1, 2), (2, 0), (2, 255), (3, 0), (3, 255),
                 (4, 1), (5, 0), (5, 0xffffffff), (6, 0), (6, 4),
                 (7, 0), (8, bytes(32)))
        for index, value in cases:
            with self.subTest(index=index, value=value):
                with self.assertRaises(PacketError):
                    decode_packet(changed_header(valid, index, value), self.session)

    def test_declared_payload_sizes_are_checked_before_matrix_allocation(self):
        gram = encode_message(self.sender, self.ids, self.session).packet
        direct = encode_direct(self.sender, self.session.direct_context).packet
        token = encode_token_only(self.ids, self.session).packet
        for data in (gram, direct, token):
            for payload in (data[56:-1], data[56:] + b"\x00"):
                with self.subTest(mode=WIRE.unpack_from(data)[2], length=len(payload)):
                    with self.assertRaises(PacketError):
                        decode_packet(replaced_payload(data, payload), self.session)
        with self.assertRaises(PacketError):
            decode_packet(changed_header(direct, 5, 0xffffffff), self.session.direct_context)

    def test_out_of_range_wire_ids_are_rejected_in_every_lookup_mode(self):
        packets = [
            encode_message(self.sender, self.ids, self.session, "dual").packet,
            encode_message(self.sender, self.ids, self.session, "canonical").packet,
            encode_token_only(self.ids, self.session).packet,
        ]
        for data in packets:
            payload = struct.pack("<I", 0xffffffff) + data[60:]
            with self.subTest(mode=WIRE.unpack_from(data)[2]):
                with self.assertRaises(PacketError):
                    decode_packet(replaced_payload(data, payload), self.session)

    def test_malformed_token_only_dtype_is_rejected(self):
        data = encode_token_only(self.ids, self.session).packet
        for code in (1, 2, 255):
            with self.subTest(code=code):
                with self.assertRaises(PacketError):
                    decode_packet(changed_header(data, 3, code), self.session)

    def test_nonfinite_raw_values_are_rejected_on_both_sides(self):
        data = encode_direct(self.sender, self.session.direct_context).packet
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                poisoned = self.sender.copy()
                poisoned[0, 0] = value
                with self.assertRaises(PacketError):
                    encode_direct(poisoned, self.session.direct_context)
                payload = struct.pack("<d", value) + data[64:]
                with self.assertRaises(PacketError):
                    decode_packet(replaced_payload(data, payload),
                                  self.session.direct_context)

    def test_nonfinite_or_negative_gram_is_rejected_without_receiver_fallback(self):
        data = encode_message(self.sender, self.ids, self.session).packet
        for value in (float("nan"), float("inf"), -1.):
            payload = data[56:64] + struct.pack("<d", value)
            with self.subTest(value=value):
                with mock.patch.object(packet, "dense_reference",
                                       side_effect=AssertionError("hidden receiver fallback")):
                    with self.assertRaises(PacketError):
                        decode_packet(replaced_payload(data, payload), self.session)

    def test_wire_gram_dimension_guard_cannot_invoke_dense_receiver(self):
        base = encode_message(self.sender, self.ids, self.session).packet
        cases = [
            (1, struct.pack("<I", 0)),
            (3, struct.pack("<3I3d", 0, 1, 0, 1., 0., 1.)),
        ]
        for k, payload in cases:
            data = changed_header(replaced_payload(base, payload), 5, k)
            with self.subTest(K=k):
                with mock.patch.object(packet, "dense_reference",
                                       side_effect=AssertionError("hidden receiver fallback")):
                    with self.assertRaises(PacketError):
                        decode_packet(data, self.session)

    def test_sender_rejects_invalid_ids_instead_of_treating_them_as_missing(self):
        invalid = ([0], [0, 1, 0], [0., 1.], [False, True],
                   [-1, 1], [0, 2], [0, 2 ** 32], [[0, 1]])
        for ids in invalid:
            with self.subTest(ids=ids):
                with self.assertRaises(PacketError):
                    encode_message(self.sender, ids, self.session)
        with self.assertRaises(PacketError):
            encode_token_only([], self.session)

    def test_sender_rejects_bad_shape_nonfinite_and_unknown_format(self):
        for sender in (np.array([1., 2., 3.]), np.zeros((0, 3)),
                       np.zeros((2, 4)), np.full((2, 3), np.nan)):
            with self.subTest(shape=sender.shape):
                with self.assertRaises(PacketError):
                    encode_message(sender, self.ids, self.session)
        for kwargs in ({"method": "unknown"}, {"precision": "float16"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(PacketError):
                    encode_message(self.sender, self.ids, self.session, **kwargs)

    def test_invalid_session_and_direct_context_are_rejected(self):
        invalid = (
            {"vocab": np.zeros((0, 3))}, {"vocab": np.array([1., 2.])},
            {"vocab": np.full((2, 3), np.nan)}, {"vocab": self.vocab, "lam": -1.},
            {"vocab": self.vocab, "eps": -1.}, {"vocab": self.vocab, "alpha": 1.1},
            {"vocab": self.vocab, "lam": 0., "eps": 0.},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=sorted(kwargs)):
                with self.assertRaises(PacketError):
                    Session(**kwargs)
        for width, digest in ((0, bytes(32)), (3, bytes(31)), (3, "x" * 32)):
            with self.subTest(width=width, digest_type=type(digest).__name__):
                with self.assertRaises(PacketError):
                    DirectContext(width, digest)
        data = encode_direct(self.sender, self.session.direct_context).packet
        with self.assertRaises(PacketError):
            decode_packet(data, object())


if __name__ == "__main__":
    unittest.main()
