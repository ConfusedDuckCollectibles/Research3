"""Independent framing, receiver-isolation and fixed-plan contracts.

Small constant unit fixtures and mocked failures only: this file does not call
the study runner, sample the primary fixtures or measure its performance grid.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import itertools
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from latent_bridge import coefficient_packet as codec
from latent_bridge import coefficient_transport as transport

ARMS = ("coefficient", "dual_direct", "thin_direct",
        "gram_guard", "gram_original", "token_lookup")
SHAPES = ((12, 192), (36, 512), (60, 960))
SEEDS = (17011, 17027, 17041)


class ScriptedSocket:
    def __init__(self, received=b"", *, read_chunk=1, send_chunk=None,
                 fail_after=None, events=None):
        self.incoming = bytearray(received)
        self.read_chunk = read_chunk
        self.send_chunk = send_chunk
        self.fail_after = fail_after
        self.sent = bytearray()
        self.events = [] if events is None else events

    def recv(self, count):
        size = min(count, self.read_chunk, len(self.incoming))
        result = bytes(self.incoming[:size])
        del self.incoming[:size]
        return result

    def send(self, value):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise OSError("injected partial send failure")
        count = len(value) if self.send_chunk is None else min(len(value), self.send_chunk)
        if self.fail_after is not None:
            count = min(count, self.fail_after - len(self.sent))
        piece = bytes(value[:count])
        self.sent.extend(piece)
        self.events.append(("send", piece))
        return count


class FakeTime:
    def __init__(self):
        self.ns = 0
        self.events = []

    def clock(self):
        return self.ns

    def sleep(self, seconds):
        self.events.append(("sleep", seconds))
        self.ns += round(seconds * 1e9)


def fixture():
    source = np.array([[1., 1., 0.], [-1., -1., 0.]])
    vocabulary = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    ids = np.array([0, 1], dtype=np.uint32)
    return source, ids, vocabulary


def synthetic_trial_records(plan):
    """Hand-written arithmetic data, never measured or used as study output."""
    times = {
        "coefficient": [10, 20, 30, 40],
        "dual_direct": [100, 1, 1, 100],
        "thin_direct": [1, 30, 30, 1],
        "gram_guard": [60, 60, 60, 60],
        "gram_original": [70, 70, 70, 70],
        "token_lookup": [1, 1, 1, 1],
    }
    return [{**row, "status": "completed", "wall_ns": times[row["arm"]][row["repetition"]]}
            for row in plan]


class MemoryDescriptions:
    """Minimal retention sink; no filesystem and no oracle-valued inputs."""
    def raw(self, value):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    def array(self, value):
        a = np.asarray(value)
        return {"shape": list(a.shape), "dtype": a.dtype.str,
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}

    def decoded(self, value):
        return {"mode": value.mode, "final": self.array(value.final),
                "pre": None if value.pre is None else self.array(value.pre)}


class CoefficientTransportTests(unittest.TestCase):
    def test_all_frame_bytes_are_paid_before_send_including_final_ack(self):
        clock = FakeTime()
        peer = ScriptedSocket(events=clock.events)
        profile = transport.Profile("unit-paced", 1_000_000.0, 0.020)
        io = transport.FrameIO(peer, profile, clock_ns=clock.clock, sleep=clock.sleep)
        payload = b"a" * (16 * 1024 + 3)
        io.send_frame(payload)
        ack = transport.make_ack(b"ACK2", hashlib.sha256(b"unit").digest())
        io.send_frame(ack, reply=True)
        expected = struct.pack(">I", len(payload)) + payload + struct.pack(">I", 40) + ack
        self.assertEqual(bytes(peer.sent), expected)
        summary, streams = io.snapshot()
        self.assertEqual(streams["sent"], expected)
        self.assertEqual(summary["sent_bytes"], len(expected))
        self.assertEqual(summary["sent_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertAlmostEqual(summary["wait_totals"]["pacing"]["requested_seconds"],
                               len(expected) / profile.bytes_per_second, places=14)
        self.assertAlmostEqual(summary["wait_totals"]["reply_holdoff"]["requested_seconds"],
                               0.020, places=14)
        paid = sent = 0
        for kind, value in clock.events:
            if kind == "sleep":
                paid += value
            else:
                self.assertLessEqual(len(value), 16 * 1024)
                sent += len(value)
                self.assertGreaterEqual(paid + 1e-12, sent / profile.bytes_per_second)
        pacing = [row["requested_seconds"] for row in summary["waits"] if row["kind"] == "pacing"]
        self.assertAlmostEqual(pacing[-1], 44 / profile.bytes_per_second, places=14)
        self.assertTrue(all(row["complete"] for row in summary["sent_frames"]))

    def test_partial_send_failure_retains_only_actual_transmitted_bytes(self):
        clock = FakeTime()
        peer = ScriptedSocket(send_chunk=3, fail_after=6, events=clock.events)
        io = transport.FrameIO(peer, transport.PROFILES[0],
                               clock_ns=clock.clock, sleep=clock.sleep)
        payload = b"abcdefghij"
        with self.assertRaises(OSError):
            io.send_frame(payload)
        summary, streams = io.snapshot()
        expected = (struct.pack(">I", len(payload)) + payload)[:6]
        self.assertEqual(streams["sent"], expected)
        self.assertEqual(summary["sent_bytes"], 6)
        self.assertEqual(summary["sent_frames"][0]["actual_wire_bytes"], 6)
        self.assertEqual(summary["sent_frames"][0]["declared_wire_bytes"], 14)
        self.assertFalse(summary["sent_frames"][0]["complete"])
        self.assertIn("partial send", summary["sent_frames"][0]["error"])

    def test_partial_reads_preserve_framing_and_exact_wire_identity(self):
        payload = b"\x00real framed bytes\xff"
        raw = struct.pack(">I", len(payload)) + payload
        io = transport.FrameIO(ScriptedSocket(raw, read_chunk=1), transport.PROFILES[0])
        self.assertEqual(io.recv_frame(), payload)
        summary, streams = io.snapshot()
        self.assertEqual(streams["received"], raw)
        self.assertEqual(summary["received_bytes"], len(raw))
        self.assertEqual(summary["received_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(summary["received_frames"][0]["complete"])

    def test_truncated_zero_and_oversized_frames_fail_with_partial_counters(self):
        cases = (
            (b"\x00\x00", EOFError, 2),
            (struct.pack(">I", 5) + b"ab", EOFError, 6),
            (struct.pack(">I", 0), ValueError, 4),
            (struct.pack(">I", 16 * 1024 * 1024 + 1), ValueError, 4),
        )
        for raw, error, expected_bytes in cases:
            with self.subTest(raw=raw):
                io = transport.FrameIO(ScriptedSocket(raw), transport.PROFILES[0])
                with self.assertRaises(error):
                    io.recv_frame()
                summary, streams = io.snapshot()
                self.assertEqual(summary["received_bytes"], expected_bytes)
                self.assertEqual(streams["received"], raw)
                self.assertFalse(summary["received_frames"][0]["complete"])

    def test_setup_and_output_ack_meanings_are_distinct_and_strict(self):
        digest = hashlib.sha256(b"received setup payload").digest()
        setup_ack = transport.make_ack(b"SET2", digest)
        self.assertEqual(len(setup_ack), 40)
        self.assertEqual(transport.validate_ack(setup_ack, b"SET2", digest), digest)
        with self.assertRaises(ValueError):
            transport.validate_ack(setup_ack, b"ACK2")
        with self.assertRaises(ValueError):
            transport.validate_ack(setup_ack, b"SET2", bytes(32))
        valid = transport.make_ack(b"ACK2", digest)
        for broken in (valid[:-1], valid + b"x", valid[:5] + b"x" + valid[6:],
                       struct.pack("<4sB3x32s", b"ACK2", 2, digest),
                       struct.pack("<4sB3x32s", b"ACK2", 1, digest),
                       transport.make_ack(b"ACK2", bytes(32), status=1)):
            with self.subTest(packet=broken), self.assertRaises(ValueError):
                transport.validate_ack(broken, b"ACK2")

    def test_receiver_reconstructs_from_socket_and_context_only_warm_and_cold(self):
        source, ids, vocab = fixture()
        sender = codec.prepare_sender(vocab, "coefficient")
        encoded = codec.encode_arm(source, ids, sender, "coefficient")
        setup = codec.encode_setup(sender, "coefficient")
        for cold in (False, True):
            with self.subTest(cold=cold):
                left, right = socket.socketpair()
                left.settimeout(2)
                right.settimeout(2)
                holder = {}
                context = None if cold else codec.prepare_receiver(sender, "coefficient")
                def worker():
                    holder["result"] = transport.receiver_exchange(
                        right, context, "coefficient", transport.PROFILES[0], cold=cold)
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    io = transport.FrameIO(left, transport.PROFILES[0])
                    if cold:
                        io.send_frame(setup)
                        receipt = io.recv_frame()
                        transport.validate_ack(receipt, b"SET2", hashlib.sha256(setup).digest())
                    io.send_frame(encoded.packet)
                    ack = io.recv_frame()
                    digest = transport.validate_ack(ack, b"ACK2")
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    result = holder["result"]
                    self.assertIsNone(result["error"])
                    np.testing.assert_allclose(result["decoded"].final, vocab[ids],
                                               atol=1e-12, rtol=1e-12)
                    raw_output = np.asarray(result["decoded"].final, dtype="<f8").tobytes()
                    self.assertEqual(digest, hashlib.sha256(raw_output).digest())
                    _, sent = io.snapshot()
                    _, received = result["io"].snapshot()
                    self.assertEqual(sent["sent"], received["received"])
                    self.assertEqual(sent["received"], received["sent"])
                finally:
                    left.close()
                    right.close()
                    thread.join(2)

    def test_receiver_error_ack_identifies_failed_setup_or_message(self):
        _, _, vocab = fixture()
        sender = codec.prepare_sender(vocab, "coefficient")
        for cold in (False, True):
            with self.subTest(cold=cold):
                left, right = socket.socketpair()
                left.settimeout(2)
                right.settimeout(2)
                context = None if cold else codec.prepare_receiver(sender, "coefficient")
                holder = {}
                def worker():
                    holder["result"] = transport.receiver_exchange(
                        right, context, "coefficient", transport.PROFILES[0], cold=cold)
                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    io = transport.FrameIO(left, transport.PROFILES[0])
                    io.send_frame(b"malformed")
                    ack = io.recv_frame()
                    magic, status, digest = struct.unpack("<4sB3x32s", ack)
                    self.assertEqual(magic, b"SET2" if cold else b"ACK2")
                    self.assertEqual(status, 1)
                    self.assertEqual(digest, bytes(32))
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertIsNotNone(holder["result"]["error"])
                    self.assertIsNone(holder["result"]["decoded"])
                finally:
                    left.close()
                    right.close()
                    thread.join(2)

    def test_real_tcp_trial_reconciles_fresh_encoding_and_actual_endpoints(self):
        source, ids, vocab = fixture()
        encoded_sources = []
        real_encode = transport.encode_arm
        def observe_encode(actual_source, actual_ids, context, arm, precision="float64"):
            self.assertFalse(np.shares_memory(actual_source, source))
            self.assertFalse(np.shares_memory(actual_ids, ids))
            encoded_sources.append(actual_source)
            return real_encode(actual_source, actual_ids, context, arm, precision)
        with patch.object(transport, "encode_arm", side_effect=observe_encode):
            for cold in (False, True):
                with self.subTest(cold=cold):
                    record, artifacts = transport.run_trial(source, ids, vocab, "dual_direct",
                                                            transport.PROFILES[0], cold=cold)
                    self.assertEqual(record["status"], "completed", record)
                    self.assertTrue(all(record["checks"].values()))
                    self.assertIsInstance(record["wall_ns"], int)
                    self.assertGreaterEqual(record["wall_ns"], 0)
                    wire = artifacts["wire"]
                    self.assertEqual(wire["sender_sent"], wire["receiver_received"])
                    self.assertEqual(wire["sender_received"], wire["receiver_sent"])
                    self.assertEqual(record["logical_wire_bytes"],
                                     len(wire["sender_sent"]) + len(wire["sender_received"]))
                    np.testing.assert_allclose(artifacts["decoded"].final, vocab[ids],
                                               atol=1e-12, rtol=1e-12)
                    expected_frames = 2 if cold else 1
                    self.assertEqual(len(record["sender_io"]["sent_frames"]), expected_frames)
                    self.assertFalse(record["fatal_receiver_alive"])
        self.assertEqual(len(encoded_sources), 2)
        self.assertFalse(np.shares_memory(encoded_sources[0], encoded_sources[1]))
        self.assertEqual(set(inspect.signature(transport.run_trial).parameters),
                         {"source", "token_ids", "vocabulary", "arm", "profile", "cold"})
        self.assertEqual(set(inspect.signature(transport.receiver_exchange).parameters),
                         {"sock", "context", "arm", "profile", "cold"})

    def test_receiver_startup_is_outside_warm_clock_and_inside_cold_clock(self):
        source, ids, vocab = fixture()
        original_options = transport._socket_options
        for cold in (False, True):
            with self.subTest(cold=cold):
                markers = {}
                main_thread = threading.current_thread()
                def delayed_receiver_options(sock):
                    if threading.current_thread() is not main_thread:
                        time.sleep(0.025)
                        original_options(sock)
                        markers["ready"] = time.perf_counter_ns()
                    else:
                        original_options(sock)
                with patch.object(transport, "_socket_options", side_effect=delayed_receiver_options):
                    record, _ = transport.run_trial(source, ids, vocab, "dual_direct",
                                                     transport.PROFILES[0], cold=cold)
                self.assertEqual(record["status"], "completed", record)
                self.assertIn("ready", markers)
                if cold:
                    self.assertLessEqual(record["start_perf_counter_ns"], markers["ready"])
                    self.assertIn("receiver_ready", record["phases_ns"])
                else:
                    self.assertGreaterEqual(record["start_perf_counter_ns"], markers["ready"])
                    self.assertIn("receiver_ready_outside_timer", record["phases_ns"])

    def test_socket_setup_failure_returns_a_retained_error_record(self):
        source, ids, vocab = fixture()
        with patch.object(transport.socket, "socket", side_effect=OSError("unit setup failure")):
            record, artifacts = transport.run_trial(source, ids, vocab, "coefficient",
                                                    transport.PROFILES[0])
        self.assertEqual(record["status"], "error")
        self.assertIn("unit setup failure", record["error"])
        self.assertIsNone(record["wall_ns"])
        self.assertIsNone(artifacts["decoded"])
        self.assertEqual(artifacts["wire"], {})

    def test_post_timing_diagnostic_failure_preserves_decoded_array_and_raw_wire(self):
        source, ids, vocab = fixture()
        with patch.object(transport.FrameIO, "snapshot", side_effect=RuntimeError("unit snapshot failure")):
            record, artifacts = transport.run_trial(source, ids, vocab, "dual_direct",
                                                    transport.PROFILES[0])
        self.assertEqual(record["status"], "error")
        self.assertIn("unit snapshot failure", record["post_timing_error"])
        self.assertIsInstance(record["wall_ns"], int)
        self.assertIsNotNone(artifacts["decoded"])
        self.assertTrue(artifacts["payloads"]["message_sent"])
        self.assertTrue(artifacts["wire"]["sender_sent"])
        self.assertEqual(artifacts["wire"]["sender_sent"], artifacts["wire"]["receiver_received"])
        self.assertEqual(artifacts["wire"]["sender_received"], artifacts["wire"]["receiver_sent"])

    def test_fixed_plan_covers_every_declared_cell_and_only_those_cells(self):
        from latent_bridge.coefficient_study import build_plan
        plan = build_plan()
        self.assertEqual(plan, build_plan())
        self.assertEqual(len(plan), 720)
        self.assertEqual(len({r["trial_id"] for r in plan}), 720)
        fields = ("mode", "K", "d", "seed", "profile", "arm", "repetition")
        actual = Counter(tuple(row[key] for key in fields) for row in plan)
        expected = Counter()
        for (k, d), seed, profile, arm, repetition in itertools.product(
                SHAPES, SEEDS, ("loopback", "moderate", "constrained"), ARMS, range(4)):
            expected[("warm", k, d, seed, profile, arm, repetition)] += 1
        for (k, d), arm, repetition in itertools.product(SHAPES, ARMS, range(4)):
            expected[("cold", k, d, 17011, "constrained", arm, repetition)] += 1
        self.assertEqual(actual, expected)

    def test_fixed_plan_pairs_all_arms_and_satisfies_declared_position_balance(self):
        from latent_bridge.coefficient_study import build_plan
        plan = build_plan()
        seen_blocks = set()
        warm, cold = defaultdict(Counter), defaultdict(Counter)
        for start in range(0, len(plan), 6):
            block = plan[start:start + 6]
            ids = {row["block_id"] for row in block}
            self.assertEqual(len(ids), 1)
            identifier = next(iter(ids))
            self.assertNotIn(identifier, seen_blocks)
            seen_blocks.add(identifier)
            self.assertEqual([row["order_position"] for row in block], list(range(6)))
            self.assertEqual({row["arm"] for row in block}, set(ARMS))
            for key in ("mode", "fixture_id", "K", "d", "seed", "profile", "repetition"):
                self.assertEqual(len({row[key] for row in block}), 1)
            for row in block:
                if row["mode"] == "warm":
                    warm[(row["K"], row["d"], row["profile"], row["arm"])][row["order_position"]] += 1
                else:
                    cold[row["arm"]][row["order_position"]] += 1
        self.assertEqual(len(seen_blocks), 120)
        self.assertEqual(len(warm), 3 * 3 * 6)
        self.assertEqual(len(cold), 6)
        for counter in (*warm.values(), *cold.values()):
            self.assertEqual(counter, Counter({position: 2 for position in range(6)}))


    def test_paired_summary_and_hindsight_selection_use_the_declared_statistics(self):
        from latent_bridge.coefficient_study import build_plan, summarize_trials
        plan = build_plan()
        summary = summarize_trials(plan, synthetic_trial_records(plan))
        for case in summary["cases"]:
            self.assertEqual(case["hindsight_reference"]["chosen_arm"], "thin_direct")
            self.assertEqual(case["arms"]["thin_direct"]["median_wall_ns"], 15.5)
            self.assertEqual(case["arms"]["coefficient"]["median_wall_ns"], 25)
            self.assertEqual(case["hindsight_reference"]["coefficient_over_comparator"],
                             [10., 2. / 3., 1., 40.])
            self.assertEqual(case["hindsight_reference"]["median_ratio"], 5.5)
            self.assertNotEqual(case["hindsight_reference"]["median_ratio"], 25. / 15.5)
        shaped = summary["aggregates"]["shaped_warm"]["comparisons"]
        self.assertTrue(shaped["hindsight_reference"]["complete"])
        self.assertAlmostEqual(shaped["hindsight_reference"]["geometric_mean_case_median"], 5.5)
        self.assertAlmostEqual(shaped["dual_direct"]["geometric_mean_case_median"], 10.2)
        self.assertAlmostEqual(shaped["gram_guard"]["geometric_mean_case_median"], 5. / 12.)
        self.assertEqual(summary["worst_warm_case_paired_median_ratio"], 5.5)

    def test_one_failed_pair_cannot_shrink_the_fixed_success_population(self):
        from latent_bridge.coefficient_study import build_plan, summarize_trials
        plan = build_plan()
        rows = synthetic_trial_records(plan)
        broken = next(row for row in rows
                      if row["mode"] == "warm" and row["profile"] == "moderate"
                      and row["arm"] == "coefficient" and row["repetition"] == 0)
        broken["status"] = "error"
        summary = summarize_trials(plan, rows)
        group = summary["aggregates"]["shaped_warm"]["comparisons"]
        for name in ("dual_direct", "thin_direct", "gram_guard", "hindsight_reference"):
            self.assertFalse(group[name]["complete"])
            self.assertEqual(group[name]["completed_cases"], 17)
            self.assertIsNone(group[name]["geometric_mean_case_median"])
            self.assertIsNotNone(group[name]["available_only_descriptive_geometric_mean"])
        self.assertFalse(summary["all_warm_hindsight_complete"])
        self.assertIsNone(summary["worst_warm_case_paired_median_ratio"])

    def test_f32_missing_fast_reference_is_descriptive_without_relabeling_route(self):
        from latent_bridge import alignment_core
        from latent_bridge.coefficient_study import diagnose_one
        source, ids, vocab = fixture()
        dense = alignment_core.dense_reference(source, vocab[ids], details=False)
        refs = {"dense": {"pre": dense.pre,
                          "processed": alignment_core.postprocess(dense.pre, vocab)}}
        sample = {"id": "unit-reference-policy", "suite": "stress",
                  "S": source, "R": vocab[ids], "vocab": vocab, "ids": ids}
        row, encoded, decoded = diagnose_one(
            sample, "gram_guard", "float32", refs, MemoryDescriptions())
        self.assertEqual(row["status"], "checked")
        self.assertTrue(row["required_passed"])
        self.assertFalse(row["fallback"])
        self.assertFalse(encoded.metadata["fallback"])
        self.assertIsNotNone(decoded.pre)
        self.assertEqual(row["preferred_reference_method"], "dual")
        self.assertFalse(row["preferred_reference_supported"])
        self.assertEqual(row["reference_method"], "dense")
        self.assertIn("descriptive", row["evaluation_reference_reason"])
        self.assertIsNotNone(row["pre_error"])
        self.assertIsNotNone(row["calibrated_error"])
        self.assertIsNotNone(row["final_error"])
        f64, _, _ = diagnose_one(sample, "gram_guard", "float64", refs, MemoryDescriptions())
        self.assertEqual(f64["status"], "error")
        self.assertFalse(f64["required_passed"])
        self.assertFalse(f64["required_checks"]["completed_diagnostic"])


if __name__ == "__main__":
    unittest.main()
