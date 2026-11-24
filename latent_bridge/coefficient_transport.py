"""Actual, bounded loopback transport for the frozen coefficient-packet study.

This module has no oracle/reference-prefix inputs. The receiver gets only its
socket, legitimate session context, arm identifier, and link profile.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import socket
import struct
import threading
import time

import numpy as np

from .coefficient_packet import (
    ARMS, prepare_sender, prepare_receiver, encode_arm, decode_arm,
    encode_setup, decode_setup,
)

MAX_FRAME_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 16 * 1024
SOCKET_TIMEOUT_SECONDS = 10.0
LENGTH = struct.Struct(">I")
ACK = struct.Struct("<4sB3x32s")
ZERO_DIGEST = bytes(32)


class TransportError(ValueError):
    """Malformed transport or failed peer operation."""


@dataclass(frozen=True)
class Profile:
    name: str
    bytes_per_second: float | None
    reply_holdoff_seconds: float

    def __post_init__(self):
        if self.bytes_per_second is not None and (
            not math.isfinite(self.bytes_per_second) or self.bytes_per_second <= 0
        ):
            raise ValueError("pacing rate must be positive and finite")
        if not math.isfinite(self.reply_holdoff_seconds) or self.reply_holdoff_seconds < 0:
            raise ValueError("reply holdoff must be finite and nonnegative")


PROFILES = (
    Profile("loopback", None, 0.0),
    Profile("moderate", 10_000_000.0, 0.005),
    Profile("constrained", 1_000_000.0, 0.020),
)


def canonical_output_bytes(array):
    a = np.asarray(array, dtype="<f8", order="C")
    if a.ndim != 2 or not np.isfinite(a).all():
        raise TransportError("decoded output must be a finite matrix")
    return a.tobytes(order="C")


def make_ack(magic, digest, status=0):
    if magic not in (b"ACK2", b"SET2") or status not in (0, 1):
        raise TransportError("invalid ACK kind/status")
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise TransportError("ACK digest must contain 32 bytes")
    if status and digest != ZERO_DIGEST:
        raise TransportError("error ACK must use the zero digest")
    return ACK.pack(magic, status, digest)


def validate_ack(payload, magic, expected_digest=None):
    if len(payload) != ACK.size:
        raise TransportError("ACK payload is not exactly 40 bytes")
    actual_magic, status, digest = ACK.unpack(payload)
    if actual_magic != magic or payload[5:8] != bytes(3) or status not in (0, 1):
        raise TransportError("malformed ACK")
    if status:
        if digest != ZERO_DIGEST:
            raise TransportError("error ACK has a nonzero digest")
        raise TransportError("receiver reported an error")
    if expected_digest is not None and digest != expected_digest:
        raise TransportError("setup receipt digest mismatch")
    return digest


class FrameIO:
    """Trace actual send/recv bytes; hash traces only after measurement.

    Optional clock/sleep injection is for deterministic contract tests. The study
    calls this class with the real time module and actual TCP sockets.
    """

    def __init__(self, sock, profile, *, clock_ns=None, sleep=None):
        self.sock = sock
        self.profile = profile
        self.clock_ns = time.perf_counter_ns if clock_ns is None else clock_ns
        self.sleep = time.sleep if sleep is None else sleep
        self.sent_chunks = []
        self.received_chunks = []
        self.sent_frames = []
        self.received_frames = []
        self.waits = []

    def _wait(self, kind, seconds):
        if seconds <= 0:
            return
        before = self.clock_ns()
        error = None
        try:
            self.sleep(seconds)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            self.waits.append({
                "kind": kind, "requested_seconds": float(seconds),
                "observed_ns": self.clock_ns() - before, "error": error,
            })

    def send_frame(self, payload, *, reply=False):
        if not isinstance(payload, bytes):
            raise TransportError("payload must be immutable bytes")
        if not 0 < len(payload) <= MAX_FRAME_BYTES:
            raise TransportError("payload outside the fixed frame limit")
        frame = LENGTH.pack(len(payload)) + payload
        attempt = {
            "payload_bytes": len(payload), "declared_wire_bytes": len(frame),
            "actual_wire_bytes": 0, "complete": False, "error": None,
        }
        self.sent_frames.append(attempt)
        try:
            if reply:
                self._wait("reply_holdoff", self.profile.reply_holdoff_seconds)
            for start in range(0, len(frame), CHUNK_BYTES):
                chunk = memoryview(frame)[start:start + CHUNK_BYTES]
                if self.profile.bytes_per_second is not None:
                    self._wait("pacing", len(chunk) / self.profile.bytes_per_second)
                offset = 0
                while offset < len(chunk):
                    count = self.sock.send(chunk[offset:])
                    if count <= 0:
                        raise ConnectionError("socket stopped sending")
                    sent = bytes(chunk[offset:offset + count])
                    self.sent_chunks.append(sent)
                    attempt["actual_wire_bytes"] += count
                    offset += count
            attempt["complete"] = True
        except BaseException as exc:
            attempt["error"] = type(exc).__name__ + ": " + str(exc)
            raise

    def _read_exact(self, count, attempt):
        parts = []
        remaining = count
        while remaining:
            value = self.sock.recv(min(65536, remaining))
            if not value:
                raise EOFError("peer closed within a frame")
            self.received_chunks.append(value)
            parts.append(value)
            attempt["actual_wire_bytes"] += len(value)
            remaining -= len(value)
        return b"".join(parts)

    def recv_frame(self):
        attempt = {
            "payload_bytes": None, "declared_wire_bytes": None,
            "actual_wire_bytes": 0, "complete": False, "error": None,
        }
        self.received_frames.append(attempt)
        try:
            header = self._read_exact(LENGTH.size, attempt)
            size = LENGTH.unpack(header)[0]
            attempt["payload_bytes"] = size
            attempt["declared_wire_bytes"] = LENGTH.size + size
            if not 0 < size <= MAX_FRAME_BYTES:
                raise TransportError("received frame outside the fixed limit")
            payload = self._read_exact(size, attempt)
            attempt["complete"] = True
            return payload
        except BaseException as exc:
            attempt["error"] = type(exc).__name__ + ": " + str(exc)
            raise

    def snapshot(self):
        """Post-timing byte hashes, counts and complete/partial frame records."""
        sent = b"".join(self.sent_chunks)
        received = b"".join(self.received_chunks)
        totals = {}
        for kind in ("pacing", "reply_holdoff"):
            selected = [x for x in self.waits if x["kind"] == kind]
            totals[kind] = {
                "calls": len(selected),
                "requested_seconds": sum(x["requested_seconds"] for x in selected),
                "observed_ns": sum(x["observed_ns"] for x in selected),
            }
        return {
            "sent_bytes": len(sent), "received_bytes": len(received),
            "sent_sha256": hashlib.sha256(sent).hexdigest(),
            "received_sha256": hashlib.sha256(received).hexdigest(),
            "sent_frames": list(self.sent_frames),
            "received_frames": list(self.received_frames),
            "waits": list(self.waits), "wait_totals": totals,
        }, {"sent": sent, "received": received}


def receiver_exchange(sock, context, arm, profile, *, cold=False):
    """Receive from bytes only; no source, fixture, label, or oracle argument."""
    io = FrameIO(sock, profile)
    answer = {
        "io": io, "decoded": None, "message_payload": None,
        "setup_payload": None, "setup_ack": None, "output_ack": None,
        "phases_ns": {}, "error": None,
    }
    error_magic = b"SET2" if cold else b"ACK2"
    try:
        if cold:
            if context is not None:
                raise TransportError("cold receiver must start without a context")
            t = time.perf_counter_ns()
            setup = io.recv_frame()
            answer["phases_ns"]["receive_setup"] = time.perf_counter_ns() - t
            answer["setup_payload"] = setup
            t = time.perf_counter_ns()
            context = decode_setup(setup, arm)
            answer["phases_ns"]["construct_context"] = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            digest = hashlib.sha256(setup).digest()
            answer["phases_ns"]["hash_setup"] = time.perf_counter_ns() - t
            response = make_ack(b"SET2", digest)
            answer["setup_ack"] = response
            t = time.perf_counter_ns()
            io.send_frame(response, reply=True)
            answer["phases_ns"]["send_setup_ack"] = time.perf_counter_ns() - t
        elif context is None:
            raise TransportError("warm receiver requires a legitimate context")
        error_magic = b"ACK2"
        t = time.perf_counter_ns()
        payload = io.recv_frame()
        answer["phases_ns"]["receive_message"] = time.perf_counter_ns() - t
        answer["message_payload"] = payload
        t = time.perf_counter_ns()
        decoded = decode_arm(payload, context, arm)
        answer["phases_ns"]["decode"] = time.perf_counter_ns() - t
        answer["decoded"] = decoded
        t = time.perf_counter_ns()
        digest = hashlib.sha256(canonical_output_bytes(decoded.final)).digest()
        answer["phases_ns"]["hash_output"] = time.perf_counter_ns() - t
        response = make_ack(b"ACK2", digest)
        answer["output_ack"] = response
        t = time.perf_counter_ns()
        io.send_frame(response, reply=True)
        answer["phases_ns"]["send_output_ack"] = time.perf_counter_ns() - t
    except BaseException as exc:
        answer["error"] = type(exc).__name__ + ": " + str(exc)
        try:
            response = make_ack(error_magic, ZERO_DIGEST, status=1)
            if error_magic == b"SET2":
                answer["setup_ack"] = response
            else:
                answer["output_ack"] = response
            io.send_frame(response, reply=True)
        except BaseException as send_exc:
            answer["error_ack_error"] = type(send_exc).__name__ + ": " + str(send_exc)
    return answer


def _socket_options(sock):
    sock.settimeout(SOCKET_TIMEOUT_SECONDS)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _close(sock):
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def run_trial(source, token_ids, vocabulary, arm, profile, *, cold=False):
    """Run one fresh real-TCP exchange and retain partial evidence on failure.

    Inputs contain no reference output or precomputed per-message statistics.
    This function does not decide numerical eligibility or compare to an oracle.
    """
    if arm not in ARMS or not isinstance(profile, Profile):
        raise ValueError("unknown arm/profile")
    whole_start = time.perf_counter_ns()
    record = {
        "status": "error", "cold": bool(cold), "arm": arm, "profile": profile.name,
        "wall_ns": None, "phases_ns": {}, "receiver_phases_ns": {},
        "error": None, "receiver_error": None, "checks": {},
        "sender_metadata": None, "receiver_metadata": None,
    }
    payloads = {}
    artifacts = {"decoded": None, "payloads": payloads, "wire": {}}
    listener = client = None
    worker = None
    holder = {}
    ready = threading.Event()
    sender_io = None
    started = finished = None
    ack_digest = None
    sender_context = receiver_context = None
    try:
        t = time.perf_counter_ns()
        source = np.array(source, dtype=np.float64, order="C", copy=True)
        token_ids = np.array(token_ids, dtype=np.uint32, order="C", copy=True)
        vocabulary = np.array(vocabulary, dtype=np.float64, order="C", copy=True)
        record["phases_ns"]["input_copy_outside_timer"] = time.perf_counter_ns() - t
        if not cold:
            t = time.perf_counter_ns()
            sender_context = prepare_sender(vocabulary, arm)
            record["phases_ns"]["sender_context_outside_timer"] = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            receiver_context = prepare_receiver(sender_context, arm)
            record["phases_ns"]["receiver_context_outside_timer"] = time.perf_counter_ns() - t

        t = time.perf_counter_ns()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.settimeout(SOCKET_TIMEOUT_SECONDS)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        address = listener.getsockname()
        record["phases_ns"]["listener_outside_timer"] = time.perf_counter_ns() - t

        def accept_and_receive():
            accepted = None
            try:
                accepted, _ = listener.accept()
                holder["socket"] = accepted
                _socket_options(accepted)
                ready.set()
                holder["result"] = receiver_exchange(
                    accepted, receiver_context, arm, profile, cold=cold
                )
            except BaseException as exc:
                holder["thread_error"] = type(exc).__name__ + ": " + str(exc)
            finally:
                _close(accepted)

        worker = threading.Thread(target=accept_and_receive, daemon=True)
        worker.start()
        if cold:
            started = time.perf_counter_ns()
        t = time.perf_counter_ns()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _socket_options(client)
        client.connect(address)
        record["phases_ns"]["client_connect" + ("" if cold else "_outside_timer")] = (
            time.perf_counter_ns() - t
        )
        t = time.perf_counter_ns()
        if not ready.wait(SOCKET_TIMEOUT_SECONDS):
            raise TimeoutError("receiver endpoint did not become ready")
        record["phases_ns"]["receiver_ready" + ("" if cold else "_outside_timer")] = (
            time.perf_counter_ns() - t
        )
        sender_io = FrameIO(client, profile)
        if cold:
            t = time.perf_counter_ns()
            sender_context = prepare_sender(vocabulary, arm)
            record["phases_ns"]["sender_context"] = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            setup = encode_setup(sender_context, arm)
            payloads["setup_sent"] = setup
            setup_digest = hashlib.sha256(setup).digest()
            record["phases_ns"]["encode_and_hash_setup"] = time.perf_counter_ns() - t
            t = time.perf_counter_ns()
            sender_io.send_frame(setup)
            setup_ack = sender_io.recv_frame()
            payloads["setup_ack_received"] = setup_ack
            validate_ack(setup_ack, b"SET2", setup_digest)
            record["phases_ns"]["setup_exchange"] = time.perf_counter_ns() - t
        else:
            started = time.perf_counter_ns()

        t = time.perf_counter_ns()
        encoded = encode_arm(source, token_ids, sender_context, arm, precision="float64")
        record["phases_ns"]["encode"] = time.perf_counter_ns() - t
        payloads["message_sent"] = encoded.packet
        record["sender_metadata"] = encoded.metadata
        t = time.perf_counter_ns()
        sender_io.send_frame(encoded.packet)
        record["phases_ns"]["send_message"] = time.perf_counter_ns() - t
        t = time.perf_counter_ns()
        output_ack = sender_io.recv_frame()
        payloads["output_ack_received"] = output_ack
        ack_digest = validate_ack(output_ack, b"ACK2")
        record["phases_ns"]["receive_and_validate_ack"] = time.perf_counter_ns() - t
        finished = time.perf_counter_ns()
        record["status"] = "completed"
    except BaseException as exc:
        finished = time.perf_counter_ns()
        record["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if started is not None:
            if finished is None:
                finished = time.perf_counter_ns()
            record["wall_ns"] = finished - started
            record["start_perf_counter_ns"] = started
            record["end_perf_counter_ns"] = finished
        _close(client)
        if worker is not None:
            worker.join(SOCKET_TIMEOUT_SECONDS + 1.0)
            if worker.is_alive():
                _close(holder.get("socket"))
                _close(listener)
                listener = None
                worker.join(1.0)
                record["error"] = (record["error"] or "") + "; receiver thread did not stop"
                record["status"] = "error"
            record["fatal_receiver_alive"] = worker.is_alive()
        _close(listener)

    result = holder.get("result")
    if result is not None:
        artifacts["decoded"] = result.get("decoded")
    try:
        # All trace hashing, cross-endpoint comparisons and retained-array checks
        # occur after the measured end, with the receiver thread already joined.
        if sender_io is not None:
            record["sender_io"], wire = sender_io.snapshot()
            artifacts["wire"].update({"sender_" + k: v for k, v in wire.items()})
        result = holder.get("result")
        if result is not None:
            record["receiver_io"], wire = result["io"].snapshot()
            artifacts["wire"].update({"receiver_" + k: v for k, v in wire.items()})
            record["receiver_phases_ns"] = result["phases_ns"]
            record["receiver_error"] = result["error"]
            artifacts["decoded"] = result["decoded"]
            if result["decoded"] is not None:
                record["receiver_metadata"] = result["decoded"].metadata
            for key in ("message_payload", "setup_payload", "setup_ack", "output_ack"):
                if result[key] is not None:
                    payloads["receiver_" + key] = result[key]
            if result["error"]:
                record["status"] = "error"
            wire = artifacts["wire"]
            record["checks"]["sender_to_receiver_exact"] = (
                wire.get("sender_sent") == wire.get("receiver_received")
            )
            record["checks"]["receiver_to_sender_exact"] = (
                wire.get("receiver_sent") == wire.get("sender_received")
            )
            decoded = result["decoded"]
            try:
                record["checks"]["output_receipt_hash"] = (
                    decoded is not None and ack_digest is not None and
                    hashlib.sha256(canonical_output_bytes(decoded.final)).digest() == ack_digest
                )
            except Exception as exc:
                record["checks"]["output_receipt_hash"] = False
                record["post_timing_output_error"] = type(exc).__name__ + ": " + str(exc)
            expected_frames = 2 if cold else 1
            for endpoint in ("sender_io", "receiver_io"):
                for direction in ("sent_frames", "received_frames"):
                    rows = record[endpoint][direction]
                    record["checks"][endpoint + "_" + direction + "_complete"] = (
                        len(rows) == expected_frames and all(
                            row["complete"] and row["actual_wire_bytes"] == row["declared_wire_bytes"]
                            for row in rows
                        )
                    )
            record["logical_wire_bytes"] = (
                record.get("sender_io", {}).get("sent_bytes", 0) +
                record.get("sender_io", {}).get("received_bytes", 0)
            )
        elif holder.get("thread_error"):
            record["receiver_error"] = holder["thread_error"]
            record["status"] = "error"
        if record["status"] == "completed" and not all(record["checks"].values()):
            record["status"] = "error"
            record["error"] = "post-timing byte/ACK reconciliation failed"
        if record["status"] == "completed" and not record["checks"]:
            record["status"] = "error"
            record["error"] = "missing receiver completion record"
    except BaseException as exc:
        record["status"] = "error"
        record["post_timing_error"] = type(exc).__name__ + ": " + str(exc)
        # Preserve un-hashed partial traces even if a diagnostic snapshot fails.
        for label, endpoint in (("sender", sender_io),
                                ("receiver", None if result is None else result.get("io"))):
            if endpoint is not None:
                for direction, chunks in (("sent", endpoint.sent_chunks),
                                          ("received", endpoint.received_chunks)):
                    try:
                        artifacts["wire"][label + "_" + direction] = b"".join(chunks)
                    except Exception:
                        pass
    record["whole_call_ns"] = time.perf_counter_ns() - whole_start
    return record, artifacts
