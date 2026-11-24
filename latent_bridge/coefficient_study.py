"""Frozen coefficient-packet v2 numerical and real-TCP study.

No algorithm/grid/timing settings are exposed as command-line overrides.
References are used only outside measured transport, never as encoder inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import numpy as np

from .alignment_core import (
    helmert, dense_reference, thin_reference, dual_alignment, postprocess, Unsupported,
)
from .alignment_packet import HEADER, DTYPES, encode_direct, decode_packet
from .coefficient_packet import (
    ARMS, prepare_sender, prepare_receiver, encode_arm, decode_arm, postprocess_v2,
)
from .coefficient_transport import (
    PROFILES, run_trial, canonical_output_bytes,
)

DESIGN_COMMIT = "49cfba7261326d58e174e51c8ddf5abba6f4a966"
DESIGN_SHA256 = "b8307677f5320c3413b5a1007bdfe26230a844c57503d53b4443e5fc542647ba"
SHAPES = ((12, 192), (36, 512), (60, 960))
SEEDS = (17011, 17027, 17041)
STRESS_SEEDS = (19001, 19013)
REPETITIONS = 4
COMPARATORS = ("dual_direct", "thin_direct", "gram_guard")
ALL_COMPARATORS = COMPARATORS + ("gram_original",)
LAM, EPS, ALPHA = 1e-3, 1e-6, 0.3
EXPECTED_TRIALS = 720
ATOL, RTOL = 1e-10, 1e-8
WRAPPER_FILES = {"launcher.json", "launch_source_manifest.json", "worker.log"}


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def json_bytes(value):
    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_bytes(path, raw):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def emit(stream, value):
    stream.write(json.dumps(clean(value), sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def rng(*seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))


def fixture_id(k, d, seed, geometry=None, warmup=False):
    prefix = "warmup" if warmup else ("primary" if geometry is None else "stress")
    suffix = "" if geometry is None else "-g" + str(geometry)
    return f"{prefix}-k{k}-d{d}-seed{seed}{suffix}"


def build_plan():
    """The frozen full Cartesian grid, then one permutation of all 120 blocks."""
    blocks = []
    for shape_index, (k, d) in enumerate(SHAPES):
        for profile_index, profile in enumerate(PROFILES):
            permutation = rng(23017, k, d, profile_index, 0).permutation(len(ARMS)).tolist()
            for seed_index, seed in enumerate(SEEDS):
                for repetition in range(REPETITIONS):
                    rotation = (2 * seed_index + repetition) % len(ARMS)
                    order = permutation[rotation:] + permutation[:rotation]
                    block = f"warm-k{k}-d{d}-seed{seed}-{profile.name}-r{repetition}"
                    blocks.append([{
                        "trial_id": block + "-" + ARMS[arm_index],
                        "block_id": block, "mode": "warm",
                        "fixture_id": fixture_id(k, d, seed), "K": k, "d": d,
                        "seed": seed, "profile": profile.name,
                        "repetition": repetition, "order_position": position,
                        "arm": ARMS[arm_index],
                    } for position, arm_index in enumerate(order)])
    cold_permutation = rng(23017, 0, 0, 2, 1).permutation(len(ARMS)).tolist()
    for shape_index, (k, d) in enumerate(SHAPES):
        for repetition in range(REPETITIONS):
            rotation = (2 * shape_index + repetition) % len(ARMS)
            order = cold_permutation[rotation:] + cold_permutation[:rotation]
            block = f"cold-k{k}-d{d}-seed{SEEDS[0]}-constrained-r{repetition}"
            blocks.append([{
                "trial_id": block + "-" + ARMS[arm_index],
                "block_id": block, "mode": "cold",
                "fixture_id": fixture_id(k, d, SEEDS[0]), "K": k, "d": d,
                "seed": SEEDS[0], "profile": "constrained",
                "repetition": repetition, "order_position": position,
                "arm": ARMS[arm_index],
            } for position, arm_index in enumerate(order)])
    order = rng(23029).permutation(len(blocks)).tolist()
    rows = [row for index in order for row in blocks[index]]
    for index, row in enumerate(rows):
        row["sequence_index"] = index
    if len(rows) != EXPECTED_TRIALS or len({x["trial_id"] for x in rows}) != EXPECTED_TRIALS:
        raise RuntimeError("fixed plan construction failed")
    return rows


def orthogonal(rows, columns, seed, k, d, slot):
    a = rng(seed, k, d, slot).normal(size=(rows, columns))
    q, r = np.linalg.qr(a, mode="reduced")
    signs = np.where(np.diag(r) >= 0, 1.0, -1.0)
    q = q * signs
    if not np.isfinite(q).all():
        raise ValueError("nonfinite QR construction")
    return q


def make_fixture(k, d, seed, geometry=None, *, warmup=False):
    """No resampling or selection on rank, margins, speed or guard outcomes."""
    r = k - 1
    h = helmert(k)
    qs = orthogonal(r, r, seed, k, d, 1)
    vs = orthogonal(d, r, seed, k, d, 2)
    qr = orthogonal(r, r, seed, k, d, 3)
    vr = orthogonal(d, r, seed, k, d, 4)
    ss, sr = np.geomspace(1.0, 0.2, r), np.geomspace(1.0, 0.1, r)
    mu_s = rng(seed, k, d, 5).normal(scale=0.2 / np.sqrt(d), size=d)
    mu_r = rng(seed, k, d, 6).normal(scale=0.1 / np.sqrt(d), size=d)
    if geometry == 1:
        ss[-1] = 0.0
    elif geometry == 2:
        sr[-1] = 0.0
    elif geometry == 3:
        ss[1:] = sr[1:] = 0.0
    elif geometry == 4:
        ss[:] = 0.0
    elif geometry == 5:
        sr[:] = 0.0
    elif geometry == 6:
        ss[-1] = 1e-5
    elif geometry == 7:
        sr[-1] = 1e-5
    elif geometry == 8:
        qs = qr = np.eye(r)
        ss, sr = np.ones(r), np.ones(r)
        ss[-1], sr[0] = 2e-4, 2e-4
    elif geometry is not None:
        raise ValueError("unknown fixed stress geometry")
    source = (h @ qs * ss) @ vs.T + mu_s
    reference = (h @ qr * sr) @ vr.T + mu_r
    target = float(np.median(np.linalg.norm(reference, axis=1)))
    distractors = rng(seed, k, d, 7).normal(size=(256, d))
    lengths = np.linalg.norm(distractors, axis=1)
    if not np.isfinite(target) or target <= 0 or np.any(lengths <= 0):
        raise ValueError("zero/nonfinite distractor scale; no resampling")
    distractors = distractors * (target / lengths)[:, None]
    vocabulary = np.vstack([reference, distractors])
    ids = np.arange(k, dtype="<u4") if geometry != 5 else np.zeros(k, dtype="<u4")
    reference = vocabulary[ids].copy()
    if not np.isfinite(source).all() or not np.isfinite(vocabulary).all():
        raise ValueError("nonfinite constructed input; no resampling")
    return {
        "id": fixture_id(k, d, seed, geometry, warmup),
        "suite": "warmup" if warmup else ("primary" if geometry is None else "stress"),
        "K": k, "d": d, "seed": seed, "geometry": geometry,
        "S": source, "R": reference, "vocab": vocabulary, "ids": ids,
        "construction": {"source_singular_values": ss.tolist(),
                         "reference_singular_values": sr.tolist(),
                         "slots": {"Qs": 1, "Vs": 2, "Qr": 3, "Vr": 4,
                                   "muS": 5, "muR": 6, "distractors": 7},
                         "repeated_id": geometry == 5,
                         "fixed_diagonal_reduced_grams": geometry == 8},
    }


class Retainer:
    """Write each novel object atomically; keep partial output independently readable."""
    def __init__(self, directory):
        self.directory = directory
        (directory / "objects").mkdir()
        self.seen = {}
        self.index = (directory / "object_manifest.jsonl").open("w", encoding="utf-8")

    def raw(self, raw, kind="bytes", description=None):
        key = sha(raw)
        suffix = ".npy" if kind == "array" else ".bin"
        relative = "objects/" + key + suffix
        if relative not in self.seen:
            atomic_bytes(self.directory / relative, raw)
            entry = {"path": relative, "sha256": key, "bytes": len(raw), "kind": kind}
            if description:
                entry.update(description)
            self.seen[relative] = entry
            emit(self.index, entry)
        return dict(self.seen[relative])

    def array(self, array):
        a = np.ascontiguousarray(array)
        if a.dtype.hasobject:
            raise ValueError("object arrays cannot be retained")
        buffer = io.BytesIO()
        np.save(buffer, a, allow_pickle=False)
        return self.raw(buffer.getvalue(), "array", {
            "shape": list(a.shape), "dtype": a.dtype.str,
            "data_sha256": sha(a.tobytes(order="C")),
            "data_bytes": int(a.nbytes),
        })

    def decoded(self, value):
        if value is None:
            return None
        return {"mode": value.mode, "final": self.array(value.final),
                "pre": None if value.pre is None else self.array(value.pre),
                "metadata": clean(value.metadata)}

    def artifacts(self, artifacts):
        return {
            "decoded": self.decoded(artifacts.get("decoded")),
            "payloads": {key: self.raw(raw) for key, raw in artifacts.get("payloads", {}).items()},
            "wire": {key: self.raw(raw) for key, raw in artifacts.get("wire", {}).items()},
        }

    def close(self):
        self.index.close()


def error_metrics(actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "error": "shape or finite-value mismatch"}
    absolute = float(np.linalg.norm(a - b))
    reference_norm, actual_norm = float(np.linalg.norm(b)), float(np.linalg.norm(a))
    return {
        "absolute_frobenius": absolute,
        "relative_frobenius": absolute / reference_norm if reference_norm else (
            0.0 if absolute == 0 else None),
        "maximum_absolute_coordinate": float(np.max(np.abs(a - b))) if a.size else 0.0,
        "reference_norm": reference_norm, "actual_norm": actual_norm,
        "norm_change": actual_norm - reference_norm,
        "passed": bool(absolute <= ATOL + RTOL * reference_norm),
    }


def stable_rows(processed):
    return ((processed["margins"] > 1e-6) & (processed["pre_norms"] > 1e-6) &
            (processed["query_norms"] > 1e-12) &
            (processed["vocab_norms"][processed["ids"]] > 1e-12))


def actual_snap_ids(encoded_metadata, decoded_metadata):
    values = decoded_metadata.get("snap_ids")
    if values is None:
        values = encoded_metadata.get("snap_ids")
    return None if values is None else np.asarray(values, dtype=np.int64)


def raw_wire_exact(packet, final):
    fields = HEADER.unpack_from(packet)
    mode, code, k, d = fields[2], fields[3], fields[5], fields[6]
    if mode not in (3, 4):
        return None
    if code not in DTYPES:
        return False
    wire = np.asarray(final, dtype=DTYPES[code], order="C").tobytes()
    return len(wire) == k * d * DTYPES[code].itemsize and wire == packet[HEADER.size:]


def references_for(fixture, retainer):
    refs, record = {}, {}
    for name, fn in (("dense", dense_reference), ("dual", dual_alignment),
                     ("thin", thin_reference)):
        try:
            started = time.perf_counter_ns()
            result = fn(fixture["S"], fixture["R"], LAM, EPS, details=False)
            processed = postprocess(result.pre, fixture["vocab"], ALPHA)
            elapsed = time.perf_counter_ns() - started
            refs[name] = {"pre": result.pre, "processed": processed}
            record[name] = {"status": "computed", "outside_transport_ns": elapsed,
                            "pre": retainer.array(result.pre),
                            "calibrated": retainer.array(processed["calibrated"]),
                            "final": retainer.array(processed["final"]),
                            "snap_ids": processed["ids"].tolist(),
                            "margins": processed["margins"].tolist()}
        except Exception as exc:
            record[name] = {"status": "unsupported" if isinstance(exc, Unsupported) else "error",
                            "error": type(exc).__name__ + ": " + str(exc)}
    return refs, record


def reference_name(arm, fallback):
    if arm == "token_lookup":
        return "token_lookup"
    if fallback:
        return "dense"
    return "thin" if arm == "thin_direct" else "dual"


def diagnose_one(fixture, arm, precision, refs, retainer):
    row = {"fixture_id": fixture["id"], "suite": fixture["suite"], "arm": arm,
           "precision": precision, "status": "error", "required_checks": {},
           "fidelity_errors_are_descriptive": precision == "float32",
           "equivalent_aligned_prefix": arm != "token_lookup"}
    encoded = decoded = None
    try:
        t = time.perf_counter_ns()
        sender = prepare_sender(fixture["vocab"], arm)
        receiver = prepare_receiver(sender, arm)
        row["context_setup_outside_encode_decode_ns"] = time.perf_counter_ns() - t
        t = time.perf_counter_ns()
        encoded = encode_arm(fixture["S"], fixture["ids"], sender, arm, precision=precision)
        row["encode_ns"] = time.perf_counter_ns() - t
        row["packet"] = retainer.raw(encoded.packet)
        row["sender_metadata"] = clean(encoded.metadata)
        t = time.perf_counter_ns()
        decoded = decode_arm(encoded.packet, receiver, arm)
        row["decode_ns"] = time.perf_counter_ns() - t
        row["decoded"] = retainer.decoded(decoded)
        row["receiver_metadata"] = clean(decoded.metadata)
        fallback = bool(encoded.metadata.get("fallback", False))
        row["fallback"] = fallback
        name = reference_name(arm, fallback)
        row["reference_method"] = name
        raw_exact = raw_wire_exact(encoded.packet, decoded.final)
        if raw_exact is not None:
            row["required_checks"]["raw_roundtrip_exact_at_wire_dtype"] = raw_exact
        row["required_checks"]["finite_final"] = bool(np.isfinite(decoded.final).all())
        fields = HEADER.unpack_from(encoded.packet)
        mode, code, k, d, payload_bytes = fields[2], fields[3], fields[5], fields[6], fields[7]
        width = 0 if mode == 5 else DTYPES[code].itemsize
        expected = HEADER.size + (
            4 * k if mode == 5 else
            k * d * width if mode in (3, 4) else
            4 * k + width * k * (k - 1) // 2
        )
        row["required_checks"]["exact_serialized_size"] = (
            len(encoded.packet) == expected and payload_bytes == expected - HEADER.size
        )
        if name == "token_lookup":
            reference = fixture["R"]
            row["final_error"] = error_metrics(decoded.final, reference)
            row["required_checks"]["token_lookup_exact"] = (
                canonical_output_bytes(decoded.final) == canonical_output_bytes(reference)
            )
            row["actual_internal_snap_ids"] = None
        else:
            row["preferred_reference_method"] = name
            row["preferred_reference_supported"] = name in refs
            if precision == "float32" and name not in refs and "dense" in refs:
                # Rounding may cross a fixed rank guard. Preserve the actual
                # accepted packet and compare descriptively to the dense rule;
                # this does not change the codec route or its fallback flag.
                name = "dense"
                row["reference_method"] = name
                row["evaluation_reference_reason"] = (
                    "unrounded fast reference unsupported; descriptive dense reference"
                )
            if name not in refs:
                raise RuntimeError("no supported declared reference: " + name)
            ref = refs[name]
            processed = ref["processed"]
            reference = processed["final"]
            row["final_error"] = error_metrics(decoded.final, reference)
            mask = stable_rows(processed)
            row["stable_rows"] = mask.tolist()
            row["stable_row_count"] = int(mask.sum())
            row["reference_margins"] = processed["margins"].tolist()
            row["reference_internal_snap_ids"] = processed["ids"].tolist()
            ids = actual_snap_ids(encoded.metadata, decoded.metadata)
            row["actual_internal_snap_ids"] = None if ids is None else ids.tolist()
            row["internal_snap_changes"] = (
                None if ids is None else int(np.count_nonzero(ids != processed["ids"]))
            )
            if decoded.pre is not None:
                # Diagnostic recomputation is outside both recorded codec calls.
                actual_post = postprocess_v2(decoded.pre, receiver)
                row["actual_calibrated"] = retainer.array(actual_post["calibrated"])
                row["actual_pre_norms"] = actual_post["pre_norms"].tolist()
                row["actual_calibrated_norms"] = actual_post["query_norms"].tolist()
                row["actual_recomputed_margins"] = actual_post["margins"].tolist()
                row["pre_error"] = error_metrics(decoded.pre, ref["pre"])
                row["calibrated_error"] = error_metrics(
                    actual_post["calibrated"], processed["calibrated"]
                )
                if precision == "float64":
                    row["required_checks"]["pre_reference"] = row["pre_error"]["passed"]
                    row["required_checks"]["calibrated_reference"] = row["calibrated_error"]["passed"]
            else:
                row["pre_error"] = row["calibrated_error"] = None
            if precision == "float64":
                if np.any(mask):
                    row["stable_final_error"] = error_metrics(decoded.final[mask], reference[mask])
                    row["required_checks"]["stable_final_reference"] = row["stable_final_error"]["passed"]
                    if ids is not None:
                        row["required_checks"]["stable_internal_snaps"] = bool(
                            np.array_equal(ids[mask], processed["ids"][mask])
                        )
                    elif not fallback:
                        row["required_checks"]["native_internal_snaps_available"] = False
                if fallback:
                    row["required_checks"]["dense_fallback_target"] = row["final_error"]["passed"]
            direct = encode_direct(reference, sender.direct_context, precision)
            direct_decoded = decode_packet(direct.packet, sender.direct_context)
            row["same_precision_direct"] = {
                "packet": retainer.raw(direct.packet),
                "decoded": retainer.array(direct_decoded.final),
                "error": error_metrics(direct_decoded.final, reference),
                "raw_roundtrip_exact": raw_wire_exact(direct.packet, direct_decoded.final),
            }
            row["required_checks"]["direct_control_exact_at_wire_dtype"] = (
                row["same_precision_direct"]["raw_roundtrip_exact"]
            )
        if fixture["suite"] == "primary" and precision == "float64" and arm in (
            "coefficient", "gram_guard"
        ):
            row["required_checks"]["native_primary_route"] = not fallback
        row["status"] = "checked"
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)
        row["required_checks"]["completed_diagnostic"] = False
    row["required_passed"] = bool(row["status"] == "checked" and all(row["required_checks"].values()))
    return row, encoded, decoded


def practical_pair_gate(coefficient, direct):
    """All-row empirical goal, additional to the stable-margin theorem checks."""
    if coefficient is None or direct is None:
        return {"passed": False, "reason": "missing f64 diagnostic output"}
    c_row, c_decoded = coefficient
    d_row, d_decoded = direct
    if c_decoded is None or d_decoded is None:
        return {"passed": False, "reason": "missing decoded output"}
    error = error_metrics(c_decoded.final, d_decoded.final)
    c_ids, d_ids = c_row.get("actual_internal_snap_ids"), d_row.get("actual_internal_snap_ids")
    snap_equal = c_ids is not None and d_ids is not None and c_ids == d_ids
    return {"passed": bool(error["passed"] and snap_equal),
            "all_row_final_error": error, "all_internal_snaps_equal": bool(snap_equal),
            "coefficient_snap_ids": c_ids, "dual_direct_snap_ids": d_ids}


def postcheck_trial(record, artifacts, fixture, refs, primary_dual):
    """External numerical checking after the real measured ACK has completed."""
    decoded = artifacts.get("decoded")
    checks, details = {}, {}
    if decoded is None:
        return {"decoded_output_present": False}, {"error": "no retained receiver output"}
    packet = artifacts.get("payloads", {}).get("message_sent")
    if packet is None:
        return {"sent_packet_present": False}, {"error": "no retained sender packet"}
    exact = raw_wire_exact(packet, decoded.final)
    if exact is not None:
        checks["raw_roundtrip_exact_at_wire_dtype"] = exact
    arm = record["arm"]
    fallback = bool((record.get("sender_metadata") or {}).get("fallback", False))
    details["fallback"] = fallback
    if arm == "token_lookup":
        checks["token_lookup_exact"] = (
            canonical_output_bytes(decoded.final) == canonical_output_bytes(fixture["R"])
        )
        return checks, details
    name = reference_name(arm, fallback)
    if name not in refs:
        return {"declared_reference_present": False}, {"reference_method": name}
    ref = refs[name]
    mask = stable_rows(ref["processed"])
    details["reference_method"] = name
    details["stable_rows"] = mask.tolist()
    details["all_row_final_error"] = error_metrics(decoded.final, ref["processed"]["final"])
    if decoded.pre is not None:
        details["pre_error"] = error_metrics(decoded.pre, ref["pre"])
        checks["pre_reference"] = details["pre_error"]["passed"]
    if np.any(mask):
        details["stable_final_error"] = error_metrics(
            decoded.final[mask], ref["processed"]["final"][mask]
        )
        checks["stable_final_reference"] = details["stable_final_error"]["passed"]
        ids = actual_snap_ids(record.get("sender_metadata") or {}, decoded.metadata)
        if ids is not None:
            checks["stable_internal_snaps"] = bool(np.array_equal(
                ids[mask], ref["processed"]["ids"][mask]
            ))
    if fallback:
        checks["dense_fallback_target"] = details["all_row_final_error"]["passed"]
    if arm in ("coefficient", "gram_guard"):
        checks["native_primary_route"] = not fallback
    if arm == "coefficient":
        if primary_dual is None:
            checks["all_row_practical_comparison_available"] = False
        else:
            details["all_row_error_to_actual_dual_direct"] = error_metrics(
                decoded.final, primary_dual["final"]
            )
            checks["all_row_practical_final"] = details["all_row_error_to_actual_dual_direct"]["passed"]
            ids = actual_snap_ids(record.get("sender_metadata") or {}, decoded.metadata)
            checks["all_row_practical_snaps"] = bool(
                ids is not None and primary_dual["snap_ids"] is not None and
                np.array_equal(ids, primary_dual["snap_ids"])
            )
    return checks, details


def geometric(values):
    return None if not values else float(math.exp(sum(math.log(x) for x in values) / len(values)))


def summarize_trials(plan, trials):
    """Use paired raw wall times, never ratios of separately aggregated times."""
    indexed = {row["trial_id"]: row for row in trials}
    cases = []
    groups = {}
    for planned in plan:
        key = (planned["mode"], planned["fixture_id"], planned["profile"])
        groups.setdefault(key, []).append(planned)
    for (mode, fid, profile), rows in sorted(groups.items()):
        values = {arm: {} for arm in ARMS}
        for item in rows:
            row = indexed.get(item["trial_id"])
            if (row and row.get("status") == "completed"
                    and isinstance(row.get("wall_ns"), int) and row["wall_ns"] > 0):
                values[item["arm"]][item["repetition"]] = row
        case = {"mode": mode, "fixture_id": fid, "profile": profile, "arms": {}, "paired": {}}
        for arm in ARMS:
            measured = values[arm]
            walls = [measured[j]["wall_ns"] for j in sorted(measured)]
            case["arms"][arm] = {
                "completed_repetitions": len(walls),
                "wall_ns": walls,
                "median_wall_ns": statistics.median(walls) if walls else None,
                "logical_wire_bytes": [
                    measured[j].get("logical_wire_bytes") for j in sorted(measured)
                ],
            }
        for comparator in ALL_COMPARATORS:
            valid = all(j in values["coefficient"] and j in values[comparator]
                        for j in range(REPETITIONS))
            ratios = ([] if not valid else [
                values["coefficient"][j]["wall_ns"] / values[comparator][j]["wall_ns"]
                for j in range(REPETITIONS)
            ])
            case["paired"][comparator] = {
                "complete": valid, "coefficient_over_comparator": ratios,
                "median_ratio": statistics.median(ratios) if valid else None,
            }
        eligible_envelope = all(
            len(values[arm]) == REPETITIONS for arm in COMPARATORS + ("coefficient",)
        )
        if eligible_envelope:
            chosen = min(COMPARATORS, key=lambda arm: case["arms"][arm]["median_wall_ns"])
            case["hindsight_reference"] = {
                "chosen_arm": chosen,
                **case["paired"][chosen],
                "selection": "smallest four-run median among fixed comparators; hindsight only",
            }
        else:
            case["hindsight_reference"] = {
                "chosen_arm": None, "complete": False,
                "coefficient_over_comparator": [], "median_ratio": None,
            }
        cases.append(case)
    aggregates = {}
    for name, predicate, expected in (
        ("shaped_warm", lambda x: x["mode"] == "warm" and x["profile"] != "loopback", 18),
        ("loopback_warm", lambda x: x["mode"] == "warm" and x["profile"] == "loopback", 9),
        ("cold_constrained", lambda x: x["mode"] == "cold", 3),
    ):
        selected = [x for x in cases if predicate(x)]
        group = {"expected_cases": expected, "observed_cases": len(selected), "comparisons": {}}
        for comparison in ALL_COMPARATORS + ("hindsight_reference",):
            data = [x["hindsight_reference"] if comparison == "hindsight_reference"
                    else x["paired"][comparison] for x in selected]
            available = [x["median_ratio"] for x in data if x["complete"]]
            complete = len(selected) == expected and len(available) == expected
            group["comparisons"][comparison] = {
                "complete": complete, "completed_cases": len(available),
                "geometric_mean_case_median": geometric(available) if complete else None,
                "available_only_descriptive_geometric_mean": geometric(available),
                "case_median_min": min(available) if available else None,
                "case_median_max": max(available) if available else None,
            }
        aggregates[name] = group
    warm = [x["hindsight_reference"]["median_ratio"] for x in cases
            if x["mode"] == "warm" and x["hindsight_reference"]["complete"]]
    return {"cases": cases, "aggregates": aggregates,
            "all_warm_hindsight_complete": len(warm) == 27,
            "worst_warm_case_paired_median_ratio": max(warm) if len(warm) == 27 else None,
            "ratio_direction": "coefficient wall time divided by comparator wall time; smaller is faster"}


def runtime_and_source(directory):
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    cpu_model_names, cpu_info_error = [], None
    try:
        values = set()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in ("model name", "Hardware") and value.strip():
                values.add(value.strip())
        cpu_model_names = sorted(values)
    except OSError as exc:
        cpu_info_error = type(exc).__name__ + ": " + str(exc)
    affinity, affinity_error = None, None
    try:
        if hasattr(os, "sched_getaffinity"):
            affinity = sorted(os.sched_getaffinity(0))
    except OSError as exc:
        affinity_error = type(exc).__name__ + ": " + str(exc)
    actual = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "processor": platform.processor(), "cpu_model_names": cpu_model_names,
        "cpu_info_error": cpu_info_error, "cpu_affinity": affinity,
        "cpu_affinity_error": affinity_error,
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "blas_configuration": configuration.getvalue(),
        "perf_counter": vars(time.get_clock_info("perf_counter")),
        "github_commit": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "expected_python": "3.11.18", "expected_numpy": "2.4.6",
        "expected_threads": 1,
    }
    root = Path(__file__).resolve().parent.parent
    paths = (
        "latent_bridge/coefficient_packet.py", "latent_bridge/coefficient_transport.py",
        "latent_bridge/coefficient_study.py", "latent_bridge/alignment_core.py",
        "latent_bridge/alignment_packet.py", "docs/coefficient_packet_design.md",
    )
    manifest = {"design_commit": DESIGN_COMMIT, "expected_design_sha256": DESIGN_SHA256,
                "github_commit": actual["github_commit"], "files": {}}
    for relative in paths:
        raw = (root / relative).read_bytes()
        manifest["files"][relative] = {"sha256": sha(raw), "bytes": len(raw)}
    design_match = manifest["files"]["docs/coefficient_packet_design.md"]["sha256"] == DESIGN_SHA256
    manifest["design_matches"] = design_match
    atomic_json(directory / "runtime_manifest.json", actual)
    atomic_json(directory / "source_manifest.json", manifest)
    discrepancies = []
    if actual["python"] != "3.11.18" or actual["numpy"] != "2.4.6":
        discrepancies.append("pinned Python/NumPy runtime mismatch")
    if any(value != "1" for value in actual["thread_environment"].values()):
        discrepancies.append("all three BLAS/OpenMP environment limits must be 1")
    if not design_match:
        discrepancies.append("frozen design hash mismatch")
    if discrepancies:
        raise RuntimeError("; ".join(discrepancies))
    return actual, manifest


def report_text(result):
    analysis = result["analysis"]
    lines = [
        "# Coefficient packet v2: original execution report", "",
        "Study status: **" + result["status"] + "**.",
        "Complete frozen engineering success: **" + str(result["engineering_success"]) + "**.", "",
        "This is a fixed synthetic CPU/loopback experiment, with imposed application pacing.",
        "It measures no pretrained model, answer quality, WAN RTT or universal communication protocol.", "",
        "## Coverage and numerical gates", "",
        f"- Planned transport slots: {EXPECTED_TRIALS}; completed: {result['trial_status_counts'].get('completed', 0)}.",
        f"- Diagnostic rows: {result['diagnostic_rows']}/300; failed required diagnostic checks: {result['failed_diagnostic_checks']}.",
        f"- Eligible primary f64 routes: {result['eligible_primary_routes']}/54.",
        f"- Practical all-row coefficient/dual gates passed: {result['practical_gates_passed']}/9.",
        "- Float32 output errors are descriptive; direct wire bytes and declared fallback targets remain checked.",
        "- Every omitted/error/ineligible slot stays in the fixed 720-slot denominator.", "",
        "## Paired actual completion time", "",
        "Ratios are coefficient wall time divided by comparator wall time; lower is faster.",
        "Each cell is an equal-case geometric mean of four-pair median ratios.",
        "An incomplete fixed population is not replaced by an available-only success claim.", "",
        "| Population | Dual direct | Thin direct | Optimized Gram | Original Gram | Hindsight envelope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, label in (("shaped_warm", "Shaped warm (18 cases)"),
                              ("loopback_warm", "Loopback warm (9 cases)"),
                              ("cold_constrained", "Cold constrained (3 cases)")):
        group = analysis["aggregates"][group_name]["comparisons"]
        values = []
        for name in ALL_COMPARATORS + ("hindsight_reference",):
            value = group[name]["geometric_mean_case_median"]
            values.append("incomplete" if value is None else f"{value:.6f}")
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines.extend([
        "", "The hindsight envelope picks the fastest comparator median separately per case.",
        "It is not a deployable policy. Token lookup is a non-equivalent byte/time reference and is excluded.",
        "Cold trials include actual table/context transmission and setup receipts; warm gains do not imply cold gains.", "",
        "## Criteria", "",
    ])
    for name, passed in result["criteria"].items():
        lines.append("- " + name + ": " + str(passed))
    lines.extend([
        "", "## Retention and interpretation", "",
        "plan.json fixes all trial IDs/order before any diagnostics. trials.jsonl retains raw wall times,",
        "phase times, waits, frames, endpoint byte/hash reconciliation and every failed/ineligible slot.",
        "objects/ retains exact content-addressed packets, wire streams and receiver arrays;",
        "object_manifest.jsonl maps their actual bytes. fixtures.npz and fixture_manifest.json retain all inputs.",
        "diagnostics.jsonl records both precisions, native/fallback routes and full-vector errors.",
        "No expected prefix, oracle decision, precomputed coefficient or prepared packet enters a measured encoder.",
        "No canonical rank-deficient codec is tested here; unsupported paths transmit the declared dense prefix.",
        "A successful numerical guard is not a universal final-vector or vocabulary-snap guarantee.",
        "The comparison is against both efficient direct routes and an independently cheaper Gram guard.",
        "This report is the original worker output; any later independent audit is a separate retained document.", "",
        f"Frozen design: {DESIGN_COMMIT}; SHA256 {DESIGN_SHA256}.", "",
    ])
    return "\n".join(lines)


def fixed_outcomes(plan, trials, diagnostic_rows, eligibility, practical, warmups, fatal):
    analysis = summarize_trials(plan, trials)
    status_counts = dict(Counter(row["status"] for row in trials))
    diagnostic_failed = sum(
        sum(not bool(value) for value in row.get("required_checks", {}).values())
        for row in diagnostic_rows
    )
    routes = sum(bool(value) for mapping in eligibility.values() for value in mapping.values())
    practical_count = sum(bool(value.get("passed")) for value in practical.values())
    complete_transport = (
        len(trials) == EXPECTED_TRIALS and status_counts.get("completed", 0) == EXPECTED_TRIALS
        and len({row["trial_id"] for row in trials}) == EXPECTED_TRIALS
    )
    coverage = (
        complete_transport and len(diagnostic_rows) == 300 and diagnostic_failed == 0
        and routes == 54 and len(warmups) == 18
        and all(row.get("status") == "completed" for row in warmups) and not fatal
    )
    shaped = analysis["aggregates"]["shaped_warm"]["comparisons"]
    shaped_passed = all(
        shaped[name]["complete"] and shaped[name]["geometric_mean_case_median"] <= 0.97
        for name in COMPARATORS + ("hindsight_reference",)
    )
    loopback = analysis["aggregates"]["loopback_warm"]["comparisons"]["hindsight_reference"]
    worst = analysis["worst_warm_case_paired_median_ratio"]
    loopback_passed = bool(
        loopback["complete"] and loopback["geometric_mean_case_median"] <= 1.10
        and analysis["all_warm_hindsight_complete"] and worst <= 1.25
    )
    criteria = {
        "complete_correct_720_slots_and_required_diagnostics": coverage,
        "shaped_warm_against_all_three_and_hindsight_at_most_0.97": shaped_passed,
        "loopback_at_most_1.10_and_worst_warm_case_at_most_1.25": loopback_passed,
        "all_nine_primary_all_row_fidelity_gates": practical_count == 9 and len(practical) == 9,
    }
    maxima, fallback_counts = {}, Counter()
    for row in diagnostic_rows:
        if row.get("status") != "checked":
            continue
        key = "/".join((row["suite"], row["arm"], row["precision"],
                        "fallback" if row.get("fallback") else "native"))
        fallback_counts[key] += 1
        error = row.get("final_error", {}).get("relative_frobenius")
        if isinstance(error, (int, float)) and (key not in maxima or error > maxima[key]["relative_error"]):
            maxima[key] = {"relative_error": error, "fixture_id": row["fixture_id"],
                           "internal_snap_changes": row.get("internal_snap_changes")}
    return {
        "study": "coefficient_packet_v2", "status": "incomplete" if fatal else "complete",
        "engineering_success": bool(all(criteria.values())), "criteria": criteria,
        "design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256,
        "trial_status_counts": status_counts, "planned_trials": EXPECTED_TRIALS,
        "diagnostic_rows": len(diagnostic_rows), "failed_diagnostic_checks": diagnostic_failed,
        "eligible_primary_routes": routes, "practical_gates_passed": practical_count,
        "warmup_count": len(warmups), "fatal_error": fatal,
        "analysis": analysis, "diagnostic_route_counts": dict(fallback_counts),
        "descriptive_final_error_maxima": maxima,
        "limits": [
            "Synthetic CPU matrices, fixed loopback TCP and imposed application service conditions only.",
            "No model inference, quality assessment, production-network or cross-host bitwise claim.",
            "Float32 fidelity is descriptive and cannot establish the primary f64 success claim.",
            "Hindsight comparator selection is not a deployable policy.",
            "Cold setup is a real first-message cost; it is not silently amortized into warm timing.",
            "All ratios use paired original wall times; no timing values are synthesized or retried.",
        ],
    }


def run(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = [p.name for p in directory.iterdir() if p.name not in WRAPPER_FILES]
    if unexpected:
        raise RuntimeError("refusing to overwrite prior study output: " + repr(sorted(unexpected)))
    begin = time.perf_counter_ns()
    plan = build_plan()
    atomic_json(directory / "plan.json", {
        "schema_version": 1, "expected_count": EXPECTED_TRIALS,
        "planned_trials": plan, "block_shuffle": "one PCG64 seed23029 permutation of all 120 blocks",
    })
    state = {"stage": "plan_saved", "finished_trials": 0, "active_trial_id": None}
    atomic_json(directory / "status.json", state)
    fixtures, references, primary_dual = {}, {}, {}
    eligibility, practical = {}, {}
    diagnostics, trials, warmups = [], [], []
    fatal = None
    retainer = None
    active = None
    active_record = None
    streams = {}
    phases = {}
    try:
        runtime_and_source(directory)
        retainer = Retainer(directory)
        for name in ("diagnostics", "trials", "warmups", "references"):
            streams[name] = (directory / (name + ".jsonl")).open("w", encoding="utf-8")
        t = time.perf_counter_ns()
        construction_records, archive = {}, {}
        specifications = [(k, d, seed, None, False) for k, d in SHAPES for seed in SEEDS]
        specifications += [(6, 24, seed, g, False) for seed in STRESS_SEEDS for g in range(1, 9)]
        specifications += [(k, d, 23041, None, True) for k, d in SHAPES]
        for k, d, seed, geometry, is_warmup in specifications:
            fid = fixture_id(k, d, seed, geometry, is_warmup)
            try:
                fixture = make_fixture(k, d, seed, geometry, warmup=is_warmup)
                fixtures[fid] = fixture
                details = {"status": "constructed", "K": k, "d": d, "seed": seed,
                           "suite": fixture["suite"], "geometry": geometry,
                           "construction": fixture["construction"], "arrays": {}}
                for name in ("S", "R", "vocab", "ids"):
                    key = fid + "__" + name
                    array = np.ascontiguousarray(fixture[name])
                    archive[key] = array
                    details["arrays"][name] = {
                        "npz_key": key, "shape": list(array.shape), "dtype": array.dtype.str,
                        "data_sha256": sha(array.tobytes()), "data_bytes": int(array.nbytes),
                    }
                construction_records[fid] = details
            except Exception as exc:
                construction_records[fid] = {"status": "error", "error": type(exc).__name__ + ": " + str(exc)}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **archive)
        raw = buffer.getvalue()
        atomic_bytes(directory / "fixtures.npz", raw)
        atomic_json(directory / "fixture_manifest.json", {
            "archive": {"sha256": sha(raw), "bytes": len(raw)}, "fixtures": construction_records,
        })
        del archive, buffer, raw
        phases["fixture_construction_and_retention_ns"] = time.perf_counter_ns() - t
        state["stage"] = "diagnostics"
        atomic_json(directory / "status.json", state)
        t = time.perf_counter_ns()
        for k, d, seed, geometry, _ in specifications[:25]:
            fid = fixture_id(k, d, seed, geometry)
            fixture = fixtures.get(fid)
            if fixture is None:
                for arm in ARMS:
                    for precision in ("float64", "float32"):
                        row = {"fixture_id": fid, "suite": "primary" if geometry is None else "stress",
                               "arm": arm, "precision": precision, "status": "error",
                               "error": "fixture construction failed; no resampling",
                               "required_checks": {"fixture_available": False}, "required_passed": False}
                        diagnostics.append(row)
                        emit(streams["diagnostics"], row)
                if geometry is None:
                    eligibility[fid] = {arm: False for arm in ARMS}
                    practical[fid] = {"passed": False, "reason": "fixture construction failed"}
                continue
            refs, reference_record = references_for(fixture, retainer)
            references[fid] = refs
            emit(streams["references"], {"fixture_id": fid, "references": reference_record})
            f64 = {}
            for arm in ARMS:
                for precision in ("float64", "float32"):
                    row, encoded, decoded = diagnose_one(fixture, arm, precision, refs, retainer)
                    diagnostics.append(row)
                    emit(streams["diagnostics"], row)
                    if precision == "float64":
                        f64[arm] = (row, decoded)
            if geometry is None:
                practical[fid] = practical_pair_gate(f64.get("coefficient"), f64.get("dual_direct"))
                eligibility[fid] = {arm: f64[arm][0]["required_passed"] for arm in ARMS}
                eligibility[fid]["coefficient"] = bool(
                    eligibility[fid]["coefficient"] and practical[fid]["passed"]
                )
                dual_row, dual_decoded = f64["dual_direct"]
                primary_dual[fid] = None if dual_decoded is None else {
                    "final": dual_decoded.final,
                    "snap_ids": dual_row.get("actual_internal_snap_ids"),
                }
                atomic_json(directory / "eligibility.json", {"routes": eligibility, "practical": practical})
            print(json.dumps({"stage": "diagnostic_complete", "fixture_id": fid,
                              "rows": len(diagnostics)}), flush=True)
        atomic_json(directory / "eligibility.json", {"routes": eligibility, "practical": practical})
        phases["reference_and_diagnostic_ns"] = time.perf_counter_ns() - t
        state["stage"] = "functional_warmups"
        atomic_json(directory / "status.json", state)
        t = time.perf_counter_ns()
        stop = False
        for k, d in SHAPES:
            fid = fixture_id(k, d, 23041, warmup=True)
            fixture = fixtures.get(fid)
            for arm in ARMS:
                if fixture is None:
                    record = {"status": "error", "error": "warmup fixture construction failed"}
                else:
                    record, artifacts = run_trial(
                        fixture["S"], fixture["ids"], fixture["vocab"], arm, PROFILES[0]
                    )
                    record["retained"] = retainer.artifacts(artifacts)
                record.update({"fixture_id": fid, "arm": arm, "included_in_timing_analysis": False,
                               "functional_warmup": True})
                record["wall_ns"] = None
                warmups.append(record)
                emit(streams["warmups"], record)
                if record.get("fatal_receiver_alive"):
                    fatal = "receiver thread survived cleanup during a functional warmup"
                    stop = True
                    break
            if stop:
                break
        phases["functional_warmup_ns"] = time.perf_counter_ns() - t
        state["stage"] = "transport"
        atomic_json(directory / "status.json", state)
        profiles = {x.name: x for x in PROFILES}
        t = time.perf_counter_ns()
        for planned in plan:
            if fatal:
                break
            active = planned
            active_record = None
            state["active_trial_id"] = planned["trial_id"]
            atomic_json(directory / "status.json", state)
            fid, arm = planned["fixture_id"], planned["arm"]
            if not eligibility.get(fid, {}).get(arm, False):
                record = {**planned, "status": "numerically_ineligible", "wall_ns": None,
                          "reason": "fixed pre-transport f64 eligibility/practical gate failed"}
            else:
                fixture = fixtures[fid]
                record, artifacts = run_trial(
                    fixture["S"], fixture["ids"], fixture["vocab"], arm,
                    profiles[planned["profile"]], cold=planned["mode"] == "cold"
                )
                active_record = record
                record.update(planned)
                record["transport_status"] = record["status"]
                record["retained"] = retainer.artifacts(artifacts)
                try:
                    checks, details = postcheck_trial(record, artifacts, fixture,
                                                     references[fid], primary_dual.get(fid))
                    record["numerical_checks"] = checks
                    record["numerical_details"] = details
                    if not checks or not all(checks.values()):
                        record["status"] = "error"
                        record["post_timing_numerical_failure"] = True
                except Exception as exc:
                    record["status"] = "error"
                    record["post_timing_numerical_error"] = type(exc).__name__ + ": " + str(exc)
                if record.get("fatal_receiver_alive"):
                    fatal = "receiver thread survived cleanup; no subsequent timed trial started"
            trials.append(record)
            emit(streams["trials"], record)
            active = active_record = None
            state["finished_trials"] = len(trials)
            state["active_trial_id"] = None
            if planned["order_position"] == len(ARMS) - 1:
                print(json.dumps({"stage": "transport_block_complete",
                                  "block_id": planned["block_id"], "finished_trials": len(trials)}),
                      flush=True)
        phases["transport_and_post_timing_retention_ns"] = time.perf_counter_ns() - t
    except BaseException as exc:
        fatal = type(exc).__name__ + ": " + str(exc)
        atomic_json(directory / "worker_failure.json", {
            "error": fatal, "stage": state["stage"], "active_trial_id": state.get("active_trial_id"),
            "traceback": traceback.format_exc(),
        })
    finally:
        if "trials" not in streams:
            streams["trials"] = (directory / "trials.jsonl").open("a", encoding="utf-8")
        completed_ids = {row["trial_id"] for row in trials}
        for planned in plan:
            if planned["trial_id"] in completed_ids:
                continue
            if active is not None and planned["trial_id"] == active["trial_id"] and active_record is not None:
                row = {**active_record, **planned, "status": "error",
                       "finalization_error": fatal or "trial output was not finalized"}
            else:
                row = {**planned, "status": "not_executed", "wall_ns": None,
                       "reason": fatal or "study stopped before this planned slot"}
            trials.append(row)
            emit(streams["trials"], row)
        for stream in streams.values():
            stream.close()
        if retainer is not None:
            retainer.close()
    result = fixed_outcomes(plan, trials, diagnostics, eligibility, practical, warmups, fatal)
    result["phase_ns"] = phases
    result["worker_elapsed_ns_before_final_report"] = time.perf_counter_ns() - begin
    atomic_json(directory / "results.json", result)
    atomic_bytes(directory / "report.md", report_text(result).encode("utf-8"))
    state.update({"stage": "finalized", "status": result["status"], "finished_trials": len(trials),
                  "active_trial_id": None, "engineering_success": result["engineering_success"],
                  "fatal_error": fatal})
    atomic_json(directory / "status.json", state)
    print(json.dumps({"stage": "finalized", "engineering_success": result["engineering_success"],
                      "status": result["status"], "trial_status_counts": result["trial_status_counts"]}),
          flush=True)
    return 0 if result["criteria"]["complete_correct_720_slots_and_required_diagnostics"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
