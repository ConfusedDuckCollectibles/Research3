"""Execute the fixed, NumPy-only alignment study; no model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from .alignment_core import (ALGORITHMS, Unsupported, ATOL, RTOL, helmert,
                             close_error, spectrum_info, postprocess,
                             final_nearest_ids)
from .alignment_packet import (Session, encode_message, decode_packet,
                               encode_direct, encode_token_only)

DESIGN_COMMIT = "e7c467f2b097666dc474a3dbb6e273bf253e785f"
DESIGN_SHA256 = "c785e3fef3af9c8fac3bb7a04ccd7980ddfa51c121f3f11a9068eda3b7a70a5e"
SEEDS = (101, 202, 303)
ARM_ORDER = ("dense", "thin", "dual", "canonical")
REPETITIONS = 8


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "positive_infinity" if value > 0 else "negative_infinity" if value < 0 else "nan"
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def json_bytes(value):
    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, raw):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def array_hash(array):
    array = np.ascontiguousarray(array)
    descriptor = json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                            sort_keys=True, separators=(",", ":")).encode()
    return sha(descriptor + b"\0" + array.tobytes())


class Arrays:
    def __init__(self, output):
        self.output, self.values, self.cases = output, {}, {}

    def add(self, case_id, name, array):
        if array is None:
            return
        array = np.ascontiguousarray(array)
        key = "a_" + array_hash(array)
        self.values.setdefault(key, array.copy())
        self.cases.setdefault(case_id, {})[name] = key

    def save(self, phase):
        path = self.output / "arrays.npz"
        temporary = path.with_name("arrays.npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **self.values)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        inventory = {key: {"shape": list(value.shape), "dtype": value.dtype.str,
                           "bytes": int(value.nbytes), "sha256": array_hash(value)}
                     for key, value in self.values.items()}
        manifest = {"phase": phase, "arrays": inventory, "cases": self.cases,
                    "npz": {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}}
        atomic_write(self.output / "arrays_manifest.json", json_bytes(manifest))
        return {"arrays": len(inventory), "case_count": len(self.cases),
                "npz_bytes": path.stat().st_size, "phase": phase}


def rng(seed, suite, k, d, case_id):
    return np.random.default_rng(np.random.SeedSequence([seed, suite, k, d, case_id]))


def orthogonal(rows, columns, generator):
    q, r = np.linalg.qr(generator.normal(size=(rows, columns)), mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs[None, :]


def points(k, d, rank, ratio, generator, sample=None):
    mean = 0.2 * generator.normal(size=d)
    if rank == 0:
        return np.broadcast_to(mean, (k, d)).copy()
    if sample is None:
        sample = orthogonal(k - 1, k - 1, generator)[:, :rank]
    features = orthogonal(d, rank, generator)
    singular = np.sqrt(k) * np.geomspace(1.0, ratio, rank)
    return (helmert(k) @ (sample * singular[None, :])) @ features.T + mean


def case(case_id, suite, s, r, vocab, ids, sr, rr, qr, seed,
         lam=1e-3, eps=1e-6, required=True, packets=False, tags=None):
    return {"id": case_id, "suite": suite, "s": s, "r": r, "vocab": vocab,
            "ids": ids, "source_rank": sr, "receiver_rank": rr, "cross_rank": qr,
            "seed": seed, "lambda": lam, "epsilon": eps, "required": required,
            "packets": packets, "tags": tags or {}}


def standard_vocab(r, generator):
    return np.vstack([r, generator.normal(size=(64, r.shape[1]))])


def make_rank_pair(k, d, geometry, generator):
    ranks = [(0, k - 1), (k - 1, 0), (0, 0), (3, k - 1),
             (k - 1, 3), (3, 2), (3, 2), (k - 1, 3), (k - 1, 2)]
    sr, rr = ranks[geometry]
    shared = orthogonal(k - 1, k - 1, generator)
    sa = shared[:, :sr] if geometry in {5, 6} else None
    rb = shared[:, :rr] if geometry == 5 else shared[:, 3:5] if geometry == 6 else None
    s = points(k, d, sr, 0.1, generator, sa)
    if geometry == 7:
        unique = points(4, d, 3, 0.1, generator)
        ids = np.repeat(np.arange(4, dtype=np.uint32), 2)
        r = unique[ids]
        vocab = standard_vocab(unique, generator)
    else:
        r = points(k, d, rr, 0.1, generator, rb)
        ids = np.arange(k, dtype=np.uint32)
        vocab = standard_vocab(r, generator)
    q = 0 if geometry == 6 else min(sr, rr)
    return s, r, vocab, ids, sr, rr, q


def algebraic_cases():
    # A: 4 K * 3 seeds * 2 spectra * 5 lambdas = 120.
    for k in (2, 4, 8, 16):
        for seed in SEEDS:
            for index, ratios in enumerate(((1.0, 1.0), (0.01, 0.03))):
                gen = rng(seed, 1, k, 64, index)
                s = points(k, 64, k - 1, ratios[0], gen)
                r = points(k, 64, k - 1, ratios[1], gen)
                vocab = standard_vocab(r, gen)
                for lam in (0.0, 1e-8, 1e-6, 1e-3, 0.1):
                    yield case(f"A-k{k}-seed{seed}-spectrum{index}-lambda{lam}", "A",
                               s, r, vocab, np.arange(k, dtype=np.uint32),
                               k - 1, k - 1, k - 1, seed, lam, packets=True)
    # B: nine exact-rank geometries and three lambdas = 27.
    for geometry in range(9):
        values = make_rank_pair(8, 32, geometry, rng(101, 2, 8, 32, geometry))
        s, r, vocab, ids, sr, rr, q = values
        for lam in (1e-8, 1e-3, 0.1):
            yield case(f"B-geometry{geometry}-lambda{lam}", "B", s, r, vocab, ids,
                       sr, rr, q, 101, lam, packets=True,
                       tags={"geometry": geometry})
    # C: construction rank remains full even when the numerical guard truncates.
    for side in range(2):
        for index, ratio in enumerate((1e-4, 1e-5, 1e-6, 1e-8)):
            gen = rng(101, 3, 8, 32, side * 4 + index)
            s = points(8, 32, 7, ratio if side == 0 else 0.1, gen)
            r = points(8, 32, 7, ratio if side == 1 else 0.1, gen)
            yield case(f"C-side{side}-ratio{ratio}", "C", s, r, standard_vocab(r, gen),
                       np.arange(8, dtype=np.uint32), 7, 7, 7, 101, required=False, packets=True,
                       tags={"known_nonzero_smallest_ratio": ratio, "changed_side": side})
    # D1: baseline, translation, and right-isometry, with fixed R and IDs.
    for seed in SEEDS:
        gen = rng(seed, 4, 8, 32, 0)
        s, r = points(8, 32, 7, 0.1, gen), points(8, 32, 7, 0.1, gen)
        vocab = standard_vocab(r, gen)
        variants = (("baseline", s), ("translation", s + gen.normal(size=32)),
                    ("rotation", s @ orthogonal(32, 32, gen)))
        for name, transformed in variants:
            yield case(f"D-isometry-{seed}-{name}", "D", transformed, r, vocab,
                       np.arange(8, dtype=np.uint32), 7, 7, 7, seed,
                       tags={"control": "isometry", "group": str(seed), "variant": name})
    # D2: analytical sign bit, decoded tokens unchanged.
    r = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    for lam in (0.0, 1e-8, 1e-3, 0.1):
        for bit in (-1, 1):
            s = np.array([[1.0, bit, 0], [-1.0, -bit, 0]])
            yield case(f"D-sign-lambda{lam}-bit{bit}", "D", s, r, r.copy(),
                       np.array([0, 1], dtype=np.uint32), 1, 1, 1, 0, lam,
                       tags={"control": "sign", "bit": bit})
    # D3: zero-mean collision and its nonzero-mean boundary.
    eye = np.eye(4)
    for nonzero_mean in (False, True):
        r = np.stack([eye[0], -eye[0]]) + (eye[1] if nonzero_mean else 0.0)
        vocab = np.vstack([r, eye[2], -eye[2]]) if nonzero_mean else np.stack(
            [eye[0], -eye[0], eye[1], -eye[1]])
        for scale in (0.25, 1.0, 4.0):
            s = scale * np.stack([eye[0] + eye[1], -eye[0] - eye[1]])
            yield case(f"D-scale-mean{int(nonzero_mean)}-scale{scale}", "D", s, r, vocab,
                       np.array([0, 1], dtype=np.uint32), 1, 1, 1, 0, 0.1,
                       tags={"control": "scale", "nonzero_mean": nonzero_mean, "scale": scale})
    # D4: same Gram, different dense null completions can occur.
    s0 = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [-1., -1., 0, 0]])
    r = np.tile(eye[0], (3, 1))
    vocab = np.vstack([r, eye[1], -eye[1], eye[2], -eye[2]])
    for name, s in (("baseline", s0), ("rotation", s0[:, [2, 3, 0, 1]])):
        yield case(f"D-null-{name}", "D", s, r, vocab, np.arange(3, dtype=np.uint32),
                   2, 0, 0, 0, 0.1, tags={"control": "null_completion", "variant": name})
    # D5: actual author bypasses and dimension guards.
    for bit in (-1, 1):
        s = np.array([[1.0, bit, 0, 0]])
        r = np.array([[1.0, 0, 0, 0]])
        yield case(f"D-one-row-{bit}", "D", s, r, np.vstack([r, -r]),
                   np.array([0], dtype=np.uint32), 0, 0, 0, 0, packets=True,
                   tags={"control": "early_return"})
    yield case("D-missing-ids", "D", s0, None, np.stack([eye[0], -eye[0]]),
               None, 2, None, None, 0, packets=True, tags={"control": "early_return"})
    for d in (4, 3):
        s = helmert(4) @ np.eye(3, d)
        reflection = np.eye(d)
        reflection[0, 0] = -1
        r = s @ reflection
        yield case(f"D-dimension-k4-d{d}", "D", s, r, np.vstack([r, np.eye(d)]),
                   np.arange(4, dtype=np.uint32), 3, 3, 3, 0, 0.1, packets=True,
                   tags={"control": "dimension_guard", "no_spare_axis": d == 3})
    # E: 4 geometries * (6 ideal + 8 floored) = 56.
    for geometry in range(4):
        gen = rng(202, 5, 8, 32, geometry)
        if geometry == 3:
            shared = orthogonal(7, 7, gen)
            s = points(8, 32, 1, 0.1, gen, shared[:, :1])
            r = points(8, 32, 1, 0.1, gen, shared[:, 1:2])
            sr, rr, q = 1, 1, 0
        else:
            sr, rr = 7, (7, 3, 0)[geometry]
            s, r = points(8, 32, sr, 0.1, gen), points(8, 32, rr, 0.1, gen)
            q = rr
        vocab = standard_vocab(r, gen)
        for eps in (0.0, 1e-6):
            lambdas = (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)
            if eps:
                lambdas += (1e-8, 0.0)
            for lam in lambdas:
                yield case(f"E-geometry{geometry}-eps{eps}-lambda{lam}", "E",
                           s, r, vocab, np.arange(8, dtype=np.uint32),
                           sr, rr, q, 202, lam, eps,
                           tags={"geometry": geometry, "limit_control": True})


class Study:
    def __init__(self, output):
        self.output = output
        self.arrays = Arrays(output)
        self.stream = (output / "cases.jsonl").open("x", encoding="utf-8")
        self.timing_stream = (output / "timings.jsonl").open("x", encoding="utf-8")
        self.counts = Counter()
        self.failures = []
        self.required_checks = 0
        self.failed_checks = 0
        self.support = Counter()
        self.max_errors = defaultdict(float)
        self.packet_summary = Counter()
        self.applicability = defaultdict(Counter)
        self.controls = {}
        self.timings = []
        self.phases = {}
        self.limit_rows = []
        self.algebraic_count = 0

    def check(self, row, name, passed, **details):
        row["checks"].append({"name": name, "passed": bool(passed), **details})

    def emit(self, row):
        required = bool(row.get("required", False))
        failed = [v["name"] for v in row.get("checks", []) if not v["passed"]]
        if required:
            self.required_checks += len(row.get("checks", []))
            self.failed_checks += len(failed)
            if failed:
                self.failures.append({"id": row["id"], "checks": failed})
        row["required_check_failures"] = failed if required else []
        row["status"] = "failed" if required and failed else (
            "verified" if required else "descriptive")
        self.stream.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        self.counts[row["suite"]] += 1

    def evaluate(self, c, retain=True, emit=True):
        started = time.perf_counter()
        s, r = c["s"], c["r"]
        k, d = s.shape
        in_scope = r is not None and 2 <= k < d
        full = in_scope and c["source_rank"] == k - 1 and c["receiver_rank"] == k - 1
        unique = in_scope and c["cross_rank"] == c["source_rank"]
        required = c["required"]
        row = {key: value for key, value in c.items()
               if key not in {"s", "r", "vocab", "ids", "packets"}}
        row.update({"K": k, "d": d, "checks": [], "methods": {}, "unique_product": bool(unique),
                    "input_hashes": {"S": array_hash(s), "R": array_hash(r) if r is not None else None,
                                     "vocabulary": array_hash(c["vocab"]),
                                     "token_ids": array_hash(c["ids"]) if c["ids"] is not None else None}})
        if retain:
            for name, value in (("S", s), ("R", r), ("vocabulary", c["vocab"]), ("token_ids", c["ids"])):
                self.arrays.add(c["id"], name, value)
        results, processed = {}, {}
        for method in ARM_ORDER:
            expected = method == "dense" or (in_scope if method == "canonical" else full)
            if r is None and method != "dense":
                row["methods"][method] = {"status": "not_applicable", "reason": "author_missing_token_bypass"}
                continue
            try:
                result = ALGORITHMS[method](s, r, c["lambda"], c["epsilon"], details=True)
                results[method] = result
                row["methods"][method] = {"status": "supported", "diagnostics": result.diagnostics,
                                          "bypass": result.bypass}
                if required:
                    self.check(row, method + "_expected_support", expected)
                if retain:
                    self.arrays.add(c["id"], "pre_" + method, result.pre)
                self.check(row, method + "_finite", np.isfinite(result.pre).all())
                if not result.bypass:
                    processed[method] = {str(alpha): postprocess(result.pre, c["vocab"], alpha)
                                         for alpha in (0.0, 0.3)}
                if method != "dense":
                    shapes = result.diagnostics.get("operation_shapes", [])
                    valid_shapes = all((max(v["shape"]) <= k if v["operation"] == "eigh"
                                        else min(v["shape"]) <= k) for v in shapes)
                    self.check(row, method + "_sample_space_decompositions", valid_shapes)
                self.support[(c["suite"], method, "supported")] += 1
            except Unsupported as exc:
                results.pop(method, None)
                processed.pop(method, None)
                row["methods"][method] = {"status": "unsupported", "reason": exc.reason,
                                          "diagnostics": exc.diagnostics}
                self.support[(c["suite"], method, "unsupported")] += 1
                if required:
                    self.check(row, method + "_required_coverage_or_expected_rejection", not expected,
                               expected_supported=expected)
            except Exception as exc:
                results.pop(method, None)
                processed.pop(method, None)
                row["methods"][method] = {"status": "error", "error_type": type(exc).__name__,
                                          "error": str(exc)}
                self.support[(c["suite"], method, "error")] += 1
                self.check(row, method + "_unexpected_exception", False, error=str(exc))
        dense = results.get("dense")
        if dense is None:
            self.check(row, "dense_reference_available", False)
        elif dense.bypass:
            self.check(row, "author_bypass_exact", np.array_equal(dense.pre, s))
        else:
            rotation = dense.diagnostics["rotation_orthogonality"]
            self.check(row, "dense_rotation_orthogonality", rotation["passed"], error=rotation)
            self.check(row, "dense_proper_rotation", dense.diagnostics["rotation_determinant_sign"] == 1)
            for method, result in results.items():
                metrics = {}
                if in_scope:
                    gram = close_error(result.white @ result.white.T, dense.sender_gram)
                    objective = close_error(np.array(np.sum(result.white * dense.target_white)),
                                            np.array(dense.diagnostics["cross_nuclear_norm"]))
                    target = close_error(result.target_white, dense.target_white)
                    metrics.update({"gram": gram, "objective": objective, "target_white": target})
                    for name, error in (("gram", gram), ("objective", objective), ("target_white", target)):
                        self.check(row, method + "_" + name, error["passed"], error=error)
                        self.max_errors[(c["suite"], method, name)] = max(self.max_errors[(c["suite"], method, name)], error["absolute"])
                pre_error = close_error(result.pre, dense.pre)
                metrics["pre_vs_dense"] = pre_error
                if unique:
                    self.check(row, method + "_unique_pre", pre_error["passed"], error=pre_error)
                    self.max_errors[(c["suite"], method, "unique_pre")] = max(
                        self.max_errors[(c["suite"], method, "unique_pre")], pre_error["absolute"])
                if method != "dense" and result.diagnostics.get("source"):
                    for key, expected_rank in (("source", c["source_rank"]),
                                               ("receiver", c["receiver_rank"]),
                                               ("cross", c["cross_rank"])):
                        observed = result.diagnostics[key]["rank"]
                        metrics[key + "_rank"] = observed
                        self.check(row, method + "_" + key + "_construction_rank",
                                   observed == expected_rank, observed=observed, expected=expected_rank)
                metrics["postprocessing"] = {}
                for alpha in (0.0, 0.3):
                    p, ref = processed[method][str(alpha)], processed["dense"][str(alpha)]
                    cal = close_error(p["calibrated"], ref["calibrated"])
                    stable = ((ref["margins"] > 1e-6) & (ref["pre_norms"] > 1e-6) &
                              (ref["query_norms"] > 1e-12) & (ref["vocab_norms"][ref["ids"]] > 1e-12))
                    final_error = close_error(p["final"], ref["final"])
                    data = {"calibrated_vs_dense": cal, "final_vs_dense_descriptive": final_error,
                            "snap_ids": p["ids"].tolist(), "snap_margins": p["margins"].tolist(),
                            "stable_rows": int(stable.sum()),
                            "all_snap_changes": int(np.count_nonzero(p["ids"] != ref["ids"]))}
                    if unique:
                        self.check(row, f"{method}_calibrated_alpha{alpha}", cal["passed"], error=cal)
                        if np.any(stable):
                            stable_error = close_error(p["final"][stable], ref["final"][stable])
                            self.check(row, f"{method}_stable_snap_alpha{alpha}",
                                       np.array_equal(p["ids"][stable], ref["ids"][stable]))
                            self.check(row, f"{method}_stable_final_alpha{alpha}",
                                       stable_error["passed"], error=stable_error)
                            data["stable_final_error"] = stable_error
                    metrics["postprocessing"][str(alpha)] = data
                row["methods"][method]["metrics"] = metrics
        self.structural_checks(c, row, results, processed)
        if c["packets"]:
            self.packet_checks(c, row, results, processed)
        row["elapsed_seconds"] = time.perf_counter() - started
        if emit:
            self.emit(row)
        if c["suite"] != "T":
            self.algebraic_count += 1
        return row, results, processed

    def structural_checks(self, c, row, results, processed):
        tag, cid = c["tags"], c["id"]
        control = tag.get("control")
        if control == "isometry":
            key = ("isometry", tag["group"])
            if tag["variant"] == "baseline":
                self.controls[key] = {m: v.pre.copy() for m, v in results.items()}
            else:
                baseline = self.controls.get(key, {})
                for m, v in results.items():
                    if m in baseline:
                        error = close_error(v.pre, baseline[m])
                        self.check(row, m + "_source_isometry_invariance", error["passed"], error=error)
        elif control == "sign":
            expected = np.sqrt(2 * (1 + c["lambda"]) / (2 + c["lambda"])) * c["r"]
            self.check(row, "sign_bit_readout_ids_unchanged",
                       np.array_equal(np.argmax(c["s"] @ c["vocab"].T, axis=1), c["ids"]))
            for m, v in results.items():
                error = close_error(v.pre, expected)
                self.check(row, m + "_closed_form_sign_example", error["passed"], error=error)
                final = close_error(processed[m]["0.3"]["final"], c["r"])
                self.check(row, m + "_sign_bit_collision_after_postprocess", final["passed"], error=final)
            key = ("sign", c["lambda"])
            gram = c["s"] @ c["s"].T
            if tag["bit"] == -1:
                self.controls[key] = (c["s"].copy(), gram)
            elif key in self.controls:
                prior, prior_gram = self.controls[key]
                self.check(row, "different_raw_sign_bit", not np.array_equal(prior, c["s"]))
                self.check(row, "identical_sign_gram", np.array_equal(prior_gram, gram))
        elif control == "scale":
            if not tag["nonzero_mean"]:
                expected = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
                for m in processed:
                    error = close_error(processed[m]["0.0"]["calibrated"], expected)
                    self.check(row, m + "_zero_mean_scale_collision", error["passed"], error=error)
            elif tag["scale"] == 0.25:
                self.controls["nonzero_mean_scale"] = {
                    m: p["0.0"]["calibrated"].copy() for m, p in processed.items()}
            elif tag["scale"] == 4.0:
                for m, p in processed.items():
                    prior = self.controls.get("nonzero_mean_scale", {}).get(m)
                    if prior is not None:
                        distance = float(np.linalg.norm(p["0.0"]["calibrated"] - prior))
                        self.check(row, m + "_nonzero_mean_counterexample", distance > 1e-4,
                                   extreme_scale_distance=distance)
        elif control == "null_completion":
            if tag["variant"] == "baseline":
                self.controls["null_completion"] = (
                    c["s"] @ c["s"].T, {m: v.pre.copy() for m, v in results.items()})
            elif "null_completion" in self.controls:
                gram, prefixes = self.controls["null_completion"]
                self.check(row, "null_completion_identical_gram",
                           np.array_equal(gram, c["s"] @ c["s"].T))
                row["null_completion_output_differences"] = {
                    m: close_error(v.pre, prefixes[m]) for m, v in results.items() if m in prefixes}
                if "canonical" in results and "canonical" in prefixes:
                    error = close_error(results["canonical"].pre, prefixes["canonical"])
                    self.check(row, "canonical_deterministic_same_gram", error["passed"], error=error)
        elif control == "dimension_guard" and tag["no_spare_axis"] and "dense" in results:
            distance = float(np.linalg.norm(results["dense"].pre - c["r"]))
            self.check(row, "proper_rotation_boundary_differs_from_reflection", distance > 1e-4,
                       distance_to_reflected_target=distance)
        if c["suite"] == "E" and c["r"] is not None:
            r, k = c["r"], len(c["r"])
            _, _, vt = np.linalg.svd(r - r.mean(axis=0), full_matrices=False)
            basis = vt[:c["receiver_rank"]]
            reference_cal = postprocess(r, c["vocab"], 0.0)["calibrated"]
            limit = {"id": cid, "geometry": tag["geometry"], "lambda": c["lambda"],
                     "epsilon": c["epsilon"], "methods": {}}
            for m, value in results.items():
                centered = value.pre - r.mean(axis=0)
                residual = centered - (centered @ basis.T) @ basis
                residual_norm = float(np.linalg.norm(residual))
                bound = np.sqrt(k * (k - 1) * c["lambda"])
                limit["methods"][m] = {
                    "pre_error_to_R": float(np.linalg.norm(value.pre - r)),
                    "calibrated_error_to_calibrated_R": float(np.linalg.norm(
                        processed[m]["0.0"]["calibrated"] - reference_cal)),
                    "outside_target_span_norm": residual_norm,
                    "ideal_bound": float(bound)}
                if c["epsilon"] == 0 and c["source_rank"] == k - 1:
                    self.check(row, m + "_ideal_unmatched_component_bound",
                               residual_norm <= bound + ATOL + RTOL * bound,
                               actual=residual_norm, bound=float(bound))
            row["limit"] = limit
            self.limit_rows.append(limit)

    def packet_checks(self, c, row, results, processed):
        session = Session(c["vocab"], c["lambda"], c["epsilon"], 0.3)
        row["session"] = {"sha256": session.digest.hex(), "cold_setup": session.cold_setup}
        row["packets"] = []
        dense = results.get("dense")
        dense_final = None if dense is None else (
            dense.pre if dense.bypass else processed["dense"]["0.3"]["final"])
        if c["ids"] is not None:
            token = encode_token_only(c["ids"], session)
            self.check(row, "token_packet_exact_size", len(token.packet) == 56 + 4 * len(c["s"]))
            row["token_only"] = token.metadata
        for method in ("dual", "canonical"):
            desired = processed.get(method, {}).get("0.3", {}).get("final")
            for precision, width in (("float64", 8), ("float32", 4)):
                entry = {"variant": method, "precision": precision}
                try:
                    encoded = encode_message(c["s"], c["ids"], session, method, precision)
                    ctx = session.direct_context if encoded.metadata["fallback"] else session
                    phase = time.perf_counter()
                    decoded = decode_packet(encoded.packet, ctx)
                    decode_seconds = time.perf_counter() - phase
                    reference = dense_final if encoded.metadata["fallback"] else desired
                    if reference is None:
                        raise RuntimeError("required unrounded reference unavailable")
                    direct = encode_direct(reference, session.direct_context, precision)
                    direct_decoded = decode_packet(direct.packet, session.direct_context)
                    error, direct_error = close_error(decoded.final, reference), close_error(
                        direct_decoded.final, reference)
                    wire_dtype = np.dtype("<f8" if precision == "float64" else "<f4")
                    reference_bytes = np.asarray(reference, dtype=wire_dtype, order="C").tobytes()
                    direct_exact = (np.asarray(direct_decoded.final, dtype=wire_dtype,
                                               order="C").tobytes() == reference_bytes)
                    raw_exact = (np.asarray(decoded.final, dtype=wire_dtype,
                                            order="C").tobytes() == reference_bytes)
                    self.check(row, f"{method}_{precision}_direct_exact_bytes", direct_exact)
                    if precision == "float32" and encoded.metadata["fallback"]:
                        self.check(row, method + "_float32_raw_packet_exact_bytes", raw_exact)
                    reference_ids = final_nearest_ids(reference, c["vocab"])
                    entry.update({"status": "decoded", "metadata": encoded.metadata,
                                  "packet_sha256": sha(encoded.packet), "decode_seconds": decode_seconds,
                                  "reference_method": "dense_fallback" if encoded.metadata["fallback"] else method,
                                  "final_error_descriptive": error, "direct_prefix_error": direct_error,
                                  "direct_prefix_bytes": len(direct.packet),
                                  "direct_roundtrip_exact_at_wire_precision": direct_exact,
                                  "raw_fallback_exact_at_wire_precision": raw_exact if encoded.metadata["fallback"] else None,
                                  "final_nearest_id_changes": int(np.count_nonzero(
                                      final_nearest_ids(decoded.final, c["vocab"]) != reference_ids)),
                                  "direct_prefix_nearest_id_changes": int(np.count_nonzero(
                                      final_nearest_ids(direct_decoded.final, c["vocab"]) != reference_ids)),
                                  "decoder": decoded.metadata})
                    if encoded.metadata["fallback"] and desired is not None:
                        entry["fallback_method_switch_error_descriptive"] = close_error(reference, desired)
                    k, d = c["s"].shape
                    expected = 56 + k * d * width if encoded.metadata["fallback"] else (
                        56 + 4 * k + width * k * (k - 1) // 2)
                    self.check(row, f"{method}_{precision}_packet_byte_count",
                               len(encoded.packet) == expected, actual=len(encoded.packet), expected=expected)
                    self.check(row, f"{method}_{precision}_direct_byte_count",
                               len(direct.packet) == 56 + k * d * width)
                    if precision == "float64":
                        if encoded.metadata["fallback"]:
                            self.check(row, method + "_float64_raw_packet_exact_bytes", raw_exact)
                        else:
                            # Compare to this specified algorithm, not a different
                            # dense null completion. Argmax-sensitive rows remain descriptive.
                            ref_pre = results[method].pre
                            ref_post = processed[method]["0.3"]
                            dec_post = postprocess(decoded.pre, c["vocab"], 0.3)
                            pre_error = close_error(decoded.pre, ref_pre)
                            cal_error = close_error(dec_post["calibrated"], ref_post["calibrated"])
                            self.check(row, method + "_float64_packet_pre", pre_error["passed"], error=pre_error)
                            self.check(row, method + "_float64_packet_calibrated",
                                       cal_error["passed"], error=cal_error)
                            stable = ((ref_post["margins"] > 1e-6) & (ref_post["pre_norms"] > 1e-6) &
                                      (ref_post["query_norms"] > 1e-12) &
                                      (ref_post["vocab_norms"][ref_post["ids"]] > 1e-12))
                            entry["float64_stable_rows"] = int(stable.sum())
                            entry["float64_pre_error"] = pre_error
                            entry["float64_calibrated_error"] = cal_error
                            if np.any(stable):
                                stable_error = close_error(decoded.final[stable], reference[stable])
                                self.check(row, method + "_float64_packet_stable_snap",
                                           np.array_equal(dec_post["ids"][stable], ref_post["ids"][stable]))
                                self.check(row, method + "_float64_packet_stable_final",
                                           stable_error["passed"], error=stable_error)
                                entry["float64_stable_final_error"] = stable_error
                        should_fallback = not (2 <= k < d and c["r"] is not None) or (
                            method == "dual" and
                            (c["source_rank"] != k - 1 or c["receiver_rank"] != k - 1))
                        self.check(row, method + "_float64_expected_route",
                                   encoded.metadata["fallback"] == should_fallback,
                                   expected_fallback=should_fallback)
                    self.packet_summary[(c["suite"], method, precision, "fallback" if encoded.metadata["fallback"] else "gram")] += 1
                    self.packet_summary[(c["suite"], method, precision, "packet_bytes")] += len(encoded.packet)
                    self.packet_summary[(c["suite"], method, precision, "direct_bytes")] += len(direct.packet)
                except Exception as exc:
                    entry.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                    self.check(row, f"{method}_{precision}_packet_exception", False, error=str(exc))
                row["packets"].append(entry)


    def postprocessing_control(self):
        pre = np.array([[0., 0, 0, 0], [1., 0, 0, 0]])
        vocab = np.array([[1., 0, 0, 0], [1., 0, 0, 0], [-1., 0, 0, 0]])
        row = {"id": "D-postprocessing-clamps-ties", "suite": "D", "required": True,
               "checks": [], "input_hashes": {"S": array_hash(pre), "vocabulary": array_hash(vocab)}}
        self.arrays.add(row["id"], "S", pre)
        self.arrays.add(row["id"], "vocabulary", vocab)
        row["postprocessing"] = {}
        for alpha in (0.0, 0.3):
            p = postprocess(pre, vocab, alpha)
            self.check(row, f"finite_alpha{alpha}", np.isfinite(p["final"]).all())
            self.check(row, f"first_index_tie_alpha{alpha}", np.array_equal(p["ids"], [0, 0]))
            self.check(row, f"zero_row_calibration_alpha{alpha}", np.array_equal(p["calibrated"][0], np.zeros(4)))
            row["postprocessing"][str(alpha)] = {name: p[name] for name in
                                                ("ids", "margins", "pre_norms", "final")}
            self.arrays.add(row["id"], f"final_alpha{alpha}", p["final"])
        self.emit(row)
        self.algebraic_count += 1

    def token_applicability(self):
        for k in (2, 8, 16):
            for size in (32, 256):
                for seed in SEEDS:
                    table = rng(seed, 6, k, 64, size).normal(size=(size, 64))
                    table_hash = array_hash(table)
                    for dist_index, distribution in enumerate(("uniform", "zipf_1.2")):
                        gen = rng(seed, 6, k, 64, size * 2 + dist_index)
                        probability = None
                        if dist_index:
                            probability = np.arange(1, size + 1, dtype=float) ** -1.2
                            probability /= probability.sum()
                        for index in range(64):
                            ids = gen.choice(size, size=k, replace=True, p=probability).astype(np.uint32)
                            r = table[ids]
                            centered = r - r.mean(axis=0)
                            info = spectrum_info(np.linalg.eigvalsh(centered @ centered.T))
                            eligible = (info["rank"] == k - 1 and not info["ambiguity_count"]
                                        and not info["negative_invalid"])
                            row = {"id": f"F-k{k}-V{size}-seed{seed}-{distribution}-{index}",
                                   "suite": "F", "required": False, "checks": [],
                                   "K": k, "d": 64, "vocabulary_size": size, "seed": seed,
                                   "distribution": distribution, "sequence_index": index,
                                   "token_ids": ids.tolist(), "token_ids_sha256": array_hash(ids),
                                   "vocabulary_sha256": table_hash, "R_sha256": array_hash(r),
                                   "duplicates": int(k - len(np.unique(ids))),
                                   "distinct_embeddings_rank_upper_bound": int(len(np.unique(ids)) - 1),
                                   "rank_info": info, "full_rank_eligible": bool(eligible)}
                            key = f"K{k}/V{size}/{distribution}/seed{seed}"
                            self.applicability[key]["sequences"] += 1
                            self.applicability[key]["eligible"] += int(eligible)
                            self.applicability[key]["has_duplicates"] += int(row["duplicates"] > 0)
                            self.applicability[key]["ambiguous"] += int(info["ambiguity_count"] > 0)
                            self.emit(row)

    def timing_record(self, cid, k, d, phase, name, repetition, position, duration, error=None):
        record = {"case_id": cid, "K": k, "d": d, "phase": phase, "name": name,
                  "repetition": repetition, "warmup": repetition < 0,
                  "order_position": position, "duration_ns": duration, "error": error}
        self.timing_stream.write(json.dumps(clean(record), sort_keys=True, allow_nan=False) + "\n")
        self.timing_stream.flush()
        self.timings.append(record)

    def time_group(self, c, phase, functions):
        names = list(functions)
        blocked = set()
        for repetition in range(-1, REPETITIONS):
            shift = 0 if repetition < 0 else repetition % len(names)
            ordered = names[shift:] + names[:shift]
            for position, name in enumerate(ordered):
                if name in blocked:
                    self.timing_record(c["id"], *c["s"].shape, phase, name, repetition,
                                       position, None, "not_measured_after_previous_failure")
                    continue
                started = time.perf_counter_ns()
                error = None
                try:
                    functions[name]()
                except Exception as exc:
                    error = type(exc).__name__ + ": " + str(exc)
                duration = time.perf_counter_ns() - started
                self.timing_record(c["id"], *c["s"].shape, phase, name,
                                   repetition, position, duration, error)
                if error:
                    blocked.add(name)
                    self.failures.append({"id": c["id"], "checks": [phase + "/" + name + "/timing_exception"],
                                          "error": error})
                    self.failed_checks += 1
                    self.required_checks += 1

    def prepare_timing(self):
        prepared = []
        for k in (8, 32, 64):
            for d in (128, 384, 768, 1024):
                gen = rng(404, 7, k, d, 0)
                s, r = points(k, d, k - 1, 0.1, gen), points(k, d, k - 1, 0.1, gen)
                vocab = np.vstack([r, gen.normal(size=(256, d))])
                c = case(f"T-k{k}-d{d}", "T", s, r, vocab,
                         np.arange(k, dtype=np.uint32), k - 1, k - 1, k - 1, 404, packets=True)
                row, values, processed = self.evaluate(c, retain=(k == 64 and d == 1024), emit=False)
                valid = not any(not item["passed"] for item in row["checks"])
                row["timing_eligible"] = bool(valid)
                row["timing_skip_reason"] = None if valid else "required_same_fixture_correctness_failed"
                self.emit(row)
                prepared.append((c, values, processed, valid))
        return prepared

    def run_timings(self, prepared):
        for c, values, processed, valid in prepared:
            if not valid:
                continue
            s, r = c["s"], c["r"]
            session = Session(c["vocab"], c["lambda"], c["epsilon"], 0.3)
            core = {name: (lambda name=name: ALGORITHMS[name](
                s, r, c["lambda"], c["epsilon"], details=False)) for name in ARM_ORDER}
            self.time_group(c, "core", core)
            # One common postprocessing kernel on a validated unique prefix.
            self.time_group(c, "postprocess", {"common": lambda: postprocess(
                values["dense"].pre, c["vocab"], 0.3)})
            direct_final = processed["dense"]["0.3"]["final"]
            gram_packets, direct_packets = {}, {}
            for method in ("dual", "canonical"):
                for precision in ("float64", "float32"):
                    label = method + "/" + precision
                    gram_packets[label] = encode_message(s, c["ids"], session, method, precision)
            for precision in ("float64", "float32"):
                direct_packets[precision] = encode_direct(direct_final, session.direct_context, precision)
            self.time_group(c, "gram_encode", {
                label: (lambda label=label: encode_message(
                    s, c["ids"], session, *label.split("/"))) for label in gram_packets})
            self.time_group(c, "gram_decode", {
                label: (lambda label=label: decode_packet(
                    gram_packets[label].packet, session.direct_context
                    if gram_packets[label].metadata["fallback"] else session))
                for label in gram_packets})
            self.time_group(c, "direct_encode", {
                precision: (lambda precision=precision: encode_direct(
                    direct_final, session.direct_context, precision)) for precision in direct_packets})
            self.time_group(c, "direct_decode", {
                precision: (lambda precision=precision: decode_packet(
                    direct_packets[precision].packet, session.direct_context)) for precision in direct_packets})
            token = encode_token_only(c["ids"], session)
            self.time_group(c, "token_encode", {"uint32_ids": lambda: encode_token_only(c["ids"], session)})
            self.time_group(c, "token_decode", {"embedding_lookup": lambda: decode_packet(token.packet, session)})
            ledger = {"case_id": c["id"], "K": len(s), "d": s.shape[1], "phase": "packet_accounting",
                      "cold_setup": session.cold_setup, "session_sha256": session.digest.hex(),
                      "gram": {name: value.metadata for name, value in gram_packets.items()},
                      "direct": {name: value.metadata for name, value in direct_packets.items()},
                      "token_only": token.metadata,
                      "direct_context_has_vocabulary": hasattr(session.direct_context, "vocab")}
            self.timing_stream.write(json.dumps(clean(ledger), sort_keys=True, allow_nan=False) + "\n")
            self.timing_stream.flush()

    def timing_summary(self):
        groups = defaultdict(list)
        paired = defaultdict(dict)
        case_errors = set()
        for record in self.timings:
            if record["error"]:
                case_errors.add(record["case_id"])
            if not record["warmup"] and not record["error"]:
                groups[(record["case_id"], record["phase"], record["name"])].append(record["duration_ns"])
                paired[(record["case_id"], record["phase"], record["name"])][record["repetition"]] = record["duration_ns"]
        summaries = {}
        for (cid, phase, name), samples in groups.items():
            summaries.setdefault(cid, {}).setdefault(phase, {})[name] = {
                "count": len(samples), "median_ns": float(np.median(samples)),
                "minimum_ns": int(min(samples)), "maximum_ns": int(max(samples))}
        for cid, phases in summaries.items():
            core = phases.get("core", {})
            valid = cid not in case_errors and all(
                core.get(method, {}).get("count") == REPETITIONS and
                set(paired[(cid, "core", method)]) == set(range(REPETITIONS))
                for method in ARM_ORDER)
            ratios = {}
            if valid:
                for method in ("dual", "canonical"):
                    method_samples = paired[(cid, "core", method)]
                    by_baseline = {}
                    for baseline in ("dense", "thin"):
                        baseline_samples = paired[(cid, "core", baseline)]
                        values = [baseline_samples[index] / method_samples[index]
                                  for index in range(REPETITIONS)]
                        by_baseline[baseline] = {
                            "ratios_in_repetition_order": values,
                            "median": float(np.median(values)), "minimum": float(min(values)),
                            "maximum": float(max(values)),
                            "ratio_of_medians_separately": (
                                core[baseline]["median_ns"] / core[method]["median_ns"])}
                    ratios[method] = {
                        "dense_over_method": by_baseline["dense"]["median"],
                        "thin_over_method": by_baseline["thin"]["median"],
                        "paired_ratios": by_baseline}
            direct_totals, gram_totals = {}, {}
            post = phases.get("postprocess", {}).get("common", {}).get("median_ns")
            if post is not None:
                for method in ARM_ORDER:
                    for precision in ("float64", "float32"):
                        needed = (core.get(method), phases.get("direct_encode", {}).get(precision),
                                  phases.get("direct_decode", {}).get(precision))
                        if all(x and x["count"] == REPETITIONS for x in needed):
                            direct_totals[method + "/" + precision] = sum(x["median_ns"] for x in needed) + post
            for label, encoded in phases.get("gram_encode", {}).items():
                decoded = phases.get("gram_decode", {}).get(label)
                if decoded and encoded["count"] == REPETITIONS and decoded["count"] == REPETITIONS:
                    gram_totals[label] = encoded["median_ns"] + decoded["median_ns"]
            phases["comparison"] = {
                "valid_core_measurement": valid, "ratios": ratios,
                "direct_arithmetic_total_ns": direct_totals,
                "gram_arithmetic_total_ns": gram_totals,
                "accounting": "Direct=core+common postprocess+direct encode+direct decode; "
                              "Gram=sender encoding including full preflight/postprocess+receiver decode. "
                              "No extra core is added to Gram. These sums are not network or LLM timings."}
        return summaries

    def result(self, status, stage, started, retention=None):
        expected = {"A": 120, "B": 27, "C": 8, "D": 31, "E": 56, "F": 2304, "T": 12}
        return {
            "study": "alignment_numerical_v1", "status": status, "stage": stage,
            "design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256,
            "expected_counts": expected, "observed_counts": dict(self.counts),
            "required_checks": self.required_checks, "failed_required_checks": self.failed_checks,
            "first_failures": self.failures[:20],
            "failure_case_count": len({item["id"] for item in self.failures}),
            "failure_entry_count": len(self.failures),
            "support_counts": {"/".join(key): value for key, value in sorted(self.support.items())},
            "maximum_absolute_errors": {"/".join(key): value for key, value in sorted(self.max_errors.items())},
            "maximum_error_scope": "Keys are suite/method/metric; suite C is descriptive stress, kept separate.",
            "packet_aggregate": {"/".join(key): value for key, value in sorted(self.packet_summary.items())},
            "packet_aggregate_scope": "Keys are suite/variant/precision/measure; timed-fixture validation is suite T.",
            "applicability_by_seed": {key: dict(value) for key, value in self.applicability.items()},
            "limit_rows": self.limit_rows, "timing": self.timing_summary(),
            "timing_coverage": getattr(self, "timing_coverage", {"status": "not_finalized"}),
            "phase_seconds": self.phases, "elapsed_seconds": time.perf_counter() - started,
            "retention": retention, "notes": [
                "Float64 independent reference; not upstream float32 bit identity.",
                "Canonical nonunique outputs are a separate optimum.",
                "Stress cases and float32 packet errors are descriptive.",
                "Unexpected unsupported required cases count as failures.",
                "Close/tied vocabulary choices retain final-vector differences without a false equality guarantee.",
                "Synthetic applicability rates are not language-model measurements.",
                "Dense and thin-SVD baselines are both reported; this is not new Procrustes mathematics.",
                "Token-only bytes and asymmetric cold vocabulary requirements are retained.",
                "No end-to-end model latency, model accuracy, or English-bandwidth claim."]}

    def close(self):
        self.stream.close()
        self.timing_stream.close()


def timing_coverage(prepared, records):
    phase_names = {
        "core": ARM_ORDER, "postprocess": ("common",),
        "gram_encode": ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32"),
        "gram_decode": ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32"),
        "direct_encode": ("float64", "float32"), "direct_decode": ("float64", "float32"),
        "token_encode": ("uint32_ids",), "token_decode": ("embedding_lookup",)}
    expected = {(c["id"], phase, name, repetition)
                for c, _, _, valid in prepared if valid
                for phase, names in phase_names.items() for name in names
                for repetition in range(-1, REPETITIONS)}
    keys = [(r["case_id"], r["phase"], r["name"], r["repetition"]) for r in records]
    observed = set(keys)
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    errors = [key for key, record in zip(keys, records) if record["error"]]
    order_errors = []
    for key, record in zip(keys, records):
        names = phase_names.get(record["phase"], ())
        if record["name"] in names:
            shift = 0 if record["repetition"] < 0 else record["repetition"] % len(names)
            expected_position = (names.index(record["name"]) - shift) % len(names)
            if record["order_position"] != expected_position:
                order_errors.append(key)
    return {"passed": observed == expected and not duplicates and not errors and not order_errors,
            "expected_rows": len(expected), "observed_rows": len(keys),
            "missing": [list(v) for v in sorted(expected - observed)],
            "unexpected": [list(v) for v in sorted(observed - expected)],
            "duplicates": [list(v) for v in sorted(duplicates)],
            "error_count": len(errors), "first_error_keys": [list(v) for v in errors[:20]],
            "order_error_count": len(order_errors),
            "first_order_error_keys": [list(v) for v in order_errors[:20]]}


def report_text(result):
    lines = [
        "# Alignment numerical study",
        "",
        "Status: **" + result["status"] + "**. Stage: " + result["stage"] + ".",
        "",
        "This is a float64 numerical study of an alignment interface, not a language-model experiment.",
        "The canonical method is a separate optimal completion when the aligned product is nonunique.",
        "",
        "## Coverage and mathematical checks",
        "",
        f"Required checks: {result['required_checks']}; failed required checks: {result['failed_required_checks']}.",
        "An unexpected unsupported required method counts as failed coverage. Stress inputs remain descriptive.",
        "",
        "| Suite | Expected | Completed |",
        "|---|---:|---:|",
    ]
    for suite, expected in result["expected_counts"].items():
        lines.append(f"| {suite} | {expected} | {result['observed_counts'].get(suite, 0)} |")
    lines += ["", "## Alignment-core timing", "",
              "Table entries are medians of eight within-repetition ratios on one CPU host. A ratio above one favors the named Gram method.",
              "Both the dense and standard thin-SVD baselines are retained; a gain over dense alone is not new fast Procrustes mathematics.",
              "", "| Fixture | Dual vs dense | Dual vs thin | Canonical vs dense | Canonical vs thin |",
              "|---|---:|---:|---:|---:|"]
    for cid, item in result["timing"].items():
        comp = item["comparison"]
        if comp["valid_core_measurement"]:
            ratio = comp["ratios"]
            lines.append(f"| {cid} | {ratio['dual']['dense_over_method']:.3f} | "
                         f"{ratio['dual']['thin_over_method']:.3f} | "
                         f"{ratio['canonical']['dense_over_method']:.3f} | "
                         f"{ratio['canonical']['thin_over_method']:.3f} |")
        else:
            lines.append(f"| {cid} | incomplete/failed | incomplete/failed | incomplete/failed | incomplete/failed |")
    lines += ["", "Complete raw timings, warmups, size ledgers, and failures are in timings.jsonl.",
              "The JSON report separately sums direct core + common postprocessing + direct encode/decode,",
              "and Gram encoding (already including sender preflight/postprocessing) + receiver decode.",
              "It does not add the core a second time to the Gram route. These arithmetic sums exclude network transfer and LLM inference.",
              "", "## Packet accounting and precision", "",
              "The frame header is 56 bytes. A Gram packet additionally sends K uint32 token IDs and",
              "K(K-1)/2 floating-point values. Direct prefixes send Kd values; token-only packets send K IDs.",
              "Each case records exact bytes, float32 Gram error, matched direct-prefix rounding error,",
              "snap/nearest-ID changes, and sender-side fallback costs.",
              "The Gram and token-lookup decoders require the shared vocabulary; direct-prefix decoding",
              "is also exercised with a context containing no vocabulary. Cold vocabulary/setup cost is additional.",
              "", "## Rank, limits, and interpretation", "",
              "Full-rank equality, canonical Gram/objective validity, and close-margin vocabulary behavior",
              "are reported separately. A canonical-versus-dense difference in a nonunique case is expected.",
              "The ideal weak-regularization limits and the fixed 1e-6 covariance floor are separate tracks.",
              "Synthetic duplicate-token rates do not estimate natural-language applicability.",
              "No result establishes a new semantic language, model quality, end-to-end model speed, or lower bandwidth than text.",
              "", "## Evidence", "",
              "cases.jsonl retains every algebraic, token-applicability, and timing-validation row.",
              "arrays.npz and arrays_manifest.json retain all algebraic inputs and available pre-calibration",
              "outputs plus the largest timing fixture. Smaller timing fixtures have fixed seeds and input hashes.",
              "The completed algebraic archive is checkpointed before timing begins.",
              f"Frozen design: {DESIGN_COMMIT}; SHA256 {DESIGN_SHA256}.",
              ""]
    if result["first_failures"]:
        lines += ["## First recorded failures", ""]
        for failure in result["first_failures"]:
            lines.append("- " + failure["id"] + ": " + ", ".join(failure["checks"]))
        lines.append("")
    return "\n".join(lines)


def runtime_and_sources(output):
    import contextlib
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        np.show_config()
    runtime = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
        "numpy_configuration": capture.getvalue(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_commit": os.environ.get("GITHUB_SHA"),
    }
    atomic_write(output / "runtime_manifest.json", json_bytes(runtime))
    root = Path(__file__).resolve().parents[1]
    files = [root / "latent_bridge" / name for name in
             ("alignment_core.py", "alignment_packet.py", "alignment_numerical.py")]
    files += [root / "docs" / "alignment_numerical_design.md", root / "docs" / "alignment_geometry.md"]
    files += sorted((root / "tests").glob("test_alignment*.py"))
    manifest = {"design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA256, "files": {}}
    for path in files:
        raw = path.read_bytes()
        manifest["files"][path.relative_to(root).as_posix()] = {"sha256": sha(raw), "bytes": len(raw)}
    launcher = output / "launch_source_manifest.json"
    if launcher.exists():
        manifest["launcher_source_manifest"] = {"sha256": sha(launcher.read_bytes()),
                                                "bytes": launcher.stat().st_size}
    atomic_write(output / "source_manifest.json", json_bytes(manifest))
    if manifest["files"]["docs/alignment_numerical_design.md"]["sha256"] != DESIGN_SHA256:
        raise RuntimeError("frozen design bytes differ")
    if runtime["python"] != "3.11.18" or runtime["numpy"] != "2.4.6":
        raise RuntimeError("unfrozen Python/NumPy version")
    if any(value != "1" for value in runtime["threads"].values()):
        raise RuntimeError("all declared BLAS/OMP thread settings must be one")
    return runtime, manifest


def write_outputs(study, status, stage, started, retention):
    result = study.result(status, stage, started, retention)
    atomic_write(study.output / "results.json", json_bytes(result))
    atomic_write(study.output / "report.md", report_text(result).encode("utf-8"))
    atomic_write(study.output / "status.json", json_bytes({
        "status": status, "stage": stage, "observed_counts": dict(study.counts),
        "failed_required_checks": study.failed_checks}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # The wrapper's launcher, source manifest, and worker.log may already exist.
    own_outputs = ("cases.jsonl", "timings.jsonl", "arrays.npz", "arrays_manifest.json",
                   "results.json", "report.md", "status.json", "worker_failure.json")
    if any((output / name).exists() for name in own_outputs):
        raise RuntimeError("refusing to overwrite a previous numerical study")
    started = time.perf_counter()
    runtime_and_sources(output)
    study = Study(output)
    retention = None
    stage = "algebraic"
    try:
        atomic_write(output / "status.json", json_bytes({"status": "running", "stage": stage}))
        phase = time.perf_counter()
        for index, c in enumerate(algebraic_cases(), 1):
            study.evaluate(c)
            if index % 40 == 0:
                retention = study.arrays.save("algebraic_checkpoint")
                write_outputs(study, "running", stage, started, retention)
                print(f"Algebraic cases completed: {index}; failed required checks: {study.failed_checks}", flush=True)
        study.postprocessing_control()
        study.phases["algebraic_seconds"] = time.perf_counter() - phase
        retention = study.arrays.save("algebraic_complete_before_applicability")
        stage = "applicability"
        write_outputs(study, "running", stage, started, retention)
        phase = time.perf_counter()
        study.token_applicability()
        study.phases["applicability_seconds"] = time.perf_counter() - phase
        stage = "timing_precheck"
        write_outputs(study, "running", stage, started, retention)
        phase = time.perf_counter()
        prepared = study.prepare_timing()
        study.phases["timing_precheck_seconds"] = time.perf_counter() - phase
        # Preserve every algebraic array and largest timing fixture before warmups.
        retention = study.arrays.save("all_algebraic_and_largest_timing_before_timing")
        stage = "timing"
        write_outputs(study, "running", stage, started, retention)
        phase = time.perf_counter()
        study.run_timings(prepared)
        study.phases["timing_seconds"] = time.perf_counter() - phase
        expected = {"A": 120, "B": 27, "C": 8, "D": 31, "E": 56, "F": 2304, "T": 12}
        if dict(study.counts) != expected:
            study.failed_checks += 1
            study.required_checks += 1
            study.failures.append({"id": "study_coverage", "checks": ["exact_fixed_suite_counts"],
                                   "observed": dict(study.counts), "expected": expected})
        coverage = timing_coverage(prepared, study.timings)
        study.timing_coverage = coverage
        study.required_checks += 1
        if not coverage["passed"]:
            study.failed_checks += 1
            study.failures.append({"id": "timing_coverage",
                                   "checks": ["exact_frozen_timing_coverage"],
                                   "details": coverage})
        status = "verified" if study.failed_checks == 0 else "completed_with_failures"
        result = write_outputs(study, status, "complete", started, retention)
        raw = json_bytes(result)
        if len(raw) <= 300000:
            print("ALIGNMENT_RESULTS_BEGIN", flush=True)
            print(raw.decode("utf-8"), end="", flush=True)
            print("ALIGNMENT_RESULTS_END", flush=True)
        else:
            print(json.dumps({"results_bytes": len(raw), "results_sha256": sha(raw),
                              "status": status, "detail": "full report retained in artifact"}), flush=True)
        return 0
    except Exception as exc:
        failure = {"stage": stage, "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "model_invocations": 0}
        atomic_write(output / "worker_failure.json", json_bytes(failure))
        try:
            retention = study.arrays.save("partial_after_exception")
            write_outputs(study, "incomplete", stage, started, retention)
        except Exception:
            traceback.print_exc()
        raise
    finally:
        study.close()


if __name__ == "__main__":
    raise SystemExit(main())
