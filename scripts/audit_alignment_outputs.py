"""Read-only verification of retained alignment evidence.

No study helpers, generators, model libraries, inference, or timing calls.
Numerical failures are scientific results, not evidence-integrity failures.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import numpy as np

ATOL, RTOL = 1e-10, 1e-8
DESIGN_COMMIT = "e7c467f2b097666dc474a3dbb6e273bf253e785f"
DESIGN_SHA256 = "c785e3fef3af9c8fac3bb7a04ccd7980ddfa51c121f3f11a9068eda3b7a70a5e"
ARMS = ("dense", "thin", "dual", "canonical")
COUNTS = {"A": 120, "B": 27, "C": 8, "D": 31, "E": 56, "F": 2304, "T": 12}
PHASES = {
    "core": ARMS, "postprocess": ("common",),
    "gram_encode": ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32"),
    "gram_decode": ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32"),
    "direct_encode": ("float64", "float32"), "direct_decode": ("float64", "float32"),
    "token_encode": ("uint32_ids",), "token_decode": ("embedding_lookup",),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_info(path):
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def array_info(a):
    a = np.ascontiguousarray(a)
    descriptor = json.dumps({"shape": list(a.shape), "dtype": a.dtype.str},
                            sort_keys=True, separators=(",", ":")).encode()
    return {"shape": list(a.shape), "dtype": a.dtype.str, "bytes": int(a.nbytes),
            "sha256": digest(descriptor + b"\0" + a.tobytes())}


def finite_json(value):
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def read_json(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate JSON key: " + k)
            out[k] = v
        return out
    def bad_constant(value):
        raise ValueError("nonstandard JSON number: " + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=bad_constant)


def residual(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"passed": False, "absolute": None, "relative": None,
                "reason": "shape_or_nonfinite"}
    error, scale = float(np.linalg.norm(a - b)), float(np.linalg.norm(b))
    return {"passed": error <= ATOL + RTOL * scale, "absolute": error,
            "relative": error / max(scale, 1e-300), "reference_norm": scale}


def rank_info(singular_values):
    values = np.square(singular_values)
    scale = max(1.0, float(np.max(values, initial=0.0)))
    scaled = values / scale
    return {"rank": int(np.count_nonzero(scaled > 1e-10)),
            "ambiguous": bool(np.any((scaled >= 1e-12) & (scaled <= 1e-8)))}


def dense_algebra(s, r, lam, eps):
    """Independent direct covariance calculation; no production helpers."""
    k, d = s.shape
    sc, rc = s - s.mean(axis=0), r - r.mean(axis=0)
    powers = []
    for centered in (sc, rc):
        covariance = centered.T @ centered / k + lam * np.eye(d)
        eigenvalues, basis = np.linalg.eigh((covariance + covariance.T) / 2)
        eigenvalues = np.maximum(eigenvalues, eps)
        if not np.isfinite(eigenvalues).all() or np.min(eigenvalues) <= 0:
            raise ValueError("nonpositive_or_nonfinite_covariance")
        inverse = (basis / np.sqrt(eigenvalues)) @ basis.T
        square = (basis * np.sqrt(eigenvalues)) @ basis.T
        powers.append((inverse, square))
    sw, rw = sc @ powers[0][0], rc @ powers[1][0]
    left, singular, right = np.linalg.svd(sw.T @ rw, full_matrices=False)
    repair = np.linalg.slogdet(left @ right)[0] < 0
    optimum = float(np.sum(singular) - (2 * singular[-1] if repair else 0))
    if repair:
        left[:, -1] *= -1
    reference = sw @ (left @ right) @ powers[1][1] + r.mean(axis=0)
    return {
        "source_gram": sw @ sw.T, "source_white": sw, "target_white": rw,
        "receiver_inverse": powers[1][0], "receiver_mean": r.mean(axis=0),
        "reference_pre": reference, "proper_trace_optimum": optimum,
        "proper_determinant_repair": bool(repair),
        "source": rank_info(np.linalg.svd(sc, compute_uv=False)),
        "receiver": rank_info(np.linalg.svd(rc, compute_uv=False)),
        "cross": rank_info(singular),
    }


def matrix_checks(pre, base, unique):
    white = (pre - base["receiver_mean"]) @ base["receiver_inverse"]
    checks = {
        "gram": residual(white @ white.T, base["source_gram"]),
        "proper_objective": residual(
            np.asarray(np.sum(white * base["target_white"])),
            np.asarray(base["proper_trace_optimum"])),
    }
    # At full ambient source rank, the orthogonal representative is unique;
    # matching Gram/objective alone does not establish its SO(d) orientation.
    d = pre.shape[1]
    if base["source"]["rank"] == d:
        q, _, _, _ = np.linalg.lstsq(base["source_white"], white, rcond=None)
        checks["orientation_reconstruction"] = residual(base["source_white"] @ q, white)
        checks["orientation_orthogonality"] = residual(q.T @ q, np.eye(d))
        sign, logabs = np.linalg.slogdet(q)
        checks["proper_orientation"] = {
            "passed": bool(sign > 0 and np.isfinite(logabs)),
            "determinant_sign": float(sign), "log_abs_determinant": float(logabs)}
    if unique:
        checks["unique_pre"] = residual(pre, base["reference_pre"])
    return checks


def expected_ids():
    """Enumerate identifiers only: no fixture values, generators or randomness."""
    names = set()
    for k in (2, 4, 8, 16):
        for seed in (101, 202, 303):
            for spectrum in (0, 1):
                for lam in (0.0, 1e-8, 1e-6, 1e-3, 0.1):
                    names.add(f"A-k{k}-seed{seed}-spectrum{spectrum}-lambda{lam}")
    for geometry in range(9):
        for lam in (1e-8, 1e-3, 0.1):
            names.add(f"B-geometry{geometry}-lambda{lam}")
    for side in (0, 1):
        for ratio in (1e-4, 1e-5, 1e-6, 1e-8):
            names.add(f"C-side{side}-ratio{ratio}")
    for seed in (101, 202, 303):
        for variant in ("baseline", "translation", "rotation"):
            names.add(f"D-isometry-{seed}-{variant}")
    for lam in (0.0, 1e-8, 1e-3, 0.1):
        for bit in (-1, 1):
            names.add(f"D-sign-lambda{lam}-bit{bit}")
    for mean in (0, 1):
        for scale in (0.25, 1.0, 4.0):
            names.add(f"D-scale-mean{mean}-scale{scale}")
    names.update(("D-null-baseline", "D-null-rotation", "D-one-row--1",
                  "D-one-row-1", "D-missing-ids", "D-dimension-k4-d4",
                  "D-dimension-k4-d3", "D-postprocessing-clamps-ties"))
    for geometry in range(4):
        for eps in (0.0, 1e-6):
            lambdas = (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)
            if eps:
                lambdas += (1e-8, 0.0)
            for lam in lambdas:
                names.add(f"E-geometry{geometry}-eps{eps}-lambda{lam}")
    for k in (2, 8, 16):
        for size in (32, 256):
            for seed in (101, 202, 303):
                for distribution in ("uniform", "zipf_1.2"):
                    for index in range(64):
                        names.add(f"F-k{k}-V{size}-seed{seed}-{distribution}-{index}")
    for k in (8, 32, 64):
        for d in (128, 384, 768, 1024):
            names.add(f"T-k{k}-d{d}")
    return names

class Audit:
    def __init__(self, root, expected_execution=None):
        self.root = root
        self.integrity, self.consistency, self.science, self.notes = [], [], [], []
        self.expected_execution = expected_execution
        self.execution, self.report, self.rows, self.timing_rows = {}, {}, [], []
        self.array_manifest, self.arrays = {}, {}
        self.complete = False
        self.evidence = {}

    def issue(self, category, message, **details):
        getattr(self, category).append({"message": message, **details})

    def path(self, name):
        parts = PurePosixPath(name)
        if parts.is_absolute() or any(p in ("", ".", "..") for p in parts.parts) or "\\" in name:
            raise ValueError("unsafe artifact path")
        path = self.root.joinpath(*parts.parts)
        if any(p.is_symlink() for p in (path, *path.parents) if p != self.root.parent):
            raise ValueError("symlink in artifact path")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("artifact path escapes root")
        return path

    def jsonl(self, name):
        path = self.path(name)
        if not path.exists():
            return []
        lines = path.read_bytes().splitlines()
        out = []
        for i, raw in enumerate(lines):
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                out.append(row)
            except (ValueError, UnicodeError):
                if i == len(lines) - 1 and not self.complete:
                    self.notes.append(name + ": retained incomplete final JSONL record")
                else:
                    self.issue("consistency", "invalid_jsonl_record", file=name, line=i + 1)
        return out

    def retention(self):
        report_path = self.path("results.json")
        if report_path.exists():
            self.report = read_json(report_path)
        self.complete = (self.report.get("stage") == "complete" and
                         self.report.get("status") in ("verified", "completed_with_failures"))
        execution_path = self.path("execution.json")
        if not execution_path.exists():
            self.notes.append("No finalized execution inventory; partial evidence is not independently anchored here.")
            if self.expected_execution:
                self.issue("integrity", "expected_execution_missing")
            if self.complete:
                self.issue("consistency", "complete_report_without_execution_inventory")
        else:
            self.execution = read_json(execution_path)
            actual_execution = file_info(execution_path)
            if self.expected_execution and actual_execution["sha256"] != self.expected_execution:
                self.issue("integrity", "execution_anchor_mismatch", actual=actual_execution)
            expected = self.execution.get("retained_files", {})
            actual_names = set()
            for p in self.root.rglob("*"):
                if p.is_symlink():
                    self.issue("integrity", "artifact_symlink", path=str(p.relative_to(self.root)))
                elif p.is_file() and p.name != "execution.json":
                    actual_names.add(p.relative_to(self.root).as_posix())
            if actual_names != set(expected):
                self.issue("integrity", "execution_file_inventory_mismatch",
                           missing=sorted(set(expected) - actual_names),
                           unexpected=sorted(actual_names - set(expected)))
            for name, wanted in expected.items():
                p = self.path(name)
                if p.is_file() and file_info(p) != wanted:
                    self.issue("integrity", "retained_file_hash_mismatch", file=name)
            self.evidence["execution"] = {
                **actual_execution, "external_digest_supplied": bool(self.expected_execution),
                "source_commit": self.execution.get("github_commit"),
                "run_id": self.execution.get("github_run_id"),
                "worker_returncode": self.execution.get("worker_returncode"),
                "failure": self.execution.get("failure"),
                "retained_file_count": len(expected)}
        for name in ("cases.jsonl", "timings.jsonl"):
            raw, compressed = self.path(name), self.path(name + ".gz")
            if raw.exists() and compressed.exists():
                h, size = hashlib.sha256(), 0
                expected_size = raw.stat().st_size
                with gzip.open(compressed, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(block)
                        if size > expected_size:
                            raise ValueError("gzip expands beyond retained raw size")
                        h.update(block)
                if {"sha256": h.hexdigest(), "bytes": size} != file_info(raw):
                    self.issue("consistency", "gzip_raw_disagreement", file=name)
                receipt = self.execution.get("compression_receipts", {}).get(name)
                if receipt and receipt != {"raw": file_info(raw), "gzip": file_info(compressed),
                                           "round_trip_verified": True}:
                    self.issue("consistency", "compression_receipt_disagreement", file=name)
            elif raw.exists() and not compressed.exists():
                self.notes.append(name + ": no finalized gzip; raw evidence remains available")
        self.rows = self.jsonl("cases.jsonl")
        self.timing_rows = self.jsonl("timings.jsonl")

    def coverage_and_reports(self):
        expected = expected_ids()
        ids = [r.get("id") for r in self.rows]
        duplicated = [x for x, n in Counter(ids).items() if n > 1]
        missing, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
        counts = dict(Counter(r.get("suite") for r in self.rows))
        if duplicated or extra:
            self.issue("consistency", "duplicate_or_unexpected_case_ids",
                       duplicated=duplicated, unexpected=extra)
        raw_required, raw_failed = 0, 0
        failed_ids = []
        for row in self.rows:
            failed = [c["name"] for c in row.get("checks", []) if not c.get("passed", False)]
            required = row.get("required", False)
            if required:
                raw_required += len(row.get("checks", []))
                raw_failed += len(failed)
                if failed:
                    failed_ids.append(row["id"])
            expected_failed = failed if required else []
            expected_status = "failed" if required and failed else "verified" if required else "descriptive"
            if (row.get("required_check_failures") != expected_failed or
                    row.get("status") != expected_status):
                self.issue("consistency", "case_check_status_disagreement", case=row.get("id"))
        reported_counts = self.report.get("observed_counts", {})
        if self.complete and (counts != reported_counts or counts != COUNTS or missing):
            self.issue("consistency", "complete_case_coverage_disagreement",
                       observed=counts, reported=reported_counts, missing=missing)
        elif not self.complete:
            for suite, count in reported_counts.items():
                if count > counts.get(suite, 0):
                    self.issue("consistency", "snapshot_count_exceeds_raw_rows", suite=suite)
        if self.report:
            if (self.report.get("design_commit") != DESIGN_COMMIT or
                    self.report.get("design_sha256") != DESIGN_SHA256):
                self.issue("consistency", "frozen_design_identity_disagreement")
        self.evidence["coverage"] = {
            "complete_report": self.complete, "expected_counts": COUNTS, "observed_counts": counts,
            "missing_case_ids": missing, "raw_required_checks": raw_required,
            "raw_failed_required_checks": raw_failed, "raw_failed_case_ids": sorted(set(failed_ids)),
            "array_scope": "All retained algebraic fixtures and largest timing fixture; no regeneration."}
        return raw_required, raw_failed

    def retained_arrays(self):
        manifest_path = self.path("arrays_manifest.json")
        archive = self.path("arrays.npz")
        if not manifest_path.exists() or not archive.exists():
            self.notes.append("No complete array archive/manifest pair is available.")
            return
        self.array_manifest = read_json(manifest_path)
        if file_info(archive) != self.array_manifest.get("npz"):
            self.issue("consistency", "array_archive_manifest_disagreement")
        expected = self.array_manifest.get("arrays", {})
        with np.load(archive, allow_pickle=False) as data:
            if set(data.files) != set(expected):
                self.issue("consistency", "array_key_inventory_disagreement")
            for key in data.files:
                a = data[key]
                if a.dtype.hasobject or a.dtype.kind not in "biufc":
                    self.issue("consistency", "nonnumeric_array", key=key)
                    continue
                actual = array_info(a)
                if expected.get(key) != actual or key != "a_" + actual["sha256"]:
                    self.issue("consistency", "array_descriptor_hash_disagreement", key=key)
                self.arrays[key] = a
        rows = {r["id"]: r for r in self.rows}
        for cid, mapping in self.array_manifest.get("cases", {}).items():
            row = rows.get(cid)
            if row is None:
                self.notes.append(cid + ": retained arrays precede a complete case record")
                continue
            for name, key in mapping.items():
                if key not in self.arrays:
                    self.issue("consistency", "missing_mapped_array", case=cid, name=name)
                    continue
                claimed = row.get("input_hashes", {}).get(name)
                if claimed is not None and array_info(self.arrays[key])["sha256"] != claimed:
                    self.issue("consistency", "case_input_array_hash_disagreement", case=cid, name=name)
        self.evidence["array_retention"] = {
            "archive": file_info(archive), "phase": self.array_manifest.get("phase"),
            "arrays": len(self.arrays), "mapped_cases": len(self.array_manifest.get("cases", {}))}

    def algebra(self):
        totals, worst = Counter(), {}
        mapped = self.array_manifest.get("cases", {})
        required_missing = []
        for row in self.rows:
            cid = row["id"]
            if row["suite"] not in ("A", "B", "C", "D", "E") and cid != "T-k64-d1024":
                continue
            if cid == "D-postprocessing-clamps-ties":
                continue
            mapping = mapped.get(cid, {})
            if "S" not in mapping or mapping["S"] not in self.arrays:
                if row.get("required"):
                    required_missing.append(cid)
                continue
            s = np.asarray(self.arrays[mapping["S"]], dtype=np.float64)
            if row.get("input_hashes", {}).get("R") is not None and (
                    "R" not in mapping or mapping["R"] not in self.arrays):
                self.issue("science", "required_reference_array_missing", case=cid,
                           required=bool(row.get("required")), incomplete_evidence=True)
                continue
            r = np.asarray(self.arrays[mapping["R"]], dtype=np.float64) if "R" in mapping else None
            k, d = s.shape
            if (k, d) != (row.get("K"), row.get("d")):
                self.issue("consistency", "declared_input_shape_disagreement", case=cid)
            scope = r is not None and 2 <= k < d
            full = scope and row.get("source_rank") == k - 1 and row.get("receiver_rank") == k - 1
            bypass = r is None or k < 2
            base = None
            if not bypass:
                try:
                    base = dense_algebra(s, r, row["lambda"], row["epsilon"])
                except Exception as error:
                    self.issue("science", "independent_reference_unavailable", case=cid,
                               required=bool(row.get("required")), error=str(error))
                    continue
            unique = bool(scope and row["suite"] != "C" and base is not None and
                          base["source"]["rank"] == base["cross"]["rank"] and
                          base["source"]["rank"] == row.get("source_rank") and
                          base["cross"]["rank"] == row.get("cross_rank") and
                          not any(base[name]["ambiguous"] for name in ("source", "receiver", "cross")))
            if row.get("required") and full and not unique:
                self.issue("science", "required_unique_scope_unconfirmed", case=cid, required=True,
                           independent_ranks={name: base[name] for name in ("source", "receiver", "cross")})
                if row.get("status") == "verified":
                    self.issue("consistency", "runner_verified_unsupported_unique_scope", case=cid)
            for method in ARMS:
                recorded = row.get("methods", {}).get(method, {})
                expected_support = method == "dense" or (scope if method == "canonical" else full)
                if r is None and method != "dense":
                    expected_support = False
                if row.get("required") and expected_support and recorded.get("status") != "supported":
                    self.issue("science", "required_method_not_supported", case=cid, method=method,
                               required=True, recorded_status=recorded.get("status"),
                               reason=recorded.get("reason", recorded.get("error")))
                if not scope and method != "dense" and recorded.get("status") == "supported":
                    self.issue("science", "fast_method_supported_outside_guard", case=cid,
                               method=method, required=bool(row.get("required")))
                key = mapping.get("pre_" + method)
                if key not in self.arrays:
                    if recorded.get("status") == "supported":
                        self.issue("consistency", "supported_pre_output_not_retained",
                                   case=cid, method=method)
                    continue
                pre = np.asarray(self.arrays[key], dtype=np.float64)
                if bypass:
                    checks = {"raw_bypass": residual(pre, s)}
                else:
                    try:
                        checks = matrix_checks(pre, base, unique)
                    except Exception as error:
                        checks = {"matrix_evaluation": {"passed": False, "reason": str(error)}}
                totals[(row["suite"], method, "audited")] += 1
                totals[(row["suite"], method, "unique" if unique else "nonunique_or_boundary")] += 1
                failed = {name: value for name, value in checks.items() if not value["passed"]}
                for name, value in checks.items():
                    label = "/".join((row["suite"], method, name))
                    absolute = value.get("absolute")
                    if absolute is not None and (label not in worst or absolute > worst[label]["absolute"]):
                        worst[label] = {"case": cid, **value}
                if failed:
                    totals[(row["suite"], method, "failed")] += 1
                    self.issue("science", "retained_matrix_check_failed", case=cid, method=method,
                               required=bool(row.get("required")), checks=failed,
                               unique_product_scope=unique)
                    if row.get("required") and row.get("status") == "verified":
                        self.issue("consistency", "runner_verified_independent_matrix_failure",
                                   case=cid, method=method)
        if required_missing:
            self.issue("science", "required_retained_inputs_missing", cases=required_missing,
                       required=True, incomplete_evidence=True)
        self.evidence["algebra"] = {
            "counts": {"/".join(k): v for k, v in sorted(totals.items())},
            "worst_residuals": worst,
            "proper_rotation_boundary": "SO(d) trace optimum subtracts twice the smallest singular value when needed; full source rank additionally requires reconstructed orthogonal Q with positive determinant.",
            "unique_comparison_scope": "2<=K<d, non-stress, unambiguous ranks matching construction, cross rank equals sender rank.",
            "canonical_scope": "Feasibility/optimality only; no claim of the exact chosen nonunique completion."}

    def timing_arithmetic(self, raw_required, raw_failed):
        records = [r for r in self.timing_rows if r.get("phase") != "packet_accounting"]
        eligible = {r["id"] for r in self.rows if r.get("suite") == "T" and r.get("timing_eligible")}
        expected = {(cid, phase, name, rep) for cid in eligible
                    for phase, names in PHASES.items() for name in names for rep in range(-1, 8)}
        keys, groups, pair, error_cases, order_errors = [], defaultdict(list), {}, set(), []
        real_errors = []
        for row in records:
            key = (row.get("case_id"), row.get("phase"), row.get("name"), row.get("repetition"))
            keys.append(key)
            names = PHASES.get(row.get("phase"), ())
            rep = row.get("repetition")
            if row.get("name") in names and isinstance(rep, int):
                shift = 0 if rep < 0 else rep % len(names)
                position = (names.index(row["name"]) - shift) % len(names)
                if row.get("order_position") != position:
                    order_errors.append(list(key))
                if row.get("warmup") is not (rep < 0):
                    self.issue("consistency", "timing_warmup_flag_disagreement", key=list(key))
            if row.get("error"):
                error_cases.add(row.get("case_id"))
                if row["error"] != "not_measured_after_previous_failure":
                    real_errors.append(row)
            elif not isinstance(row.get("duration_ns"), int) or row["duration_ns"] < 0:
                self.issue("consistency", "invalid_successful_duration", key=list(key))
            elif isinstance(rep, int) and rep >= 0:
                group = key[:3]
                groups[group].append(row["duration_ns"])
                pair.setdefault(group, {})[rep] = row["duration_ns"]
        observed = set(keys)
        duplicates = [list(k) for k, n in Counter(keys).items() if n > 1]
        all_errors = [r for r in records if r.get("error")]
        coverage_ok = observed == expected and not duplicates and not all_errors and not order_errors
        summaries = {}
        for (cid, phase, name), values in groups.items():
            calculated = {"count": len(values), "median_ns": float(np.median(values)),
                          "minimum_ns": min(values), "maximum_ns": max(values)}
            summaries.setdefault(cid, {}).setdefault(phase, {})[name] = calculated
            if self.complete:
                claimed = self.report.get("timing", {}).get(cid, {}).get(phase, {}).get(name)
                if claimed != calculated:
                    self.issue("consistency", "timing_summary_disagreement",
                               case=cid, phase=phase, name=name, calculated=calculated)
        paired = {}
        for cid in summaries:
            valid = cid not in error_cases and all(
                len(groups.get((cid, "core", m), [])) == 8 and
                set(pair.get((cid, "core", m), {})) == set(range(8)) for m in ARMS)
            claimed = self.report.get("timing", {}).get(cid, {}).get("comparison", {})
            if self.complete and claimed.get("valid_core_measurement") != valid:
                self.issue("consistency", "timing_validity_disagreement", case=cid)
            if not valid:
                continue
            paired[cid] = {}
            for method in ("dual", "canonical"):
                paired[cid][method] = {}
                for baseline in ("dense", "thin"):
                    ratios = [pair[(cid, "core", baseline)][j] / pair[(cid, "core", method)][j]
                              for j in range(8)]
                    value = {"ratios_in_repetition_order": ratios,
                             "median": float(np.median(ratios)),
                             "minimum": min(ratios), "maximum": max(ratios),
                             "ratio_of_medians_separately":
                                 summaries[cid]["core"][baseline]["median_ns"] /
                                 summaries[cid]["core"][method]["median_ns"]}
                    paired[cid][method][baseline] = value
                    if self.complete:
                        shown = claimed.get("ratios", {}).get(method, {})
                        if (shown.get("paired_ratios", {}).get(baseline) != value or
                                shown.get(baseline + "_over_method") != value["median"]):
                            self.issue("consistency", "paired_timing_ratio_disagreement",
                                       case=cid, method=method, baseline=baseline)
        self.evidence["timing"] = {
            "remeasurements": 0, "expected_call_rows": len(expected), "observed_call_rows": len(keys),
            "coverage_passed": coverage_ok,
            "missing": [list(k) for k in sorted(expected - observed)],
            "unexpected": [list(k) for k in sorted(observed - expected)],
            "duplicates": duplicates, "order_errors": order_errors,
            "error_rows_including_unmeasured_after_failure": len(all_errors),
            "actual_exception_rows": len(real_errors), "paired_ratios": paired,
            "scope": "Arithmetic from recorded durations only; no performance counter or benchmark invocation."}
        if self.complete:
            counts_bad = self.evidence["coverage"]["observed_counts"] != COUNTS
            expected_failed = raw_failed + len(real_errors) + int(counts_bad) + int(not coverage_ok)
            expected_required = raw_required + len(real_errors) + int(counts_bad) + 1
            if (self.report.get("failed_required_checks") != expected_failed or
                    self.report.get("required_checks") != expected_required):
                self.issue("consistency", "global_required_check_aggregation_disagreement",
                           expected_failed=expected_failed, expected_required=expected_required)
            expected_status = "verified" if expected_failed == 0 else "completed_with_failures"
            if self.report.get("status") != expected_status:
                self.issue("consistency", "global_scientific_status_disagreement",
                           expected_status=expected_status)
            if self.report.get("timing_coverage", {}).get("passed") != coverage_ok:
                self.issue("consistency", "reported_timing_coverage_disagreement")

    def run(self):
        self.retention()
        if self.integrity:
            return
        raw_required, raw_failed = self.coverage_and_reports()
        self.retained_arrays()
        self.algebra()
        self.timing_arithmetic(raw_required, raw_failed)

    def receipt(self):
        required = [item for item in self.science if item.get("required")]
        return finite_json({
            "audit_status": "completed" if not self.integrity else "completed_with_integrity_issues",
            "audit_runtime": {"python": platform.python_version(), "numpy": np.__version__,
                              "script_sha256": digest(Path(__file__).read_bytes())},
            "artifact_directory": str(self.root),
            "data_integrity": {
                "status": "issues_found" if self.integrity else
                          "consistent" if self.execution else "unanchored_partial",
                "issues": self.integrity},
            "report_consistency": {"status": "issues_found" if self.consistency else "consistent",
                                   "issues": self.consistency},
            "scientific_validation": {
                "scope": "Retained pre-calibration outputs: feasibility, proper objective, scoped unique-product equivalence.",
                "required_matrix_failures_detected": bool(required),
                "issues": self.science,
                "original_study_status": self.report.get("status"),
                "complete_case_coverage": self.complete and not
                    self.evidence.get("coverage", {}).get("missing_case_ids", ["unavailable"]),
                "original_recorded_failed_required_checks": self.report.get("failed_required_checks"),
                "scientific_failures_do_not_invalidate_authentic_original_bytes": True},
            "evidence": self.evidence, "notes": self.notes,
            "new_fixture_generations": 0, "model_imports_or_inference": 0,
            "optimization_invocations": 0, "timing_measurements": 0,
            "limits": [
                "This recalculates algebra on original saved inputs; it does not repeat the study or its timings.",
                "Valid nonunique outputs need not equal a particular SVD completion; the canonical basis choice is not reimplemented.",
                "No independent postprocessing, vocabulary-search, packet decoding, or transport-byte execution is performed.",
                "Synthetic applicability embeddings and smaller timing arrays were not regenerated.",
                "Timing checks cover recorded call coverage, per-arm summaries and paired ratios; composed direct/Gram total fields are not independently recomputed.",
                "Reported required-check aggregation is checked separately from independently recalculated matrix properties.",
                "An external execution digest authenticates the caller-selected execution block, not arbitrary source history.",
                "Wall-clock chronology, physical clock accuracy, and host performance are not independently established."]})


def self_test():
    # Fixed elementary cases, not new random fixtures or study-grid execution.
    s = np.array([[1., 1., 0.], [-1., -1., 0.]])
    r = np.array([[1., 0., 0.], [-1., 0., 0.]])
    base = dense_algebra(s, r, 1e-3, 1e-6)
    pre = np.sqrt(2 * (1 + 1e-3) / (2 + 1e-3)) * r
    assert all(x["passed"] for x in matrix_checks(pre, base, True).values())
    damaged = pre.copy()
    damaged[0, 1] += 1
    assert not all(x["passed"] for x in matrix_checks(damaged, base, True).values())
    s2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [-1., -1., 0., 0.]])
    r2 = np.tile([1., 0., 0., 0.], (3, 1))
    b2 = dense_algebra(s2, r2, 0.1, 1e-6)
    alternative = (b2["reference_pre"] - r2.mean(axis=0))[:, [2, 3, 0, 1]] + r2.mean(axis=0)
    assert all(x["passed"] for x in matrix_checks(alternative, b2, False).values())
    s3 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
    r3 = s3 * np.array([-1., 1., 1.])
    b3 = dense_algebra(s3, r3, 0.1, 1e-6)
    assert b3["proper_determinant_repair"]
    assert all(x["passed"] for x in matrix_checks(b3["reference_pre"], b3, False).values())
    assert not matrix_checks(r3, b3, False)["proper_objective"]["passed"]
    # A full-rank improper output can match both Gram and the SO optimum trace.
    h = np.array([[1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)],
                  [-1/np.sqrt(2), 1/np.sqrt(6), 1/np.sqrt(12)],
                  [0., -2/np.sqrt(6), 1/np.sqrt(12)],
                  [0., 0., -3/np.sqrt(12)]])
    reflection = np.diag([-1., 1., 1.])
    quarter_turn = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    b4 = dense_algebra(h, h @ reflection, 0.1, 1e-6)
    trap = matrix_checks(h @ reflection @ quarter_turn, b4, False)
    assert trap["gram"]["passed"] and trap["proper_objective"]["passed"]
    assert trap["orientation_reconstruction"]["passed"]
    assert trap["orientation_orthogonality"]["passed"]
    assert not trap["proper_orientation"]["passed"]
    assert len(expected_ids()) == sum(COUNTS.values())
    print(json.dumps({"self_test": "passed", "random_generators": 0,
                      "study_grid_invocations": 0, "timing_measurements": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-execution-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.artifact or args.receipt:
            parser.error("--self-test cannot be combined with artifact audit")
        self_test()
        return 0
    if args.artifact is None or args.receipt is None:
        parser.error("--artifact and --receipt are required")
    root, receipt = args.artifact.resolve(), args.receipt.resolve()
    if not root.is_dir() or receipt.is_relative_to(root):
        parser.error("artifact must exist and receipt must be outside it")
    if receipt.exists():
        parser.error("refusing to replace an existing audit receipt")
    if args.expected_execution_sha256 and (
            len(args.expected_execution_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in args.expected_execution_sha256)):
        parser.error("expected execution digest must be lowercase SHA256")
    audit = Audit(root, args.expected_execution_sha256)
    unexpected_error = None
    try:
        audit.run()
    except Exception as error:
        unexpected_error = {"type": type(error).__name__, "message": str(error)}
    result = audit.receipt()
    if unexpected_error:
        result["audit_status"] = "incomplete_audit"
        result["audit_exception"] = unexpected_error
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    print(json.dumps({
        "audit_status": result["audit_status"],
        "data_integrity": result["data_integrity"]["status"],
        "report_consistency": result["report_consistency"]["status"],
        "required_matrix_failures_detected":
            result["scientific_validation"]["required_matrix_failures_detected"],
        "receipt_sha256": digest(receipt.read_bytes()),
        "receipt_bytes": receipt.stat().st_size}, sort_keys=True))
    return 2 if unexpected_error else 1 if audit.integrity else 0


if __name__ == "__main__":
    raise SystemExit(main())
