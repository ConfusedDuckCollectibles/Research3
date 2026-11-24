"""Independent retained-evidence audit for contextual-information v1.

No scientific worker, inverse, model, tokenizer, forward pass, or benchmark is
imported or invoked. The audit checks the actual retained estimates; it does not
repeat the author's alignment, recover missing observations, or certify model
logits from hashes. Run on the ORIGINAL extracted artifact, before publication
omits redundant raw JSONL files. Write the receipt outside that artifact.

Exact comparisons: files, decompression, array bytes, integer dyadic arithmetic,
IDs, counts, and collision membership. Only floating arithmetic independently
recomputed on a different NumPy/BLAS host uses the declared tolerances below.
Scientific thresholds remain the original thresholds without tolerance changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import traceback

import numpy as np

DESIGN_COMMIT = "c23a20ec420ba19680801a09ac8b053735807d38"
DESIGN_SHA = "6f50e1d3faad3240033748d7434f088ec26b7ac6716511923892e66ac0e78a09"
THEORY_SHA = "f78a41cee2b1ee90b349bcbee0fe00a588b9738863034172a8c8c09ddc3992d7"
AUTHOR_SHA = "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d"
CORE_SHA = "de8909692a928266c03e22d12b13c800b14e2495a8acad39ae6b95dfc46bfa53"
MODEL = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
SCALAR_RTOL, SCALAR_ATOL = 5e-12, 1e-15
MATRIX_RTOL, MATRIX_ATOL = 5e-12, 1e-12
TREES = "alder birch cedar elm fir maple oak willow".split()
COLORS = "red blue green yellow purple orange black white".split()
WORLDS, PROBES = ("original", "swapped"), ("unique", "repeated")
ROUTES = ("author_matched_public32", "author_ideal_public64",
          "reference_f64", "reference_f64_cast_f32")
CHANNELS = {"author_f32": "author_final_f32",
            "reference_f64_cast_f32": "reference_final_cast_f32",
            "direct_gram_f32": "direct_gram_f32", "raw_state_f32": "source"}
THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)
MODEL_FILES = {
    "config.json": (698, "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17"),
    "generation_config.json": (111, "494cbba887cc299273f0c54d7612596918e74c7881a27cbd95e4e7e2464ea090"),
    "model.safetensors": (28143920, "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1"),
    "special_tokens_map.json": (441, "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c"),
    "tokenizer.json": (2114042, "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2"),
    "tokenizer_config.json": (4834, "eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809"),
}
SOURCE_PATHS = {
    "docs/contextual_information_design.md",
    "docs/statebridge_information_recovery.md",
    "latent_bridge/alignment_core.py",
    "scripts/run_alignment_numerical.py",
    "third_party/statebridge/state_bridge.py",
    "latent_bridge/__init__.py",
    "latent_bridge/information_recovery.py",
    "latent_bridge/contextual_information.py",
    "scripts/run_contextual_information.py",
    "scripts/contextual_model_lock.json",
    "third_party/statebridge/LICENSE",
    "third_party/statebridge/README.md",
    "tests/test_information_recovery.py",
    "tests/test_contextual_information.py",
    "tests/test_contextual_execution.py",
    "tests/test_alignment_execution.py",
    ".github/workflows/contextual-information-v1.yml",
    ".github/workflows/tests.yml", "pyproject.toml",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def file_info(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def raw(array):
    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ValueError("non-real or object array")
    value = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
    return value.tobytes(order="C")


def bits_equal(a, b):
    return (a.shape == b.shape and a.dtype.newbyteorder("<") == b.dtype.newbyteorder("<")
            and raw(a) == raw(b))


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def number(x):
    value = float(x)
    return value if math.isfinite(value) else None


def errors(actual, reference):
    """Independent metric implementation, including zero-norm underflow policy."""
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    result = dict(status="invalid", reason=None, finite=False,
                  absolute_frobenius=None, relative_frobenius=None,
                  maximum_absolute_coordinate=None, reference_norm=None,
                  actual_norm=None, norm_change=None)
    if a.shape != b.shape:
        return {**result, "reason": "shape_mismatch"}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {**result, "reason": "nonfinite_input"}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = a - b
        distance, norm_b, norm_a = (float(np.linalg.norm(v)) for v in (delta, b, a))
        maximum = float(np.abs(delta).max()) if delta.size else 0.0
        relative = (distance / norm_b if norm_b > 0 else
                    (0.0 if not np.any(delta != 0) else math.inf))
        change = norm_a - norm_b
    good = all(math.isfinite(v) for v in (distance, norm_b, norm_a, maximum, relative, change))
    return dict(status="computed" if good else "nonfinite_metric",
                reason=None if good else ("zero_reference_norm" if norm_b == 0 else "nonfinite_metric"),
                finite=good, absolute_frobenius=number(distance),
                relative_frobenius=number(relative), maximum_absolute_coordinate=number(maximum),
                reference_norm=number(norm_b), actual_norm=number(norm_a), norm_change=number(change))


def finite_success(error, threshold):
    return (isinstance(error, dict) and error.get("finite") is True
            and finite_number(error.get("relative_frobenius"))
            and error["relative_frobenius"] <= threshold)


def rank_info(array):
    a = np.asarray(array, dtype=np.float64)
    singular = np.linalg.svd(a, compute_uv=False)
    cutoff = float(np.finfo(np.float64).eps * max(a.shape) * singular[0])
    return {"rank": int(np.count_nonzero(singular > cutoff)),
            "singular_values": singular.tolist(), "cutoff": cutoff,
            "rcond": float(np.finfo(np.float64).eps * max(a.shape))}


def centered_grams(source):
    s = np.asarray(source, dtype=np.float64)
    k = len(s)
    h = np.zeros((k, k - 1), dtype=np.float64)
    for j in range(1, k):
        h[:j, j-1] = 1.0 / math.sqrt(j * (j+1))
        h[j, j-1] = -j / math.sqrt(j * (j+1))
    sc = s - np.mean(s, axis=0)
    a = h.T @ sc
    full, reduced = sc @ sc.T, a @ a.T
    return sc, (full + full.T) * 0.5, (reduced + reduced.T) * 0.5, a


def integer_hash(matrix, exponent, kind):
    return sha(canonical({
        "schema": "exact-binary32-centered-gram-v1", "kind": kind,
        "shape": [len(matrix), len(matrix[0]) if matrix else 0],
        "common_exponent": exponent,
        "integers": [[str(x) for x in row] for row in matrix],
    }))


def exact_pair(a, b):
    """Independent rational oracle; no IEEE-bit parser or production helper."""
    a, b = np.asarray(a), np.asarray(b)
    if (a.ndim != 2 or a.shape != b.shape or a.dtype.kind != "f" or b.dtype.kind != "f"
            or a.dtype.itemsize != 4 or b.dtype.itemsize != 4
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("finite, equal-shape binary32 matrices required")
    exponents = [0]
    for array in (a, b):
        for scalar in array.flat:
            value = abs(float(scalar))
            if value:
                exponents.append(-149 if value < 2.0**-126 else math.frexp(value)[1] - 24)
    exponent = min(exponents)
    denominator = 1 << (-exponent)
    matrices, centered, grams = [], [], []
    k, d = a.shape
    for array in (a, b):
        integers = []
        for row in array:
            values = []
            for scalar in row:
                numerator, divisor = float(scalar).as_integer_ratio()
                if denominator % divisor:
                    raise ValueError("common exponent does not represent source")
                values.append(numerator * (denominator // divisor))
            integers.append(values)
        sums = [sum(integers[i][j] for i in range(k)) for j in range(d)]
        t = [[k * integers[i][j] - sums[j] for j in range(d)] for i in range(k)]
        gram = [[sum(x*y for x, y in zip(t[i], t[j])) for j in range(k)] for i in range(k)]
        matrices.append(integers)
        centered.append(t)
        grams.append(gram)
    result = {
        "schema_version": 1, "shape": list(a.shape),
        "numeric_values_equal": bool(np.array_equal(a, b)), "raw_bits_equal": bits_equal(a, b),
        "gram_equal": grams[0] == grams[1], "common_exponent": exponent,
        "denominator_k_squared": k*k, "denominator_power_of_two": -2*exponent,
        "integer_encoding": "sorted compact ASCII JSON; decimal-string integers; exact-binary32-centered-gram-v1",
    }
    for i, side in enumerate(("left", "right")):
        for label, matrix, kind in (("source", matrices[i], "source"),
                                    ("centered", centered[i], "centered_numerator"),
                                    ("gram", grams[i], "gram_numerator")):
            result[side + "_" + label + "_integer_sha256"] = integer_hash(matrix, exponent, kind)
    return result


def frozen_cohort():
    """Reconstruct only the one published corpus; no new fixture search."""
    rng, used, families, attempts = random.Random(41029), set(), [], 0
    for index in range(64):
        target = index % 8
        partner = (target + 1 + ((index // 8) % 7)) % 8
        counts = {"draws": 0, "rejected_original_reserved": 0, "rejected_swapped_reserved": 0}
        while attempts < 1000000:
            original = list(range(8))
            rng.shuffle(original)
            attempts += 1
            counts["draws"] += 1
            swapped = original.copy()
            swapped[target], swapped[partner] = swapped[partner], swapped[target]
            if tuple(original) in used:
                counts["rejected_original_reserved"] += 1
            elif tuple(swapped) in used:
                counts["rejected_swapped_reserved"] += 1
            else:
                used.update((tuple(original), tuple(swapped)))
                break
        else:
            raise ValueError("fixed cohort did not finish")
        identifier = f"F{index:03d}"
        question = f"What color is the {TREES[target]} specimen?"
        worlds = {}
        for name, permutation in zip(WORLDS, (original, swapped)):
            lines = [f"Family {identifier}.", "Private record:"]
            lines += [f"{tree}: {COLORS[color]}" for tree, color in zip(TREES, permutation)]
            lines += ["Question: " + question, "Answer:"]
            worlds[name] = {"permutation": permutation, "world_id": "".join(map(str, permutation)),
                            "answer_index": permutation[target], "answer": COLORS[permutation[target]],
                            "prompt": "\n".join(lines)}
        families.append({"family_id": identifier, "index": index, "target_index": target,
                         "partner_index": partner, "question": question,
                         "worlds": worlds, "generation": counts})
    reserved = [list(x) for x in sorted(used)]
    return {"schema_version": 1, "seed": 41029, "trees": TREES, "colors": COLORS,
            "families": families, "generation": {"total_candidates": attempts,
            "accepted_families": 64, "reserved_world_count": 128},
            "reserved_worlds": reserved, "reserved_worlds_sha256": sha(canonical(reserved)),
            "reservation_hash_encoding": "canonical ASCII JSON of lexicographically sorted integer lists"}


def fixed_plan(cohort):
    return {"schema_version": 1, "expected_count": 256, "planned_records": [
        {"record_id": f"{f['family_id']}/{world}/{probe}", "family_id": f["family_id"],
         "world": world, "probe": probe, "question": f["question"], "target_index": f["target_index"],
         "answer_index": f["worlds"][world]["answer_index"]}
        for f in cohort["families"] for world in WORLDS for probe in PROBES]}


def finite_summary(values):
    items = [float(x) for x in values if finite_number(x)]
    return {"finite_count": len(items), "minimum": min(items) if items else None,
            "median": statistics.median(items) if items else None,
            "maximum": max(items) if items else None}


def ceilings(collisions, distinct, planned=64):
    missing = planned - collisions - distinct
    if min(collisions, distinct, missing) < 0:
        raise ValueError("invalid collision population")
    observed = collisions + distinct
    return {"expected_pairs": planned, "confirmed_collisions": collisions,
            "confirmed_distinct": distinct, "missing_or_invalid": missing,
            "full_codebook_ceiling": 1-collisions/(2*planned) if missing == 0 else None,
            "full_codebook_ceiling_bounds": [1-(collisions+missing)/(2*planned), 1-collisions/(2*planned)],
            "observed_only_ceiling": 1-collisions/(2*observed) if observed else None,
            "observed_pair_denominator": observed}


def collision_summary(rows):
    result = {"channels": {}}
    for name in (*CHANNELS, "token_only_f32"):
        states = [row.get("channels", {}).get(name, {}).get("byte_collision") for row in rows]
        result["channels"][name] = ceilings(sum(x is True for x in states), sum(x is False for x in states))
    for destination, field in (("preexisting_source_collision_ids", "preexisting_source_collision"),
                               ("alignment_created_witnesses", "alignment_created_collision"),
                               ("exact_gram_loss_witnesses", "exact_gram_loss_collision")):
        result[destination] = [r["pair_id"] for r in rows if r.get(field) is True]
    result["preexisting_source_collision_count"] = len(result["preexisting_source_collision_ids"])
    result["exact_source_comparison_failures"] = [r["pair_id"] for r in rows
                                                if "exact_source_comparison_error" in r]
    return result


def independent_summary(rows, pairs, completed):
    result = {"schema_version": 1,
              "study_complete": bool(completed and len(rows) == 256 and all(r["status"] == "completed" for r in rows)),
              "planned_records": 256, "recorded_records": len(rows),
              "completed_records": sum(r["status"] == "completed" for r in rows),
              "error_records": sum(r["status"] == "error" for r in rows),
              "missing_records": 256-len(rows), "cohorts": {}}
    for probe in PROBES:
        group = [r for r in rows if r["probe"] == probe]
        lookup = {r["record_id"]: r for r in group}
        if len(lookup) != len(group):
            raise ValueError("duplicate numerical rows")
        recovery = {}
        for name in ROUTES:
            statuses, reasons, values = Counter(), Counter(), []
            counts = {str(t): 0 for t in THRESHOLDS}
            for r in group:
                outcome = r.get("recovery", {}).get(name)
                if outcome is None:
                    statuses["not_observed"] += 1
                    continue
                statuses[outcome["status"]] += 1
                if outcome.get("reason"):
                    reasons[outcome["reason"]] += 1
                error = outcome.get("gram_error")
                if error:
                    values.append(error.get("relative_frobenius"))
                for t in THRESHOLDS:
                    counts[str(t)] += int(finite_success(error, t))
            statuses["not_observed"] += 128-len(group)
            recovery[name] = {"planned_denominator": 128, "statuses": dict(statuses),
                              "rejection_reasons": dict(reasons), "success_counts": counts,
                              "finite_relative_errors": finite_summary(values)}
        native = [r for r in group if "extraction" in r]
        complete_pairs = both_correct = 0
        for f in range(64):
            pair = [lookup.get(f"F{f:03d}/{w}/{probe}") for w in WORLDS]
            if all(r is not None and "extraction" in r for r in pair):
                complete_pairs += 1
                both_correct += int(all(r["extraction"]["native_correct"] for r in pair))
        correct = sum(r["extraction"]["native_correct"] for r in native)
        result["cohorts"][probe] = {
            "planned_records": 128, "recorded": len(group),
            "completed": sum(r["status"] == "completed" for r in group), "missing": 128-len(group),
            "recovery": recovery,
            "direct_gram_relative_error": finite_summary([r.get("direct_gram_error", {}).get("relative_frobenius") for r in group]),
            "source_whitening_inverse_condition_factors": finite_summary([
                v for r in group for v in r.get("source_geometry", {}).get("inverse_condition_factors_public32", [])]),
            "forced_token_mean_log_probability": finite_summary([
                r["extraction"]["forced_log_probability_mean"] for r in native]),
            "native_readout": {"observed_records": len(native), "correct_records": correct,
                "planned_record_denominator": 128, "accuracy": correct/128 if len(native) == 128 else None,
                "observed_complete_pairs": complete_pairs, "both_correct_pairs": both_correct,
                "planned_pair_denominator": 64, "paired_both_correct": both_correct/64 if complete_pairs == 64 else None,
                "readouts_across_probes_are_duplicates_not_independent": True},
            "collisions": collision_summary([r for r in pairs if r["probe"] == probe]),
        }
    primary = result["cohorts"]["unique"]
    count = primary["recovery"]["author_matched_public32"]["success_counts"]["0.001"]
    available = primary["recorded"] == 128 and primary["completed"] == 128
    result["primary_numerical_target"] = {
        "relative_frobenius_threshold": .001, "required_successes": 122,
        "planned_denominator": 128, "successes": count, "success_fraction_planned": count/128,
        "met": count >= 122 if available else None, "available": available,
    }
    eligible = [r for r in rows if r.get("validation", {}).get("f64_reference_eligible") is True]
    f64_failures = [r["record_id"] for r in eligible if r["validation"].get("f64_reference_recovery_valid") is not True]
    unknown = [r["record_id"] for r in rows if r["probe"] == "unique" and
               "f64_reference_eligible" not in r.get("validation", {})]
    first = [r for r in rows if r["probe"] == "repeated" and
             isinstance(r.get("validation", {}).get("first_step_contract"), bool)]
    public = [r for r in rows if isinstance(r.get("validation", {}).get("public32_contract"), bool)]
    expected_pairs = {f"F{f:03d}/{p}" for f in range(64) for p in PROBES}
    pair_counts = Counter(r["pair_id"] for r in pairs)
    exact_ids = {r["pair_id"] for r in pairs if "exact_source_comparison" in r}
    missing_ids, extra_ids = sorted(expected_pairs-exact_ids), sorted(set(pair_counts)-expected_pairs)
    duplicates = sorted(p for p, n in pair_counts.items() if n > 1)
    exact_failures = [r["pair_id"] for r in pairs if "exact_source_comparison_error" in r]
    missing_primary = 128-sum(r["probe"] == "unique" for r in rows)
    checks = {
        "all_256_records_completed": len(rows) == 256 and all(r["status"] == "completed" for r in rows),
        "f64_reference": {"relative_threshold": 1e-8, "eligible_count": len(eligible),
            "failed_ids": f64_failures, "eligibility_unknown_ids": unknown,
            "ineligible_count": sum(r["probe"] == "unique" and r.get("validation", {}).get("f64_reference_eligible") is False for r in rows),
            "missing_primary_records": missing_primary,
            "passed": bool(eligible) and not f64_failures and not unknown and missing_primary == 0},
        "first_state_and_logits": {"expected_checks": 128, "observed": len(first),
            "failed_ids": [r["record_id"] for r in first if not r["validation"]["first_step_contract"]],
            "passed": len(first) == 128 and all(r["validation"]["first_step_contract"] for r in first)},
        "public32_contract": {"expected_checks": 256, "observed": len(public),
            "failed_ids": [r["record_id"] for r in public if not r["validation"]["public32_contract"]],
            "passed": len(public) == 256 and all(r["validation"]["public32_contract"] for r in public)},
        "exact_pair_arithmetic": {"expected_pairs": 128, "recorded_pairs": len(pairs),
            "failed_ids": exact_failures, "missing_exact_comparison_ids": missing_ids,
            "extra_pair_ids": extra_ids, "duplicate_pair_ids": duplicates,
            "passed": len(pairs) == 128 and not exact_failures and not missing_ids and not extra_ids and not duplicates},
    }
    result["required_validation"] = checks
    result["required_validation_passed"] = checks["all_256_records_completed"] and all(
        checks[k]["passed"] for k in checks if k != "all_256_records_completed")
    return result


class Audit:
    def __init__(self, directory, expected_execution):
        self.root = Path(directory).resolve()
        self.expected_execution = expected_execution
        self.counts = Counter()
        self.issues = {"data_integrity": [], "report_consistency": []}
        self.notes = []
        self.rows, self.pairs = [], []
        self.array_entries, self.cache = {}, {}
        self.checked_metrics = 0
        self.unverified_partial = []
        self.supplementary_exact_recomputable_ids = []
        self.execution = self.read_json("execution.json", required=True) or {}
        self.results = self.read_json("results.json") or {}
        self.status = self.read_json("status.json") or {}
        self.terminal = self.status.get("status") in {"completed", "completed_with_validation_failure"}
        self.expected_cohort = frozen_cohort()
        self.expected_plan = fixed_plan(self.expected_cohort)
        self.plan_by_id = {r["record_id"]: r for r in self.expected_plan["planned_records"]}
        self.expected_ids = list(self.plan_by_id)
        self.expected_pair_ids = [f"F{f:03d}/{p}" for f in range(64) for p in PROBES]
        self.torn_lines = {}
        self.public = {}
        self.recomputed = None

    def check(self, condition, label, category="data_integrity", detail=None):
        self.counts[category] += 1
        if not bool(condition):
            issue = {"check": label}
            if detail is not None:
                issue["detail"] = str(detail)[:1200]
            self.issues[category].append(issue)
        return bool(condition)

    def note(self, message):
        if message not in self.notes:
            self.notes.append(message)

    def path(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("unsafe artifact path: " + str(relative))
        result = self.root / path
        if result.is_symlink() or any(p.is_symlink() for p in result.parents if p != self.root.parent):
            raise ValueError("symlink in artifact path: " + str(relative))
        if not result.resolve().is_relative_to(self.root):
            raise ValueError("escaped artifact path")
        return result

    def read_json(self, relative, required=False):
        path = self.path(relative)
        if not path.exists():
            if required:
                self.check(False, "missing_json:" + relative)
            else:
                self.note("Unavailable retained document: " + relative)
            return None
        try:
            return json.loads(path.read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonstandard JSON number " + value)))
        except (ValueError, UnicodeError, OSError) as exc:
            self.check(False, "json:" + relative, detail=exc)
            return None

    def read_jsonl(self, relative):
        path = self.path(relative)
        if not path.exists():
            self.check(not self.terminal, "missing_jsonl:" + relative)
            self.note("Unavailable observation stream: " + relative)
            return []
        lines, rows = path.read_bytes().splitlines(keepends=True), []
        for index, line in enumerate(lines, 1):
            try:
                value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonstandard JSON number ' + value)))
                if not isinstance(value, dict):
                    raise ValueError("JSONL row must be an object")
                rows.append(value)
            except (ValueError, UnicodeError) as exc:
                partial_tail = not self.terminal and index == len(lines) and not line.endswith(b"\n")
                if partial_tail:
                    self.torn_lines[relative] = {"line": index, "bytes": len(line), "sha256": sha(line)}
                    self.note("Unrepaired trailing partial line in " + relative)
                else:
                    self.check(False, "jsonl:" + relative + ":" + str(index), detail=exc)
        return rows

    def compare(self, observed, expected, label, category="report_consistency"):
        """Expected-key subset comparison; stored prose is outside this audit."""
        if isinstance(expected, dict):
            if not self.check(isinstance(observed, dict), label + ":object", category):
                return
            for key, value in expected.items():
                if self.check(key in observed, label + ":" + key + ":present", category):
                    self.compare(observed[key], value, label + "." + key, category)
        elif isinstance(expected, (list, tuple)):
            if label.endswith(".singular_values") and isinstance(observed, list):
                self.matrix_close(observed, expected, label)
                return
            if not self.check(isinstance(observed, list) and len(observed) == len(expected),
                              label + ":list", category):
                return
            for i, value in enumerate(expected):
                self.compare(observed[i], value, f"{label}[{i}]", category)
        elif isinstance(expected, float):
            self.check(finite_number(observed) and math.isclose(
                float(observed), expected, rel_tol=SCALAR_RTOL, abs_tol=SCALAR_ATOL),
                label, category, (observed, expected))
        else:
            self.check(type(observed) is type(expected) and observed == expected,
                       label, category, (observed, expected))

    def matrix_close(self, observed, expected, label):
        a, b = np.asarray(observed), np.asarray(expected)
        same = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        return self.check(same and bool(np.allclose(a, b, rtol=MATRIX_RTOL, atol=MATRIX_ATOL)),
                          label, "report_consistency")

    def inventory(self):
        actual = file_info(self.path("execution.json"))
        self.execution_info = {**actual, "external_digest_supplied": True,
            "source_commit": self.execution.get("github_commit"),
            "run_id": str(self.execution.get("github_run_id", ""))}
        self.check(bool(re.fullmatch(r"[0-9a-f]{64}", self.expected_execution or "")),
                   "external_execution_digest_format")
        self.check(actual["sha256"] == self.expected_execution, "external_execution_digest_match")
        self.compare(self.execution, {
            "frozen_design_commit": DESIGN_COMMIT, "design_sha256": DESIGN_SHA,
            "github_run_attempt": 1, "github_run_number": 1, "budget_seconds": 600,
            "run_kind": "preregistered_contextual_information_v1", "empirical_retries": 0,
            "parent_model_imports_or_inference": 0,
        }, "execution")
        inventory = self.execution.get("retained_files", {})
        self.check(isinstance(inventory, dict), "inventory_object")
        if not isinstance(inventory, dict):
            inventory = {}
        self.check("execution.json" not in inventory, "inventory_excludes_self")
        for name, meta in inventory.items():
            path = self.path(name)
            if self.check(path.is_file(), "inventory_file:" + name):
                self.compare(file_info(path), meta, "inventory:" + name, "data_integrity")
        actual_files = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}
        self.check(actual_files == set(inventory) | {"execution.json"}, "complete_inventory",
                   detail=sorted(actual_files ^ (set(inventory) | {"execution.json"})))
        receipts = self.execution.get("compression_receipts", {})
        for name, receipt in receipts.items():
            for key, suffix in (("raw", ""), ("gzip", ".gz")):
                path = self.path(name + suffix)
                if self.check(path.is_file(), "compression_present:" + name + suffix):
                    self.compare(file_info(path), receipt[key], "compression:" + name + ":" + key, "data_integrity")
            self.check(receipt.get("round_trip_verified") is True, "compression_recorded_verified:" + name)
        # Present contradictory gzip is always invalid, even if a kill prevented a receipt.
        for path in self.root.glob("*.jsonl"):
            compressed = self.path(path.name + ".gz")
            if compressed.exists():
                try:
                    self.check(gzip.decompress(compressed.read_bytes()) == path.read_bytes(),
                               "gzip_roundtrip:" + path.name)
                except (OSError, EOFError, ValueError) as exc:
                    self.check(False, "gzip_roundtrip:" + path.name, detail=exc)
            elif self.terminal:
                self.check(False, "terminal_missing_gzip:" + path.name)
            else:
                self.note("Incomplete compression is preserved: " + path.name)
            if path.name not in receipts:
                self.check(not self.terminal, "terminal_missing_compression_receipt:" + path.name)
                self.note("No compression receipt for partial stream: " + path.name)
        self.inventory_totals = {"files_excluding_execution": len(inventory),
                                 "bytes_excluding_execution": sum(m.get("bytes", 0) for m in inventory.values())}

    def provenance(self):
        source = self.read_json("source_manifest.json", self.terminal)
        launch = self.read_json("launch_source_manifest.json", True)
        launcher = self.read_json("launcher.json", True)
        identity = {k: self.execution.get(k) for k in
                    ("github_commit", "github_run_id", "frozen_design_commit", "design_sha256")}
        for name, document in (("source_manifest", source), ("launch_source_manifest", launch), ("launcher", launcher)):
            if document:
                self.compare(document, identity, name)
        if source and launch:
            self.compare(source.get("files", {}), launch.get("files", {}), "source_files")
            self.check(set(source.get("files", {})) == SOURCE_PATHS, "frozen_source_path_coverage")
            for path, digest in {
                "docs/contextual_information_design.md": DESIGN_SHA,
                "docs/statebridge_information_recovery.md": THEORY_SHA,
                "latent_bridge/alignment_core.py": CORE_SHA,
                "third_party/statebridge/state_bridge.py": AUTHOR_SHA,
            }.items():
                self.check(source.get("files", {}).get(path, {}).get("sha256") == digest,
                           "frozen_source_digest:" + path)
        model = self.read_json("model_manifest.json", self.terminal)
        if model:
            self.compare(model, {"id": MODEL, "revision": MODEL_REVISION,
                                "hub_credentials_used": False, "remote_code_allowed": False}, "model")
            if self.terminal:
                self.check(model.get("status") == "verified", "terminal_model_verified")
            for name, meta in model.get("files", {}).items():
                if self.check(name in MODEL_FILES, "model_file_name:" + name):
                    size, digest = MODEL_FILES[name]
                    self.compare(meta, {"bytes": size, "sha256": digest}, "model:" + name)
            if self.terminal:
                self.check(set(model.get("files", {})) == set(MODEL_FILES), "model_file_coverage")
        runtime = self.read_json("runtime_manifest.json", self.terminal)
        if runtime:
            self.compare(runtime, {"python": "3.11.18",
                "packages": {"torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
                             "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2"},
                "torch_threads": 1, "torch_interop_threads": 1, "deterministic_algorithms": True}, "runtime")
            self.check(all(v == "1" for v in runtime.get("thread_environment", {}).values())
                       and len(runtime.get("thread_environment", {})) == 5, "single_thread_environment")
        if self.results.get("provenance"):
            self.compare(self.results["provenance"], identity, "result_provenance")

    def load_bundles(self):
        manifests = self.read_jsonl("array_manifest.jsonl")
        for entry in manifests:
            name = entry.get("bundle")
            if not self.check(isinstance(name, str) and name not in self.array_entries,
                              "unique_bundle_manifest:" + str(name)):
                continue
            path = self.path(name)
            if not self.check(path.is_file(), "bundle_exists:" + name):
                continue
            self.compare(file_info(path), {"sha256": entry["sha256"], "bytes": entry["bytes"]},
                         "bundle:" + name, "data_integrity")
            with np.load(path, allow_pickle=False) as archive:
                self.check(set(archive.files) == set(entry["arrays"]), "bundle_member_coverage:" + name)
                for member in archive.files:
                    array = archive[member]
                    payload = raw(array)
                    actual = {"dtype": array.dtype.str, "shape": list(array.shape),
                              "raw_bytes": len(payload), "raw_sha256": sha(payload)}
                    self.compare(entry["arrays"].get(member), actual,
                                 "array:" + name + ":" + member, "data_integrity")
            self.array_entries[name] = entry
        known = set(self.array_entries)
        extras = {str(p.relative_to(self.root)) for p in self.root.rglob("*.npz")} - known
        self.check(not extras or not self.terminal, "terminal_unindexed_npz", detail=sorted(extras))
        if extras:
            self.note("Partial unindexed NPZ evidence preserved without repair: " + ", ".join(sorted(extras)))

    def array(self, ref):
        if not isinstance(ref, dict) or not {"bundle", "member"}.issubset(ref):
            self.check(False, "array_ref_shape")
            return None
        name, member = ref["bundle"], ref["member"]
        entry = self.array_entries.get(name)
        if not self.check(entry is not None and member in entry.get("arrays", {}), "array_ref_resolves:" + str(name) + ":" + str(member)):
            return None
        self.compare(ref, {"bundle": name, "member": member, **entry["arrays"][member]},
                     "array_ref:" + name + ":" + member, "data_integrity")
        if name not in self.cache:
            with np.load(self.path(name), allow_pickle=False) as archive:
                value = {key: archive[key] for key in archive.files}
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[name] = value
        return self.cache[name][member]

    def row_array(self, row, member):
        ref = row.get("arrays", {}).get(member)
        return self.array(ref) if ref else None

    def tokenization_and_public(self):
        cohort = self.read_json("cohort.json", self.terminal)
        plan = self.read_json("plan.json", self.terminal)
        if cohort:
            self.compare(cohort, self.expected_cohort, "fixed_cohort")
        if plan:
            self.compare(plan, self.expected_plan, "fixed_plan")
        self.tokenization = self.read_json("tokenization.json", self.terminal) or {}
        token = self.tokenization
        manifest = self.read_json("public_manifest.json", self.terminal) or {}
        if not token or not manifest:
            return
        table_ref = manifest.get("table", {}).get("vocabulary_f32")
        table = self.array(table_ref)
        if table is None:
            return
        self.check(table.ndim == 2 and table.shape[1] == 128 and table.dtype == np.dtype("float32")
                   and np.isfinite(table).all(), "vocabulary_shape_dtype")
        self.vocab = table
        table64 = table.astype(np.float64)
        norms = np.linalg.norm(table64, axis=1)
        self.norms = norms
        self.matrix_close(self.array(manifest["table"]["vocabulary_norms_f64"]), norms, "public64_vocab_norms")
        all_tokens = []
        probe_ids = token.get("probe_ids", {})
        unique, repeated = probe_ids.get("unique", []), probe_ids.get("repeated", [])
        colors = token.get("color_ids", [])
        self.check(len(unique) == 8 and len(set(unique)) == 8, "unique_probe_ids")
        self.check(repeated == [unique[0]] * 8 if unique else False, "repeated_probe_ids")
        self.check(len(colors) == 8 and len(set(colors)) == 8, "color_token_ids")
        expected_prompt_keys = {f"F{f:03d}/{w}" for f in range(64) for w in WORLDS}
        prompts = token.get("prompt_ids", {})
        self.check(set(prompts) == expected_prompt_keys, "prompt_token_coverage")
        for key, ids in prompts.items():
            self.check(isinstance(ids, list) and 0 < len(ids) <= 256, "prompt_length:" + key)
            self.check(token.get("prompt_lengths", {}).get(key) == len(ids), "stored_prompt_length:" + key)
            all_tokens.extend(ids)
        all_tokens.extend(unique + repeated + colors)
        self.check(all(type(x) is int and 0 <= x < len(table) for x in all_tokens), "token_ranges")
        expected_pairs = []
        for f in range(64):
            name = f"F{f:03d}"
            a, b = (prompts.get(name + "/" + w, []) for w in WORLDS)
            same_length, same_bag = len(a) == len(b), Counter(a) == Counter(b)
            self.check(bool(a) and same_length and same_bag, "paired_token_bag:" + name)
            expected_pairs.append({"family_id": name, "same_length": same_length, "same_token_multiset": same_bag})
        self.compare(token, {"schema_version": 1, "paired_preflight": expected_pairs, "passed": True}, "tokenization")
        self.check(manifest.get("prepared_before_private_extraction") is True, "public_preparation_recorded")
        for probe in PROBES:
            group = manifest["probes"][probe]
            arrays = {name: self.array(ref) for name, ref in group["arrays"].items()}
            self.check(np.array_equal(arrays["probe_ids"], probe_ids[probe]), "public_probe_ids:" + probe)
            self.public[probe] = {"arrays": arrays, "metadata": group}
            for precision in ("public32", "public64"):
                meta = group[precision]
                dtype = np.float32 if precision == "public32" else np.float64
                expected_ref = table[np.asarray(probe_ids[probe])].astype(dtype)
                self.check(bits_equal(arrays[precision + "_reference"], expected_ref), "public_reference:" + probe + ":" + precision)
                params = {
                    "K": 8, "d": 128, "vocabulary_size": len(table), "probe_ids": probe_ids[probe],
                    "convention": precision, "lam": float(dtype(.001)),
                    "anchor_gain": float(dtype(.3)), "complement": float(dtype(1-.3)),
                    "covariance_floor": float(dtype(1e-6)), "row_floor": float(dtype(1e-6)),
                    "cosine_floor": float(dtype(1e-12)),
                    "vocabulary_zero_rows": int(np.count_nonzero(norms == 0)),
                    "vocabulary_nonzero_below_floor_rows": int(np.count_nonzero((norms > 0) & (norms < float(dtype(1e-12))))),
                    "anchor_geometry_supported": not bool(np.any((norms > 0) & (norms < float(dtype(1e-12))))),
                }
                self.compare(meta, params, "public_metadata:" + probe + ":" + precision)
                expected_r = expected_ref.astype(np.float64)
                self.compare(meta["reference"], rank_info(expected_r), "public_reference_rank:" + probe + ":" + precision)
                self.compare(meta["centered_reference"], rank_info(expected_r-expected_r.mean(axis=0)),
                             "public_centered_reference_rank:" + probe + ":" + precision)
                if precision == "public64":
                    self.matrix_close(arrays["public64_mean"], expected_r.mean(axis=0).reshape(1, -1), "public64_mean:" + probe)
                    self.compare(meta, {"target_norm": float(norms.mean())}, "public64_target_norm:" + probe)
        stages = [entry.get("stage") for entry in self.array_entries.values()]
        first_private = next((i for i, stage in enumerate(stages) if stage != "public_before_extraction"), len(stages))
        self.check(first_private >= 3 and all(x == "public_before_extraction" for x in stages[:3]),
                   "public_manifest_precedes_private_bundles")


    def metric(self, row, label, observed, actual, reference):
        if actual is None or reference is None:
            if row.get("status") == "completed":
                self.check(False, "completed_metric_missing_arrays:" + row["record_id"] + ":" + label)
            else:
                self.unverified_partial.append(row["record_id"] + ":" + label)
            return observed
        recomputed = errors(actual, reference)
        self.compare(observed, recomputed, "metric:" + row["record_id"] + ":" + label)
        self.checked_metrics += 1
        return recomputed

    def records(self):
        original = self.read_jsonl("records.jsonl")
        ids = [r.get("record_id") for r in original]
        self.check(ids == self.expected_ids[:len(ids)], "record_order_and_unique_prefix")
        self.check(len(ids) <= 256, "record_count_at_most_256")
        self.original_rows = original
        first = {}
        for row in original:
            identifier = row.get("record_id")
            if identifier not in self.plan_by_id:
                self.check(False, "unknown_record:" + str(identifier))
                continue
            planned = self.plan_by_id[identifier]
            self.compare(row, planned, "record_identity:" + identifier)
            self.check(row.get("status") in {"completed", "error"}, "record_status:" + identifier)
            if row.get("status") == "error":
                self.check(row is original[-1], "stop_after_operational_error:" + identifier)
                self.check(bool(row.get("operational_errors")), "error_has_reason:" + identifier)
            else:
                self.check(not row.get("operational_errors"), "completed_without_operational_error:" + identifier)
            cleaned = copy.deepcopy(row)
            arr = {}
            source_names = {"source", "first_color_logits", "color_probabilities",
                            "forced_logits", "forced_log_probabilities"}
            prefix = "bundles/" + identifier.replace("/", "_")
            for name, ref in row.get("arrays", {}).items():
                self.check(ref.get("member") == name, "row_member_key:" + identifier + ":" + name)
                expected_bundle = prefix + ("_source.npz" if name in source_names else ".npz")
                self.check(ref.get("bundle") == expected_bundle, "row_bundle_identity:" + identifier + ":" + name)
                arr[name] = self.array(ref)
            complete = row["status"] == "completed"
            required = source_names | {"source_centered_f64", "source_reduced_gram_f64", "source_full_gram_f64",
                                      "direct_gram_f32", "author_final_f32", "reference_final_f64",
                                      "reference_final_cast_f32", "reference_pre_f64", "reference_anchor_ids"}
            if complete:
                self.check(required.issubset(arr) and all(arr[k] is not None for k in required),
                           "completed_array_coverage:" + identifier)
            source = arr.get("source")
            extraction = row.get("extraction")
            if source is not None:
                self.check(source.shape == (8, 128) and source.dtype == np.dtype("float32")
                           and np.isfinite(source).all(), "source_contract:" + identifier)
                key = row["family_id"] + "/" + row["world"]
                logits = arr.get("first_color_logits")
                if logits is not None:
                    self.check(logits.shape == (8,) and logits.dtype == np.dtype("float32")
                               and np.isfinite(logits).all(), "first_logits:" + identifier)
                    if row["probe"] == "unique":
                        first[key] = (source[0].copy(), logits.copy())
                        first_check = "baseline_recorded"
                    else:
                        first_check = key in first and bits_equal(source[0], first[key][0]) and bits_equal(logits, first[key][1])
                    self.compare(row.get("validation", {}).get("first_step_contract"), first_check, "first_step:" + identifier)
                    cleaned.setdefault("validation", {})["first_step_contract"] = first_check
                if extraction:
                    self.extraction(row, arr)
                    if logits is not None:
                        cleaned["extraction"]["native_correct"] = int(np.argmax(logits)) == row["answer_index"]
                    lp = arr.get("forced_log_probabilities")
                    if lp is not None:
                        cleaned["extraction"]["forced_log_probability_mean"] = sum(float(x) for x in lp) / 8
                sc, gamma, g, reduced_source = centered_grams(source)
                for name, computed in (("source_centered_f64", sc), ("source_full_gram_f64", gamma),
                                       ("source_reduced_gram_f64", g)):
                    if arr.get(name) is not None:
                        self.matrix_close(arr[name], computed, "source_truth:" + identifier + ":" + name)
                source_geometry = row.get("source_geometry")
                if source_geometry:
                    expected_geometry = {name: rank_info(value) for name, value in
                        (("source", source), ("centered", sc), ("reduced_source", reduced_source))}
                    expected_geometry["source_mean"] = source.astype(np.float64).mean(axis=0).tolist()
                    # Use the already verified retained truth to compare dependent metrics.
                    truth = arr.get("source_reduced_gram_f64")
                    if truth is not None:
                        eigenvalues = np.linalg.eigvalsh(truth)
                        expected_geometry.update(reduced_eigenvalues=eigenvalues.tolist(),
                                                 reduced_gram_norm=float(np.linalg.norm(truth)),
                                                 full_gram_norm=float(np.linalg.norm(arr["source_full_gram_f64"])))
                        for precision in ("public32", "public64"):
                            lam = float(np.float32(.001)) if precision == "public32" else .001
                            expected_geometry["inverse_condition_factors_" + precision] = (1+eigenvalues/(8*lam)).tolist()
                    self.compare(source_geometry, expected_geometry, "source_geometry:" + identifier)
                    cleaned["source_geometry"].update(expected_geometry)
                    group = self.public.get(row["probe"])
                    if group:
                        r = group["arrays"]["public64_reference"]
                        eligible = (row["probe"] == "unique" and expected_geometry["centered"]["rank"] == 7
                                    and rank_info(r)["rank"] == 8
                                    and not np.any((self.norms > 0) & (self.norms < 1e-12)))
                        self.compare(row.get("validation", {}).get("f64_reference_eligible"), bool(eligible), "eligible:" + identifier)
                        cleaned.setdefault("validation", {})["f64_reference_eligible"] = bool(eligible)
            truth = arr.get("source_reduced_gram_f64")
            if "direct_gram_error" in row:
                cleaned["direct_gram_error"] = self.metric(row, "direct_gram", row["direct_gram_error"],
                                                          arr.get("direct_gram_f32"), truth)
                if truth is not None and arr.get("direct_gram_f32") is not None:
                    self.check(bits_equal(arr["direct_gram_f32"], truth.astype(np.float32)), "direct_gram_rounding:" + identifier)
            author_arrays = {}
            author = row.get("author")
            if author:
                for name, ref in author.get("local_refs", {}).items():
                    if "pending_member" in ref:
                        self.check(not complete, "unresolved_complete_author_local:" + identifier + ":" + name)
                        self.unverified_partial.append(identifier + ":author_local:" + name)
                    else:
                        shared = {"mu_E": "mean", "Cov_E": "covariance", "sqrt_Cov_E": "root", "E_w": "target_white"}
                        expected_ref = row.get("arrays", {}).get("author_" + name)
                        if name in shared and row["probe"] in self.public:
                            public_ref = self.public[row["probe"]]["metadata"]["arrays"]["public32_" + shared[name]]
                            self.check(ref == expected_ref or ref == public_ref, "author_local_reference_identity:" + identifier + ":" + name)
                        else:
                            self.check(ref == expected_ref, "author_local_reference_identity:" + identifier + ":" + name)
                        author_arrays[name] = self.array(ref)
                expected_locals = {"Hc", "H_w", "E_w", "mu_H", "mu_E", "Cov_H", "Cov_E", "R",
                                   "sqrt_Cov_E", "aligned_flat", "flat", "top_ids"}
                self.compare(author.get("captured_locals"), sorted(expected_locals), "author_capture_names:" + identifier)
                checks = {}
                if row["probe"] in self.public:
                    for name, public_name in {"mu_E": "mean", "Cov_E": "covariance",
                                             "sqrt_Cov_E": "root", "E_w": "target_white"}.items():
                        value = author_arrays.get(name)
                        if value is not None:
                            public_value = self.public[row["probe"]]["arrays"]["public32_" + public_name]
                            checks[name] = bits_equal(value, public_value)
                    if len(checks) == 4:
                        self.compare(author.get("public_contract_checks"), checks, "author_public_checks:" + identifier)
                        passed = all(checks.values())
                        self.compare(row.get("validation", {}).get("public32_contract"), passed, "public32_contract:" + identifier)
                        cleaned.setdefault("validation", {})["public32_contract"] = passed
                final = arr.get("author_final_f32")
                if final is not None:
                    self.check(final.shape == (8, 128) and final.dtype == np.dtype("float32"), "author_final_contract:" + identifier)
                    self.compare(author, {"output_dtype": "torch.float32", "actual_return_bytes": True,
                                          "output_finite": bool(np.isfinite(final).all())}, "author_output:" + identifier)
                for label, name in (("pre_rank", "aligned_flat"), ("calibrated_rank", "flat")):
                    if author_arrays.get(name) is not None:
                        self.compare(author.get(label), rank_info(author_arrays[name]), "author_rank:" + identifier + ":" + label)
                if author_arrays.get("aligned_flat") is not None and row["probe"] in self.public:
                    expected_mean = self.public[row["probe"]]["arrays"]["public32_mean"].astype(np.float64).reshape(-1)
                    self.metric(row, "author_pre_mean", author.get("pre_mean_residual"),
                                author_arrays["aligned_flat"].astype(np.float64).mean(axis=0), expected_mean)
            if arr.get("reference_final_f64") is not None and arr.get("reference_final_cast_f32") is not None:
                self.check(bits_equal(arr["reference_final_cast_f32"], arr["reference_final_f64"].astype(np.float32)),
                           "reference_output_only_cast:" + identifier)
            vector_sources = {
                "author_vs_reference_final": (arr.get("author_final_f32"), arr.get("reference_final_f64")),
                "author_vs_reference_pre": (author_arrays.get("aligned_flat"), arr.get("reference_pre_f64")),
                "reference_cast_vs_reference_final": (arr.get("reference_final_cast_f32"), arr.get("reference_final_f64")),
            }
            for name, observed in row.get("vector_differences", {}).items():
                if name in vector_sources:
                    self.metric(row, name, observed, *vector_sources[name])
            self.check(set(row.get("recovery", {})).issubset(ROUTES), "recovery_route_names:" + identifier)
            if complete:
                self.check(set(row.get("recovery", {})) == set(ROUTES), "complete_recovery_routes:" + identifier)
            for name, recovery in row.get("recovery", {}).items():
                self.check(recovery.get("status") in {"estimated", "rejected", "unsupported"}, "inverse_status:" + identifier + ":" + name)
                self.check(recovery.get("accuracy_comparison_after_inverse_return") is True, "evaluation_scope_flag:" + identifier + ":" + name)
                estimate = arr.get("inverse_" + name + "_gram")
                if recovery["status"] == "estimated":
                    self.check(recovery.get("reason") is None, "estimated_no_rejection_reason:" + identifier + ":" + name)
                    if complete:
                        self.check(estimate is not None and estimate.shape == (7, 7) and np.isfinite(estimate).all(),
                                   "estimated_gram_present:" + identifier + ":" + name)
                    cleaned["recovery"][name]["gram_error"] = self.metric(
                        row, name, recovery.get("gram_error"), estimate, truth)
                else:
                    self.check(estimate is None and recovery.get("gram_error") is None
                               and isinstance(recovery.get("reason"), str), "rejection_no_estimate:" + identifier + ":" + name)
                derived = arr.get("inverse_" + name + "_anchor_ids")
                true_ids = author_arrays.get("top_ids") if name.startswith("author_") else arr.get("reference_anchor_ids")
                if derived is not None and true_ids is not None:
                    disagreements = int(np.count_nonzero(derived.reshape(-1) != true_ids.reshape(-1)))
                    self.compare(recovery.get("actual_anchor_disagreements"), disagreements, "anchor_disagreement:" + identifier + ":" + name)
                # Check the retained domain evidence, without rerunning the inverse.
                eigenvalues = arr.get("inverse_" + name + "_white_eigenvalues")
                if estimate is not None and eigenvalues is not None:
                    self.check(np.isfinite(eigenvalues).all() and np.all(eigenvalues >= 0)
                               and np.all(eigenvalues < 8), "estimated_spectrum_in_domain:" + identifier + ":" + name)
            if cleaned.get("validation", {}).get("f64_reference_eligible"):
                valid = finite_success(cleaned.get("recovery", {}).get("reference_f64", {}).get("gram_error"), 1e-8)
                self.compare(row.get("validation", {}).get("f64_reference_recovery_valid"), valid, "f64_validation:" + identifier)
                cleaned["validation"]["f64_reference_recovery_valid"] = valid
            for name, value in row.get("phase_seconds", {}).items():
                self.check(finite_number(value) and value >= 0, "recorded_phase_nonnegative:" + identifier + ":" + name)
            self.rows.append(cleaned)
        self.check(not original or original[-1].get("status") != "error" or not self.terminal,
                   "operational_error_is_not_terminal_completion")

    def extraction(self, row, arrays):
        name, item = row["record_id"], row["extraction"]
        logits, probabilities = arrays.get("first_color_logits"), arrays.get("color_probabilities")
        if logits is not None:
            prediction = int(np.argmax(logits))
            self.compare(item, {"native_prediction_index": prediction,
                                "native_correct": prediction == row["answer_index"],
                                "source_shape": [8, 128]}, "native_readout:" + name)
            if probabilities is not None:
                p = np.exp(logits.astype(np.float64) - np.max(logits))
                p /= p.sum()
                self.matrix_close(probabilities, p, "restricted_probabilities:" + name)
        self.compare(item, {"model_forward_calls": 8, "sampling": False,
                            "cache_reused_across_sequences": False,
                            "hook": "gpt_neox.layers[-1], output before final_layer_norm"}, "extraction_contract:" + name)
        prompt = self.tokenization.get("prompt_ids", {}).get(row["family_id"] + "/" + row["world"])
        ids = self.tokenization.get("probe_ids", {}).get(row["probe"])
        if prompt is not None and ids is not None:
            steps = []
            for step in range(8):
                length = len(prompt) + step
                steps.append({"step": step, "consumed_ids": prompt if step == 0 else [ids[step-1]],
                              "position_ids": list(range(len(prompt))) if step == 0 else [length-1],
                              "attention_mask_length": length, "predicts_token_id": ids[step],
                              "has_input_cache": step > 0})
            self.compare(item.get("steps"), steps, "extraction_steps:" + name)
        hashes = item.get("full_logits_sha256", [])
        self.check(len(hashes) == 8 and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in hashes),
                   "logit_digest_format_only:" + name)
        probabilities = arrays.get("forced_log_probabilities")
        if probabilities is not None:
            self.check(probabilities.shape == (8,) and probabilities.dtype == np.dtype("float64")
                       and np.isfinite(probabilities).all() and np.all(probabilities <= 0),
                       "forced_log_probability_shape_range:" + name)
            total = sum(float(x) for x in probabilities)
            self.compare(item, {"forced_log_probability_sum": total, "forced_log_probability_mean": total/8},
                         "forced_log_probability_aggregation:" + name)
        if arrays.get("forced_logits") is not None:
            a = arrays["forced_logits"]
            self.check(a.shape == (8,) and a.dtype == np.dtype("float32") and np.isfinite(a).all(),
                       "forced_logits_shape:" + name)

    def collisions(self):
        original = self.read_jsonl("collisions.jsonl")
        self.original_pairs = original
        identifiers = [r.get("pair_id") for r in original]
        self.check(identifiers == self.expected_pair_ids[:len(identifiers)], "collision_order_and_unique_prefix")
        rows = {r["record_id"]: r for r in self.rows}
        for row in original:
            pair_id = row.get("pair_id")
            if not self.check(pair_id in self.expected_pair_ids, "known_pair:" + str(pair_id)):
                continue
            family_id, probe = pair_id.split("/")
            family = self.expected_cohort["families"][int(family_id[1:])]
            self.compare(row, {"family_id": family_id, "probe": probe, "question": family["question"],
                              "answers": [family["worlds"][w]["answer_index"] for w in WORLDS]}, "pair_identity:" + pair_id)
            endpoints = [rows.get(f"{family_id}/{world}/{probe}") for world in WORLDS]
            self.check(endpoints[1] is not None, "pair_follows_swapped_record:" + pair_id)
            arrays = [{name: self.row_array(r, name) for name in (*CHANNELS.values(), "source_reduced_gram_f64")}
                      if r else {} for r in endpoints]
            computed = {k: copy.deepcopy(v) for k, v in row.items()
                        if k not in {"channels", "exact_source_comparison", "exact_source_comparison_error",
                                     "stored_reduced_gram_f64_byte_equal", "reduced_gram_difference"}}
            computed["channels"] = {}
            for channel, member in CHANNELS.items():
                a, b = arrays[0].get(member), arrays[1].get(member)
                valid = (a is not None and b is not None and a.shape == b.shape
                         and a.dtype == np.dtype("float32") and b.dtype == np.dtype("float32")
                         and np.isfinite(a).all() and np.isfinite(b).all())
                content = {"valid_pair": bool(valid), "byte_collision": bits_equal(a, b) if valid else None}
                if valid:
                    content["difference"] = errors(a, b)
                self.compare(row.get("channels", {}).get(channel), content, "collision:" + pair_id + ":" + channel)
                computed["channels"][channel] = content
            public = self.public.get(probe, {}).get("arrays", {}).get("public32_reference")
            token = {"valid_pair": True, "byte_collision": True}
            if public is not None:
                token["public_reference_raw_sha256"] = sha(raw(public))
            self.compare(row.get("channels", {}).get("token_only_f32"), token, "token_floor:" + pair_id)
            computed["channels"]["token_only_f32"] = token
            a, b = arrays[0].get("source"), arrays[1].get("source")
            if a is not None and b is not None:
                try:
                    exact = exact_pair(a, b)
                    if "exact_source_comparison" in row:
                        self.compare(row["exact_source_comparison"], exact, "exact_dyadic:" + pair_id)
                        self.check("exact_source_comparison_error" not in row, "exact_dyadic_noncontradictory_status:" + pair_id)
                        computed["exact_source_comparison"] = exact
                    elif "exact_source_comparison_error" in row:
                        error = row["exact_source_comparison_error"]
                        self.check(isinstance(error, dict) and isinstance(error.get("type"), str)
                                   and isinstance(error.get("message"), str),
                                   "recorded_exact_error_shape:" + pair_id)
                        computed["exact_source_comparison_error"] = copy.deepcopy(error)
                        self.supplementary_exact_recomputable_ids.append(pair_id)
                        self.note("Some original exact-comparison failures are now independently computable; their original failed gate and absent witness status are preserved.")
                    else:
                        self.check(False, "missing_recorded_exact_attempt:" + pair_id)
                except (ValueError, FloatingPointError) as exc:
                    self.check("exact_source_comparison_error" in row and "exact_source_comparison" not in row,
                               "retained_exact_comparison_failure:" + pair_id, detail=exc)
                    computed["exact_source_comparison_error"] = row.get("exact_source_comparison_error", {})
            else:
                self.check("exact_source_comparison" not in row, "no_unretained_exact_witness:" + pair_id)
            a, b = (x.get("source_reduced_gram_f64") for x in arrays)
            if a is not None and b is not None:
                computed["stored_reduced_gram_f64_byte_equal"] = bits_equal(a, b)
                computed["reduced_gram_difference"] = errors(a, b)
                self.compare(row.get("stored_reduced_gram_f64_byte_equal"), computed["stored_reduced_gram_f64_byte_equal"],
                             "stored_gram_bits:" + pair_id)
                self.compare(row.get("reduced_gram_difference"), computed["reduced_gram_difference"], "stored_gram_difference:" + pair_id)
            actual = computed["channels"]["author_f32"]
            computed["status"] = "compared" if actual["valid_pair"] else "missing_or_invalid"
            exact = computed.get("exact_source_comparison")
            computed.update(preexisting_source_collision=None, alignment_created_collision=None,
                            exact_gram_loss_collision=None)
            if actual["valid_pair"] and exact is not None:
                collision = actual["byte_collision"]
                new = collision and row["answers"][0] != row["answers"][1] and not exact["numeric_values_equal"]
                computed.update(preexisting_source_collision=bool(collision and exact["numeric_values_equal"]),
                                alignment_created_collision=bool(new),
                                exact_gram_loss_collision=bool(new and not exact["gram_equal"]))
            self.compare(row, {k: computed[k] for k in ("status", "preexisting_source_collision",
                                                       "alignment_created_collision", "exact_gram_loss_collision")},
                         "collision_classification:" + pair_id)
            self.pairs.append(computed)


    def summaries(self):
        completed = self.results.get("study_complete") is True
        self.recomputed = independent_summary(self.rows, self.pairs, completed)
        if self.results:
            count = self.results.get("recorded_records")
            self.check(type(count) is int and 0 <= count <= len(self.rows), "snapshot_record_prefix")
            if type(count) is int and 0 <= count <= len(self.rows):
                try:
                    pair_count = sum(self.results["cohorts"][p]["collisions"]["channels"]["token_only_f32"]
                                     ["observed_pair_denominator"] for p in PROBES)
                except (KeyError, TypeError):
                    pair_count = -1
                self.check(type(pair_count) is int and 0 <= pair_count <= len(self.pairs), "snapshot_pair_prefix")
                if type(pair_count) is int and 0 <= pair_count <= len(self.pairs):
                    if self.terminal:
                        self.check(count == len(self.rows) and pair_count == len(self.pairs), "terminal_snapshot_current")
                    elif count != len(self.rows) or pair_count != len(self.pairs):
                        self.note("Partial atomic results snapshot lags preserved raw observations; both populations are reported.")
                    expected = independent_summary(self.rows[:count], self.pairs[:pair_count], completed)
                    self.compare(self.results, expected, "results")
                    self.snapshot_population = {"records": count, "pairs": pair_count}
            if "expected_model_forward_calls" in self.results:
                observed_calls = sum(r.get("extraction", {}).get("model_forward_calls", 0) for r in self.rows)
                self.compare(self.results, {
                    "expected_model_forward_calls": 2048,
                    "retained_complete_extraction_forward_calls": observed_calls,
                    "forward_call_count_is_exact": all("extraction" in r for r in self.rows),
                    "native_model_training": False, "receiver_generation": False,
                    "public_contexts_prepared_before_private_extraction": True,
                    "empirical_retries": 0,
                }, "result_execution_scope")
        if self.terminal:
            self.check(self.recomputed["study_complete"], "terminal_study_complete")
            expected_status = ("completed" if self.recomputed["required_validation_passed"]
                               else "completed_with_validation_failure")
            self.compare(self.status, {
                "status": expected_status, "recorded_records": len(self.rows),
                "completed_records": self.recomputed["completed_records"],
                "required_validation_passed": self.recomputed["required_validation_passed"],
                "primary_numerical_target_met": self.recomputed["primary_numerical_target"]["met"],
            }, "terminal_status")
        accounting = self.read_json("completion_accounting.json", True)
        ids = [r["record_id"] for r in self.original_rows]
        counts = Counter(ids)
        bad_read = bool(self.torn_lines.get("records.jsonl")) or any(
            issue["check"].startswith("jsonl:records.jsonl:") for issue in self.issues["data_integrity"])
        all_slots = (set(ids) == set(self.expected_ids) and len(ids) == 256 and not bad_read)
        all_completed = all_slots and all(r.get("status") == "completed" for r in self.original_rows)
        if accounting and self.path("plan.json").exists():
            self.compare(accounting, {
                "expected_records": 256, "plan_status": "valid", "planned_slots": 256,
                "recorded_slots": len(ids), "unique_recorded_slots": len(counts),
                "status_counts": dict(Counter(r["status"] for r in self.original_rows)),
                "duplicate_ids": sorted(k for k, n in counts.items() if n > 1),
                "unknown_ids": sorted(set(ids)-set(self.expected_ids)),
                "unexecuted_slots": [{"record_id": i, "status": "not_executed",
                    "reason": "no complete retained record after child termination"}
                    for i in self.expected_ids if i not in counts],
                "all_slots_recorded": all_slots, "all_records_completed_successfully": all_completed,
            }, "completion_accounting")
            self.check(bool(accounting.get("read_errors")) == bad_read, "completion_read_error_presence")
        self.compare(self.execution, {"all_slots_recorded": all_slots,
                                     "all_records_completed_successfully": all_completed},
                     "supervisor_completion")
        self.classification = (
            ("complete_study" if self.recomputed["required_validation_passed"] else "complete_study_with_validation_failure")
            if self.recomputed["study_complete"] else "incomplete_study")
        gates = self.recomputed["required_validation"]
        self.required_failed = sum(not (value if type(value) is bool else value["passed"])
                                   for value in gates.values())

    def run(self):
        stages = (
            ("inventory", self.inventory), ("provenance", self.provenance),
            ("retained_arrays", self.load_bundles), ("frozen_inputs_public", self.tokenization_and_public),
            ("records", self.records), ("pairs", self.collisions), ("statistics", self.summaries),
        )
        for name, function in stages:
            try:
                function()
            except Exception as exc:
                self.check(False, "audit_stage_exception:" + name, detail=type(exc).__name__ + ": " + str(exc))
                self.note("Audit stage could not finish: " + name)
        integrity = not any(self.issues.values())
        science = {
            "study_classification": getattr(self, "classification", "incomplete_study"),
            "required_checks_failed": getattr(self, "required_failed", 5),
            "required_checks_failed_definition": "Count of false original mandatory gate families: completion, float64 reference, duplicated first step, public32 identity, exact-pair coverage.",
            "recomputed_summary": self.recomputed,
            "recomputed_summary_scope": "Independent retained-array recomputation where evidence exists. Entries named in partial_metrics_without_retained_inputs are carried original claims, not independently recomputed estimates; they cannot establish a complete scientific result.",
            "supplementary_exact_recomputable_after_original_failure_ids": self.supplementary_exact_recomputable_ids,
            "original_report_primary_target": self.results.get("primary_numerical_target"),
            "original_status": self.status,
            "original_execution_worker_returncode": self.execution.get("worker_returncode"),
            "original_execution_failure": self.execution.get("failure"),
            "original_retention_errors": self.execution.get("retention_errors", []),
            "latest_retained_observation_counts": {"records": len(self.rows), "pairs": len(self.pairs)},
            "summary_snapshot_population": getattr(self, "snapshot_population", None),
            "independently_checked_error_metrics": self.checked_metrics,
            "partial_metrics_without_retained_inputs": self.unverified_partial,
            "no_complete_metric_claim_for_unretained_partial_arrays": True,
            "unrepaired_partial_lines": self.torn_lines,
            "scope": [
                "File inventory, gzip equivalence, all indexed NPZ members and record references are authenticated against the externally supplied execution digest.",
                "The one frozen cohort is reconstructed only for verification; recorded token bags/steps are checked without rerunning a tokenizer.",
                "Source centering, full/reduced Gram matrices, numerical ranks, retained-estimate error metrics, fixed thresholds, native restricted readout, collision membership and finite-codebook ceilings are independently recomputed.",
                "Exact dyadic source comparisons use Python rational values and arbitrary integers, independently of the production IEEE-bit parser.",
                "The audit does not import the worker, inverse, alignment core, author application, Torch or Transformers; it does not replay alignment or the inverse or certify all of their rejection predicates.",
                "Actual model forward passes, hidden-state authenticity and teacher-forced full-vocabulary probabilities are not independently reproduced. Full-logit digests are checked only for format; retained probability aggregation is checked.",
                "Author/public identity and original selected anchor disagreements are checked from retained arrays; unretained private locals and missing model evidence are not reconstructed.",
                "Finite numerical target failure or valid operational incompleteness does not itself cause an integrity failure; missing or contradictory evidence needed for a claimed completed result does.",
                "Source manifests and model download receipts are authenticated original claims, not new source/model downloads by this audit.",
            ],
            "cross_host_recomputation_tolerances": {
                "scalar_rtol": SCALAR_RTOL, "scalar_atol": SCALAR_ATOL,
                "matrix_rtol": MATRIX_RTOL, "matrix_atol": MATRIX_ATOL,
                "exact_domains": "file/raw-array bytes, gzip, integer arithmetic, IDs, fixed counts and collision flags",
                "scientific_thresholds_changed": False,
            },
            "notes": self.notes,
        }
        return {
            "schema_version": 1, "audit_kind": "independent_contextual_information_retained_evidence_v1",
            "audit_status": "completed" if integrity else "completed_with_integrity_issues",
            "artifact_directory": str(self.root),
            "audit_runtime": {"script_sha256": file_info(Path(__file__).resolve())["sha256"],
                              "python": platform.python_version(), "numpy": np.__version__},
            "evidence": {"execution": getattr(self, "execution_info", {
                "external_digest_supplied": bool(self.expected_execution), "sha256": None, "bytes": None,
                "source_commit": self.execution.get("github_commit"), "run_id": self.execution.get("github_run_id")}),
                "inventory_totals": getattr(self, "inventory_totals", None)},
            "data_integrity": {"passed": not self.issues["data_integrity"],
                "checks": self.counts["data_integrity"], "issue_count": len(self.issues["data_integrity"]),
                "issues": self.issues["data_integrity"]},
            "report_consistency": {"passed": not self.issues["report_consistency"],
                "checks": self.counts["report_consistency"], "issue_count": len(self.issues["report_consistency"]),
                "issues": self.issues["report_consistency"]},
            "scientific_validation": science,
            "new_fixture_generations": 0, "model_imports_or_inference": 0,
            "optimization_invocations": 0, "timing_measurements": 0,
        }


def self_test():
    """Tiny fixed algebra/format controls only; no empirical cohort execution."""
    checks = 0

    def require(value, label):
        nonlocal checks
        if not bool(value):
            raise AssertionError(label)
        checks += 1

    a = np.array([[1, -0.0], [-1, 0]], dtype=np.float32)
    b = a.copy()
    b[0, 1] = 0.0
    result = exact_pair(a, b)
    require(result["numeric_values_equal"] and result["gram_equal"] and not result["raw_bits_equal"],
            "signed-zero differs only in raw bits")
    expected_source = [[1 << 23, 0], [-(1 << 23), 0]]
    require(result["common_exponent"] == -23, "normal significand exponent")
    require(result["left_source_integer_sha256"] == integer_hash(expected_source, -23, "source"),
            "explicit independent integer witness")
    a = np.array([[2.0**60, 1], [0, 0]], dtype=np.float32)
    b = np.array([[2.0**60, 2], [0, 0]], dtype=np.float32)
    require(np.array_equal(centered_grams(a)[1], centered_grams(b)[1]), "rounded Gram control")
    require(not exact_pair(a, b)["gram_equal"], "exact integer Gram distinguishes rounded collision")
    tiny = np.array([[np.nextafter(np.float32(0), np.float32(1))], [0]], dtype=np.float32)
    require(exact_pair(tiny, tiny)["common_exponent"] == -149, "subnormal common exponent")
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = x[:, ::-1] + np.float32(8)
    require(exact_pair(x, y)["gram_equal"] and not exact_pair(x, y)["numeric_values_equal"],
            "translation and right permutation preserve centered Gram")
    require(errors(np.array([1e-200]), np.array([0.0]))["relative_frobenius"] is None,
            "underflow does not make unequal zero-reference inputs equal")
    require(errors(np.zeros(2), np.zeros(2))["relative_frobenius"] == 0, "zero equality")
    require(finite_success({"finite": True, "relative_frobenius": .001}, .001), "inclusive fixed threshold")
    require(not finite_success({"finite": True, "relative_frobenius": np.nextafter(.001, math.inf)}, .001),
            "no threshold widening")
    c = ceilings(10, 50)
    require(c["missing_or_invalid"] == 4 and c["full_codebook_ceiling"] is None, "missing pairs stay unknown")
    require(c["full_codebook_ceiling_bounds"] == [1-14/128, 1-10/128], "fixed 64-pair bounds")
    require(ceilings(64, 0)["full_codebook_ceiling"] == .5, "all collisions balanced ceiling")
    require(ceilings(0, 64)["full_codebook_ceiling"] == 1, "all distinct upper bound")
    s = independent_summary([], [], False)
    require(s["primary_numerical_target"]["planned_denominator"] == 128
            and s["primary_numerical_target"]["met"] is None, "empty evidence is incomplete")
    require(s["cohorts"]["unique"]["recovery"]["author_matched_public32"]["statuses"] == {"not_observed": 128},
            "missing inverse observations remain in denominator")
    return {"self_test": "passed", "checks": checks,
            "model_imports_or_inference": 0, "new_fixture_generations": 0,
            "optimization_invocations": 0, "timing_measurements": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--execution-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        if args.artifact_dir is None:
            return 0
    if not args.artifact_dir or not args.execution_sha256 or not args.output:
        parser.error("--artifact-dir, --execution-sha256 and --output are required for artifact audit")
    directory, output = Path(args.artifact_dir).resolve(), Path(args.output).resolve()
    if output == directory or output.is_relative_to(directory):
        parser.error("audit receipt must be outside the original artifact")
    audit = Audit(directory, args.execution_sha256)
    receipt = audit.run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "audit_status": receipt["audit_status"],
        "data_integrity": receipt["data_integrity"]["passed"],
        "report_consistency": receipt["report_consistency"]["passed"],
        "study_classification": receipt["scientific_validation"]["study_classification"],
        "required_checks_failed": receipt["scientific_validation"]["required_checks_failed"],
        "receipt": str(output),
    }, sort_keys=True))
    return 0 if receipt["audit_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
