"""Strict behavioral-reconstruction gate; no model inference or fitting here."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

FROZEN_COMMIT = "fb793ba31d7a10975a9e5c6032b44ecb5eb5774f"
HISTORICAL_MODULES = (
    "latent_bridge/query_conditioned.py", "latent_bridge/query_conditioned_data.py",
    "latent_bridge/counterfactual_runner.py", "latent_bridge/private_counterfactual.py",
    "latent_bridge/private_lookup.py", "latent_bridge/transformers_runner.py",
)
CE_ATOL, CE_RTOL, GATE_ATOL, GATE_RTOL = 1e-5, 1e-5, 1e-6, 1e-5


def close(new, old, atol=CE_ATOL, rtol=CE_RTOL):
    return (isinstance(new, (int, float)) and isinstance(old, (int, float))
            and math.isfinite(new) and math.isfinite(old) and abs(new - old) <= atol + rtol * abs(old))


def prediction_blob(directory):
    directory = Path(directory)
    if (directory / "predictions.jsonl").is_file():
        return (directory / "predictions.jsonl").read_bytes()
    return gzip.decompress((directory / "predictions.jsonl.gz").read_bytes())


def compare_results(reference, reconstructed, old_rows, new_rows):
    """Pure comparison used by tests and the file-backed production gate."""
    failures, counts = [], {"prediction_rows": 0, "validation_records": 0, "selected_gates": 0}

    def require(condition, name):
        if not condition:
            failures.append(name)

    for key in ("config", "candidate_colors", "candidate_token_ids", "versions", "model_files", "data_file_sha256", "data_manifest"):
        require(reference.get(key) == reconstructed.get(key), "exact_result_field:" + key)
    expected_runs = {run["seed"]: run for run in reference.get("runs", [])}
    actual_runs = {run["seed"]: run for run in reconstructed.get("runs", [])}
    require(set(expected_runs) == set(actual_runs), "seed_set")
    for seed, old in expected_runs.items():
        if seed not in actual_runs:
            continue
        new = actual_runs[seed]
        for key in ("selected_epochs", "initial_bridge_state_sha256", "shared_epoch_training_order_sha256", "optimizer_steps_per_arm", "parameters_per_arm"):
            require(old.get(key) == new.get(key), f"seed:{seed}:{key}")
        old_gates, new_gates = old.get("selected_query_gates", {}), new.get("selected_query_gates", {})
        require(set(old_gates) == set(new_gates), f"seed:{seed}:gate_arm_set")
        for arm, values in old_gates.items():
            actual = new_gates.get(arm, [])
            require(len(values) == len(actual) and all(close(a, b, GATE_ATOL, GATE_RTOL) for a, b in zip(actual, values)), f"seed:{seed}:gates:{arm}")
            counts["selected_gates"] += len(values)
        old_history = {(row["epoch"], row["arm"]): row["validation"] for row in old.get("history", [])}
        new_history = {(row["epoch"], row["arm"]): row["validation"] for row in new.get("history", [])}
        require(len(old_history) == len(old.get("history", [])) and len(new_history) == len(new.get("history", [])), f"seed:{seed}:duplicate_history")
        old_validation = {("selected", arm): val for arm, val in old.get("selected_validation", {}).items()}
        new_validation = {("selected", arm): val for arm, val in new.get("selected_validation", {}).items()}
        old_validation.update(old_history)
        new_validation.update(new_history)
        require(set(old_validation) == set(new_validation), f"seed:{seed}:validation_keys")
        for key, value in old_validation.items():
            actual = new_validation.get(key, {})
            require(actual.get("record_accuracy") == value.get("record_accuracy") and actual.get("records") == value.get("records"), f"seed:{seed}:validation_accuracy:{key}")
            require(close(actual.get("restricted_ce"), value.get("restricted_ce")), f"seed:{seed}:validation_ce:{key}")
            counts["validation_records"] += 1
    keys = ("seed", "view", "condition", "family", "query_key", "variant")
    def indexed(rows):
        result = {}
        for row in rows:
            key = tuple(row.get(name) for name in keys)
            if key in result:
                failures.append("duplicate_prediction:" + repr(key))
            result[key] = row
        return result
    old_predictions, new_predictions = indexed(old_rows), indexed(new_rows)
    require(set(old_predictions) == set(new_predictions), "prediction_identity_set")
    choice_errors = ce_errors = 0
    max_ce_difference = 0.0
    for key, old in old_predictions.items():
        new = new_predictions.get(key)
        if new is None:
            continue
        choice_errors += new.get("choice") != old.get("choice") or new.get("gold") != old.get("gold")
        if not close(new.get("restricted_ce"), old.get("restricted_ce")):
            ce_errors += 1
        if isinstance(new.get("restricted_ce"), (int, float)) and math.isfinite(new["restricted_ce"]):
            max_ce_difference = max(max_ce_difference, abs(new["restricted_ce"] - old["restricted_ce"]))
    require(choice_errors == 0, "predicted_choice_or_gold_mismatch")
    require(ce_errors == 0, "prediction_ce_tolerance")
    counts.update(prediction_rows=len(old_predictions), choice_or_gold_mismatches=choice_errors,
                  ce_mismatches=ce_errors, maximum_absolute_prediction_ce_difference=max_ce_difference)
    return {"passed": not failures, "failures": failures, "checked": counts}


def verify_reconstruction(reference_dir, reconstruction_dir, source_manifest, project_root):
    reference_dir, reconstruction_dir, project_root = map(Path, (reference_dir, reconstruction_dir, project_root))
    old = json.loads((reference_dir / "results.json").read_text())
    new = json.loads((reconstruction_dir / "results.json").read_text())
    old_blob, new_blob = prediction_blob(reference_dir), prediction_blob(reconstruction_dir)
    result = compare_results(old, new, [json.loads(line) for line in old_blob.splitlines()], [json.loads(line) for line in new_blob.splitlines()])
    source = json.loads(Path(source_manifest).read_text())
    failures = result["failures"]
    if source.get("frozen_commit") != FROZEN_COMMIT:
        failures.append("source_manifest_frozen_commit")
    if set(source.get("files", {})) != set(HISTORICAL_MODULES):
        failures.append("source_manifest_module_set")
    source_checks = {}
    for name in HISTORICAL_MODULES:
        path = project_root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        expected = source.get("files", {}).get(name, {}).get("expected_sha256")
        supplied_actual = source.get("files", {}).get(name, {}).get("actual_sha256")
        source_checks[name] = {"expected_sha256": expected, "actual_sha256": actual}
        if not expected or expected != actual or supplied_actual not in (None, actual):
            failures.append("historical_source_hash:" + name)
    if len(old_blob.splitlines()) != 41472:
        failures.append("reference_prediction_count")
    if hashlib.sha256(old_blob).hexdigest() != "324d8184f7730507da651eea0c49e16b646563d0746bc4d54f5747dfa798757c":
        failures.append("reference_prediction_sha256")
    result.update(passed=not failures, source_checks=source_checks, historical_source_commit=FROZEN_COMMIT,
        raw_prediction_sha256_exact=hashlib.sha256(old_blob).hexdigest() == hashlib.sha256(new_blob).hexdigest(),
        reference_raw_prediction_sha256=hashlib.sha256(old_blob).hexdigest(), reconstructed_raw_prediction_sha256=hashlib.sha256(new_blob).hexdigest(),
        tolerances={"ce_atol": CE_ATOL, "ce_rtol": CE_RTOL, "gate_atol": GATE_ATOL, "gate_rtol": GATE_RTOL},
        interpretation="Verified behavioral reconstruction only if all gates pass; absent historical final checkpoint hashes, no claim of historical message byte identity.")
    return result
