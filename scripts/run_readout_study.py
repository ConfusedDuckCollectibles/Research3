"""Reconstruct the frozen bridge cohort, then run the prespecified readout study."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_pretrained_pilot import DOWNLOAD_FILES, MODELS, file_digest
from run_counterfactual_study import emit_report


ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "fb793ba31d7a10975a9e5c6032b44ecb5eb5774f"
REFERENCE_COMMIT = "97846d07118361d7548ff13f94eab1b276ddf9e9"
HISTORICAL_MODULES = (
    "latent_bridge/query_conditioned.py",
    "latent_bridge/query_conditioned_data.py",
    "latent_bridge/counterfactual_runner.py",
    "latent_bridge/private_counterfactual.py",
    "latent_bridge/transformers_runner.py",
    "latent_bridge/private_lookup.py",
)
PINNED_PACKAGES = {
    "torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
    "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify_sources(output: Path) -> Path:
    files = {}
    for name in HISTORICAL_MODULES:
        reference = subprocess.run(
            ["git", "show", f"{FROZEN_COMMIT}:{name}"], cwd=ROOT,
            check=True, capture_output=True, timeout=30,
        ).stdout
        files[name] = {
            "expected_sha256": hashlib.sha256(reference).hexdigest(),
            "actual_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
        }
    reference_files = {}
    for filename in ("results.json", "predictions.jsonl.gz", "dataset_manifest.json", "model_manifest.json"):
        name = "results/query_conditioned_study/" + filename
        reference = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:{name}"], cwd=ROOT,
            check=True, capture_output=True, timeout=30,
        ).stdout
        reference_files[name] = {
            "expected_sha256": hashlib.sha256(reference).hexdigest(),
            "actual_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
        }
    manifest = {
        "frozen_commit": FROZEN_COMMIT, "files": files,
        "reference_results_commit": REFERENCE_COMMIT, "reference_files": reference_files,
        "all_match": all(item["expected_sha256"] == item["actual_sha256"]
                         for item in (*files.values(), *reference_files.values())),
        "expected_source": "git show of the published immutable historical commit",
    }
    path = output / "source_manifest.json"
    write_json(path, manifest)
    if not manifest["all_match"]:
        raise RuntimeError("Historical source or reference outputs changed; reconstruction not started")
    return path


def main() -> None:
    started = time.perf_counter()
    # Leave time for artifact retention within the declared 20-minute job cap.
    deadline = started + 1020
    output = ROOT / "checkpoints" / "readout-study"
    output.mkdir(parents=True, exist_ok=True)
    original_data = ROOT / "local_data" / "readout_original"
    fresh_data = ROOT / "local_data" / "readout_fresh"
    reconstruction = output / "reconstruction"
    commands = []
    phases = {}
    failure = None
    model_manifest_path = output / "model_manifest.json"

    def run(label: str, arguments: list[str]) -> None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Readout wrapper execution budget exhausted")
        before = time.perf_counter()
        environment = dict(os.environ)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        commands.append({"phase": label, "argv": arguments})
        print(f"Starting {label}", flush=True)
        try:
            subprocess.run(arguments, cwd=ROOT, env=environment, check=True, timeout=remaining)
        finally:
            phases[label] = time.perf_counter() - before

    try:
        source_manifest = verify_sources(output)
        actual_versions = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
        runtime = {
            "python": platform.python_version(), "expected_python": "3.11.18",
            "actual_packages": actual_versions, "expected_packages": PINNED_PACKAGES,
            "platform": platform.platform(), "github_commit": os.environ.get("GITHUB_SHA"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        }
        write_json(output / "runtime_manifest.json", runtime)
        if runtime["python"] != runtime["expected_python"] or actual_versions != PINNED_PACKAGES:
            raise RuntimeError("Pinned reconstruction runtime differs; no model outcomes started")

        from huggingface_hub import snapshot_download

        model_manifest = {
            "status": "downloaded_pinned_public_pretrained_models", "models": {},
            "python": platform.python_version(), "platform": platform.platform(),
            "github_commit": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "historical_training_commit": FROZEN_COMMIT,
            "reference_results_commit": REFERENCE_COMMIT,
            "design_sha256": hashlib.sha256((ROOT / "docs/readout_design.md").read_bytes()).hexdigest(),
            "remote_code_allowed": False, "hub_credentials_used": False,
        }
        for role, spec in MODELS.items():
            destination = ROOT / "local_models" / role
            before = time.perf_counter()
            print(f"Downloading public {role}: {spec['id']} @ {spec['revision']}", flush=True)
            snapshot_download(repo_id=spec["id"], revision=spec["revision"],
                              local_dir=str(destination), allow_patterns=DOWNLOAD_FILES,
                              token=False, max_workers=2)
            if not (destination / "model.safetensors").is_file():
                raise RuntimeError(f"Pinned {role} model lacks expected safetensors weights")
            model_manifest["models"][role] = {
                **spec, "files": {p.name: file_digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
                "download_seconds": time.perf_counter() - before,
            }
        write_json(model_manifest_path, model_manifest)

        run("recreate_original_data", [
            sys.executable, "-m", "latent_bridge.query_conditioned_data",
            "--output", str(original_data), "--seed", "20261125",
            "--train-families", "128", "--validation-families", "32", "--test-families", "32",
        ])
        run("reconstruct_original_cohort", [
            sys.executable, "-m", "latent_bridge.query_conditioned",
            "--sender-model", str(ROOT / "local_models" / "sender"),
            "--receiver-model", str(ROOT / "local_models" / "receiver"),
            "--data-dir", str(original_data), "--output", str(reconstruction),
            "--seeds", "101", "202", "303", "--epochs", "8", "--batch-size", "16",
            "--slots", "2", "--bottleneck", "8", "--learning-rate", "0.001",
            "--weight-decay", "0.01", "--max-length", "192", "--threads", "2",
            "--model-manifest", str(model_manifest_path),
        ])
        run("generate_fresh_diagnostic_data", [
            sys.executable, "-m", "latent_bridge.readout_data",
            "--original-data-dir", str(original_data), "--output", str(fresh_data),
            "--seed", "20261126", "--test-families", "32",
        ])
        run("verify_reconstruction_and_probe", [
            sys.executable, "-m", "latent_bridge.readout_study",
            "--sender-model", str(ROOT / "local_models" / "sender"),
            "--receiver-model", str(ROOT / "local_models" / "receiver"),
            "--original-data-dir", str(original_data), "--reconstruction-dir", str(reconstruction),
            "--reference-dir", str(ROOT / "results" / "query_conditioned_study"),
            "--data-dir", str(fresh_data), "--output", str(output),
            "--source-manifest", str(source_manifest), "--model-manifest", str(model_manifest_path),
            "--seeds", "101", "202", "303", "--epochs", "80", "--batch-size", "64",
            "--learning-rate", "0.001", "--weight-decay", "0.01",
            "--max-source-positions", "64", "--threads", "2",
        ])
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Study stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        retained = {
            str(path.relative_to(output)): file_digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "execution.json"
        }
        execution = {
            "status": "completed" if failure is None else "stopped_with_recorded_failure",
            "failure": failure, "run_kind": "prespecified_frozen_message_readout_diagnostic",
            "historical_training_commit": FROZEN_COMMIT, "reference_results_commit": REFERENCE_COMMIT,
            "github_commit": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "commands": commands, "phase_seconds": phases,
            "wall_seconds_including_downloads": time.perf_counter() - started,
            "wrapper_budget_seconds": 1020, "artifact_retention_days": 90,
            "design_sha256": hashlib.sha256((ROOT / "docs/readout_design.md").read_bytes()).hexdigest(),
            "retained_files": retained,
        }
        write_json(output / "execution.json", execution)
        # Small recovery fallback only; complete data/checkpoints use the ZIP artifact.
        for name in ("reconstruction_gate.json", "report.md", "source_manifest.json", "runtime_manifest.json", "execution.json"):
            path = output / name
            if path.is_file() and path.stat().st_size <= 300_000:
                emit_report(path)
        print(json.dumps({"status": execution["status"], "failure": failure,
                          "wall_seconds": execution["wall_seconds_including_downloads"],
                          "retained_file_count": len(retained)}, sort_keys=True), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
