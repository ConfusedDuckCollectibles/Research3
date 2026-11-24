"""Execute the fixed three-seed paired-fact study, without paid model APIs."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zlib

from run_pretrained_pilot import DOWNLOAD_FILES, MODELS, file_digest


ROOT = Path(__file__).resolve().parents[1]


def emit_report(path):
    payload = path.read_bytes()
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    print(f"STUDY_FILE_BEGIN {path.name} {hashlib.sha256(payload).hexdigest()}", flush=True)
    for index, offset in enumerate(range(0, len(encoded), 3000)):
        print(f"STUDY_FILE_CHUNK {path.name} {index} {encoded[offset:offset + 3000]}", flush=True)
    print(f"STUDY_FILE_END {path.name}", flush=True)


def main():
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    output = ROOT / "checkpoints" / "counterfactual-study"
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "local_data" / "counterfactual"
    manifest = {
        "status": "downloaded_pinned_public_pretrained_models",
        "models": {},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "github_commit": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "remote_code_allowed": False,
        "hub_credentials_used": False,
        "design_sha256": hashlib.sha256((ROOT / "docs/counterfactual_design.md").read_bytes()).hexdigest(),
    }
    for role, spec in MODELS.items():
        destination = ROOT / "local_models" / role
        print(f"Downloading public {role}: {spec['id']} @ {spec['revision']}", flush=True)
        before = time.perf_counter()
        snapshot_download(
            repo_id=spec["id"], revision=spec["revision"],
            local_dir=str(destination), allow_patterns=DOWNLOAD_FILES,
            token=False, max_workers=2,
        )
        if not (destination / "model.safetensors").is_file():
            raise RuntimeError(f"pinned {role} snapshot lacks expected safetensors weights")
        manifest["models"][role] = {
            **spec,
            "files": {p.name: file_digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
            "download_seconds": time.perf_counter() - before,
        }
    model_manifest = output / "model_manifest.json"
    model_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    generator = [
        sys.executable, "-m", "latent_bridge.private_counterfactual",
        "--output", str(data), "--seed", "20261124",
        "--train-families", "128", "--validation-families", "32",
        "--test-families", "32",
    ]
    subprocess.run(generator, cwd=ROOT, check=True)
    (output / "dataset_manifest.json").write_bytes((data / "manifest.json").read_bytes())
    command = [
        sys.executable, "-m", "latent_bridge.counterfactual_runner",
        "--sender-model", str(ROOT / "local_models" / "sender"),
        "--receiver-model", str(ROOT / "local_models" / "receiver"),
        "--data-dir", str(data), "--output", str(output),
        "--seeds", "101", "202", "303", "--epochs", "8",
        "--batch-size", "16", "--slots", "2", "--bottleneck", "8",
        "--learning-rate", "0.001", "--max-length", "192", "--threads", "2",
        "--model-manifest", str(model_manifest),
    ]
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    predictions = output / "predictions.jsonl"
    compressed_predictions = output / "predictions.jsonl.gz"
    compressed_predictions.write_bytes(gzip.compress(predictions.read_bytes(), compresslevel=9, mtime=0))
    execution = {
        "generator_command": generator,
        "runner_command": command,
        "wall_seconds_including_downloads": time.perf_counter() - started,
        "models": MODELS,
        "run_kind": "prespecified_three_seed_counterfactual_validity_study",
        "design_sha256": manifest["design_sha256"],
        "raw_predictions": file_digest(predictions),
        "compressed_predictions": file_digest(compressed_predictions),
        "report_sha256": hashlib.sha256((output / "report.md").read_bytes()).hexdigest(),
    }
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    for name in ("results.json", "predictions.jsonl.gz", "report.md", "dataset_manifest.json", "model_manifest.json", "execution.json"):
        emit_report(output / name)


if __name__ == "__main__":
    main()
