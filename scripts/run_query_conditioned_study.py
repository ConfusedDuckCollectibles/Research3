"""Run the fixed fresh-world public-query conditioning follow-up."""
from __future__ import annotations

import gzip
import hashlib
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


def main():
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    output = ROOT / "checkpoints" / "query-conditioned-study"
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "local_data" / "query_conditioned"
    manifest = {
        "status": "downloaded_pinned_public_pretrained_models", "models": {},
        "python": platform.python_version(), "platform": platform.platform(),
        "github_commit": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "remote_code_allowed": False, "hub_credentials_used": False,
        "design_sha256": hashlib.sha256((ROOT / "docs/query_conditioned_design.md").read_bytes()).hexdigest(),
    }
    for role, spec in MODELS.items():
        destination = ROOT / "local_models" / role
        print(f"Downloading public {role}: {spec['id']} @ {spec['revision']}", flush=True)
        before = time.perf_counter()
        snapshot_download(repo_id=spec["id"], revision=spec["revision"],
                          local_dir=str(destination), allow_patterns=DOWNLOAD_FILES,
                          token=False, max_workers=2)
        if not (destination / "model.safetensors").is_file():
            raise RuntimeError(f"pinned {role} snapshot lacks safetensors weights")
        manifest["models"][role] = {
            **spec, "files": {p.name: file_digest(p) for p in sorted(destination.iterdir()) if p.is_file()},
            "download_seconds": time.perf_counter() - before,
        }
    model_manifest = output / "model_manifest.json"
    model_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    generator = [sys.executable, "-m", "latent_bridge.query_conditioned_data",
                 "--output", str(data), "--seed", "20261125",
                 "--train-families", "128", "--validation-families", "32", "--test-families", "32"]
    subprocess.run(generator, cwd=ROOT, check=True)
    (output / "dataset_manifest.json").write_bytes((data / "manifest.json").read_bytes())
    command = [
        sys.executable, "-m", "latent_bridge.query_conditioned",
        "--sender-model", str(ROOT / "local_models" / "sender"),
        "--receiver-model", str(ROOT / "local_models" / "receiver"),
        "--data-dir", str(data), "--output", str(output),
        "--seeds", "101", "202", "303", "--epochs", "8", "--batch-size", "16",
        "--slots", "2", "--bottleneck", "8", "--learning-rate", "0.001",
        "--weight-decay", "0.01", "--max-length", "192", "--threads", "2",
        "--model-manifest", str(model_manifest),
    ]
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    predictions = output / "predictions.jsonl"
    compressed = output / "predictions.jsonl.gz"
    compressed.write_bytes(gzip.compress(predictions.read_bytes(), compresslevel=9, mtime=0))
    execution = {
        "generator_command": generator, "runner_command": command,
        "wall_seconds_including_downloads": time.perf_counter() - started,
        "models": MODELS, "run_kind": "prespecified_fresh_world_query_conditioning_study",
        "design_sha256": manifest["design_sha256"],
        "raw_predictions": file_digest(predictions), "compressed_predictions": file_digest(compressed),
        "report_sha256": hashlib.sha256((output / "report.md").read_bytes()).hexdigest(),
    }
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    for name in ("results.json", "predictions.jsonl.gz", "report.md", "dataset_manifest.json", "model_manifest.json", "execution.json"):
        emit_report(output / name)


if __name__ == "__main__":
    main()
