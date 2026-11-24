"""Run the fixed, public-model CPU pilot documented in docs/pilot_design.md.

Only this wrapper downloads files. The experiment itself loads local models
with remote code disabled. Model weights and checkpoints are not published.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zlib


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "sender": {
        "id": "distilbert/distilgpt2",
        "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    },
    "receiver": {
        "id": "EleutherAI/pythia-14m",
        "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
    },
}
DOWNLOAD_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.model",
]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def emit_report(path, name):
    """Losslessly carry small public reports in logs if artifacts are unavailable."""
    payload = path.read_bytes()
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    print(f"PILOT_FILE_BEGIN {name} {hashlib.sha256(payload).hexdigest()}", flush=True)
    for index, offset in enumerate(range(0, len(encoded), 3000)):
        print(f"PILOT_FILE_CHUNK {name} {index} {encoded[offset:offset + 3000]}", flush=True)
    print(f"PILOT_FILE_END {name}", flush=True)


def main():
    from huggingface_hub import snapshot_download
    from latent_bridge.private_lookup import write_tasks

    started = time.perf_counter()
    output = ROOT / "checkpoints" / "pretrained-pilot"
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "local_data" / "private_lookup"
    manifest = {
        "status": "downloaded_pinned_public_pretrained_models",
        "models": {},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "github_commit": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "remote_code_allowed": False,
        "hub_credentials_used": False,
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
    dataset = write_tasks(
        data, seed=20261123, train_worlds=64, validation_worlds=16,
        test_worlds=16, shift_worlds=16,
    )
    (output / "dataset_manifest.json").write_text(json.dumps(dataset, indent=2) + "\n")
    command = [
        sys.executable, "-m", "latent_bridge.transformers_runner",
        "--sender-model", str(ROOT / "local_models" / "sender"),
        "--receiver-model", str(ROOT / "local_models" / "receiver"),
        "--train", str(data / "train.jsonl"),
        "--validation", str(data / "validation.jsonl"),
        "--test", str(data / "test.jsonl"),
        "--generalization-test", str(data / "generalization.jsonl"),
        "--model-manifest", str(model_manifest),
        "--slots", "2", "--bottleneck", "8", "--epochs", "3",
        "--seed", "101", "--max-length", "128", "--threads", "2",
        "--max-new-tokens", "4", "--learning-rate", "0.001",
        "--output", str(output),
    ]
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    execution = {
        "command": command,
        "wall_seconds_including_downloads": time.perf_counter() - started,
        "models": MODELS,
        "run_kind": "one_seed_bounded_exploratory_pilot",
    }
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    for name in ("results.json", "predictions.jsonl", "dataset_manifest.json", "model_manifest.json", "execution.json"):
        emit_report(output / name, name)


if __name__ == "__main__":
    main()
