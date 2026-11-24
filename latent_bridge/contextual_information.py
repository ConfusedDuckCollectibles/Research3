"""One frozen contextual-state/information-recovery assay.

The pretrained worker is activated only by the separately published supervisor.
Pure helpers support offline tests; this module never starts an experiment on
import. The actual author method is compiled without importing its application.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import contextlib
import io
import importlib.metadata
import json
import math
import statistics
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = "docs/contextual_information_design.md"
DESIGN_COMMIT = "c23a20ec420ba19680801a09ac8b053735807d38"
DESIGN_SHA256 = "6f50e1d3faad3240033748d7434f088ec26b7ac6716511923892e66ac0e78a09"
THEORY_PATH = "docs/statebridge_information_recovery.md"
THEORY_SHA256 = "f78a41cee2b1ee90b349bcbee0fe00a588b9738863034172a8c8c09ddc3992d7"
VENDOR_SOURCE = ROOT / "third_party/statebridge/state_bridge.py"
VENDOR_SHA256 = "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d"
LICENSE_BLOB = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LOCK_PATH = ROOT / "scripts/contextual_model_lock.json"
LOCK_BLOB = "eba3adf2509be7409c6bc37e4f830a076c575c68"
CORE_SHA256 = "de8909692a928266c03e22d12b13c800b14e2495a8acad39ae6b95dfc46bfa53"
MODEL_ID = "EleutherAI/pythia-14m"
MODEL_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"
SEED = 41029
TREES = ("alder", "birch", "cedar", "elm", "fir", "maple", "oak", "willow")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white")
PROBE_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight")
WORLD_NAMES = ("original", "swapped")
PROBE_NAMES = ("unique", "repeated")
AUTHOR_LOCALS = ("Hc", "H_w", "E_w", "mu_H", "mu_E", "Cov_H", "Cov_E",
                 "R", "sqrt_Cov_E", "aligned_flat", "flat", "top_ids")
EXPECTED_PACKAGES = {
    "torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
    "tokenizers": "0.21.4", "safetensors": "0.8.0",
    "huggingface-hub": "0.36.2",
}
REQUIRED_MODEL_FILES = {
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
}


class PreflightError(ValueError):
    """A fixed pre-outcome input contract is unsupported; no replacement."""


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == -float("inf"):
            return "-Infinity"
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return value


def canonical_json(value):
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def file_info(path):
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            digest.update(block)
    return {"bytes": count, "sha256": digest.hexdigest()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(json_safe(value), sort_keys=True, indent=2,
                      ensure_ascii=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(stream, value):
    stream.write(canonical_json(value).decode("utf-8") + "\n")
    stream.flush()


def build_cohort():
    """Generate the exact 64 paired families without any model/tokenizer."""
    rng = random.Random(SEED)
    reserved = set()
    families = []
    total = 0
    for f in range(64):
        target = f % 8
        partner = (target + 1 + ((f // 8) % 7)) % 8
        counts = {"draws": 0, "rejected_original_reserved": 0,
                  "rejected_swapped_reserved": 0}
        while True:
            if total >= 1000000:
                raise PreflightError("fixed_generator_candidate_limit")
            p = list(range(8))
            rng.shuffle(p)
            total += 1
            counts["draws"] += 1
            swapped = list(p)
            swapped[target], swapped[partner] = swapped[partner], swapped[target]
            if tuple(p) in reserved:
                counts["rejected_original_reserved"] += 1
                continue
            if tuple(swapped) in reserved:
                counts["rejected_swapped_reserved"] += 1
                continue
            reserved.update((tuple(p), tuple(swapped)))
            break
        family_id = f"F{f:03d}"
        question = f"What color is the {TREES[target]} specimen?"
        worlds = {}
        for name, permutation in zip(WORLD_NAMES, (p, swapped)):
            lines = [f"Family {family_id}.", "Private record:"]
            lines.extend(f"{tree}: {COLORS[color]}"
                         for tree, color in zip(TREES, permutation))
            lines.extend((f"Question: {question}", "Answer:"))
            worlds[name] = {
                "permutation": permutation, "world_id": "".join(map(str, permutation)),
                "answer_index": permutation[target], "answer": COLORS[permutation[target]],
                "prompt": "\n".join(lines),
            }
        families.append({
            "family_id": family_id, "index": f, "target_index": target,
            "partner_index": partner, "question": question,
            "worlds": worlds, "generation": counts,
        })
    ordered = [list(p) for p in sorted(reserved)]
    return {
        "schema_version": 1, "seed": SEED, "trees": list(TREES),
        "colors": list(COLORS), "families": families,
        "generation": {"total_candidates": total, "accepted_families": 64,
                       "reserved_world_count": len(reserved)},
        "reserved_worlds": ordered,
        "reserved_worlds_sha256": sha256_bytes(canonical_json(ordered)),
        "reservation_hash_encoding": "canonical ASCII JSON of lexicographically sorted integer lists",
    }


def build_plan(cohort=None):
    cohort = build_cohort() if cohort is None else cohort
    planned = []
    for family in cohort["families"]:
        for world in WORLD_NAMES:
            for probe in PROBE_NAMES:
                planned.append({
                    "record_id": f"{family['family_id']}/{world}/{probe}",
                    "family_id": family["family_id"], "world": world, "probe": probe,
                    "question": family["question"],
                    "target_index": family["target_index"],
                    "answer_index": family["worlds"][world]["answer_index"],
                })
    return {"schema_version": 1, "expected_count": 256, "planned_records": planned}


def tokenize_cohort(tokenizer, cohort):
    def single(text):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) != 1 or not isinstance(ids[0], int) or ids[0] < 0:
            raise PreflightError("required_single_token:" + repr(text))
        return int(ids[0])

    unique = [single(" " + word) for word in PROBE_WORDS]
    color_ids = [single(" " + color) for color in COLORS]
    if len(set(unique)) != 8:
        raise PreflightError("probe_ids_not_distinct")
    if len(set(color_ids)) != 8:
        raise PreflightError("color_ids_not_distinct")
    prompts, lengths, pair_checks = {}, {}, []
    for family in cohort["families"]:
        pair = []
        for world in WORLD_NAMES:
            key = family["family_id"] + "/" + world
            ids = list(tokenizer.encode(family["worlds"][world]["prompt"],
                                        add_special_tokens=False))
            if not ids or len(ids) > 256 or any(not isinstance(x, int) or x < 0 for x in ids):
                raise PreflightError("invalid_prompt_tokens:" + key)
            prompts[key] = [int(x) for x in ids]
            lengths[key] = len(ids)
            pair.append(ids)
        same_length = len(pair[0]) == len(pair[1])
        same_bag = Counter(pair[0]) == Counter(pair[1])
        if not same_length or not same_bag:
            raise PreflightError("paired_prompt_token_bag_mismatch:" + family["family_id"])
        pair_checks.append({"family_id": family["family_id"],
                            "same_length": True, "same_token_multiset": True})
    return {
        "schema_version": 1, "probe_ids": {"unique": unique, "repeated": [unique[0]] * 8},
        "color_ids": color_ids, "prompt_ids": prompts, "prompt_lengths": lengths,
        "paired_preflight": pair_checks, "passed": True,
    }


def raw_array(value):
    import numpy as np
    a = np.asarray(value)
    if a.dtype.hasobject or a.dtype.kind not in "biuf":
        raise ValueError("Only real numeric arrays may be retained")
    dtype = a.dtype.newbyteorder("<")
    a = np.array(a, dtype=dtype, order="C", copy=True)
    return a, a.tobytes(order="C")


def array_bits_equal(left, right):
    a, ab = raw_array(left)
    b, bb = raw_array(right)
    return a.dtype.str == b.dtype.str and a.shape == b.shape and ab == bb


def load_author_method(path=VENDOR_SOURCE):
    """Compile only the SHA-pinned method, without executing author imports."""
    import torch
    import torch.nn.functional as functional
    path = Path(path)
    data = path.read_bytes()
    if sha256_bytes(data) != VENDOR_SHA256:
        raise ValueError("pinned_author_source_mismatch")
    tree = ast.parse(data.decode("utf-8"), filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "StateBridge"]
    if len(classes) != 1:
        raise ValueError("author_class_not_unique")
    methods = [node for node in classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name == "_align_hidden_sequence"]
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError("author_method_not_unique_or_decorated")
    method = methods[0]
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "F": functional, "__builtins__": __builtins__}
    exec(compile(module, str(path), "exec"), namespace)
    function = namespace["_align_hidden_sequence"]
    body_dump = ast.dump(method, annotate_fields=True, include_attributes=True)
    return function, {
        "vendor_sha256": sha256_bytes(data),
        "method_ast_sha256": sha256_bytes(body_dump.encode("utf-8")),
        "method_name": "_align_hidden_sequence", "body_modified": False,
        "application_imported": False, "method_line": method.lineno,
        "method_end_line": method.end_lineno,
    }


def invoke_with_capture(function, context, hidden, token_ids):
    """Copy named tensor locals at return; restore the existing profiler."""
    import torch
    captured = {}
    seen = 0
    previous = sys.getprofile()

    def profile(frame, event, result):
        nonlocal seen
        if event == "return" and frame.f_code is function.__code__:
            seen += 1
            for name in AUTHOR_LOCALS:
                value = frame.f_locals.get(name)
                if isinstance(value, torch.Tensor):
                    captured[name] = value.detach().cpu().clone()

    try:
        sys.setprofile(profile)
        output = function(context, hidden, token_ids)
    finally:
        sys.setprofile(previous)
    if seen != 1:
        raise RuntimeError("author_return_capture_count:" + str(seen))
    if not isinstance(output, torch.Tensor):
        raise TypeError("author_return_not_tensor")
    return output, captured


def extract_sequence(model, prompt_ids, probe_ids, color_ids):
    """Eight authentic cached prediction-step states; no token is sampled."""
    import numpy as np
    import torch
    if len(probe_ids) != 8 or len(color_ids) != 8 or not prompt_ids:
        raise ValueError("fixed_extraction_shape")
    if model.training:
        raise ValueError("model_must_be_eval")
    layer = model.gpt_neox.layers[-1]
    states, color_logits, forced_logits, log_probs, hashes, steps = [], None, [], [], [], []
    hook_values = []

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.dtype != torch.float32:
            raise ValueError("unexpected_hook_tensor")
        hook_values.append(hidden[:, -1:, :].detach().cpu().clone())

    handle = layer.register_forward_hook(hook)
    past = None
    try:
        with torch.no_grad():
            for step in range(8):
                hook_values.clear()
                consumed = list(prompt_ids) if step == 0 else [int(probe_ids[step - 1])]
                total = len(prompt_ids) + step
                positions = list(range(len(prompt_ids))) if step == 0 else [total - 1]
                inputs = torch.tensor([consumed], dtype=torch.long)
                position_ids = torch.tensor([positions], dtype=torch.long)
                output = model(
                    input_ids=inputs, attention_mask=torch.ones((1, total), dtype=torch.long),
                    position_ids=position_ids, cache_position=position_ids[0],
                    past_key_values=past, use_cache=True,
                    output_hidden_states=False, return_dict=True,
                )
                if len(hook_values) != 1:
                    raise RuntimeError("one_last_block_hook_required_per_call")
                state = hook_values[0].numpy().reshape(-1).copy()
                logits_tensor = output.logits[0, -1].detach().cpu()
                if logits_tensor.dtype != torch.float32:
                    raise RuntimeError("native_logits_not_float32")
                logits = logits_tensor.numpy().copy()
                if not np.isfinite(state).all() or not np.isfinite(logits).all():
                    raise RuntimeError("nonfinite_authentic_state_or_logits")
                if max(probe_ids + color_ids) >= len(logits):
                    raise ValueError("requested_token_outside_output_table")
                states.append(state)
                if step == 0:
                    color_logits = logits[np.asarray(color_ids, dtype=np.int64)].copy()
                selected = int(probe_ids[step])
                values64 = logits.astype(np.float64)
                maximum = float(values64.max())
                normalizer = maximum + float(np.log(np.exp(values64 - maximum).sum()))
                forced_logits.append(float(logits[selected]))
                log_probs.append(float(values64[selected] - normalizer))
                hashes.append(sha256_bytes(raw_array(logits)[1]))
                steps.append({
                    "step": step, "consumed_ids": consumed,
                    "position_ids": positions, "attention_mask_length": total,
                    "predicts_token_id": selected, "has_input_cache": step > 0,
                })
                past = output.past_key_values
                if past is None:
                    raise RuntimeError("missing_incremental_cache")
    finally:
        handle.remove()
    native64 = color_logits.astype(np.float64)
    probabilities = np.exp(native64 - native64.max())
    probabilities /= probabilities.sum()
    return {
        "source": np.stack(states).astype(np.float32),
        "first_color_logits": color_logits,
        "color_probabilities": probabilities,
        "native_prediction_index": int(np.argmax(color_logits)),
        "forced_logits": np.asarray(forced_logits, dtype=np.float32),
        "forced_log_probabilities": np.asarray(log_probs, dtype=np.float64),
        "forced_log_probability_sum": float(sum(log_probs)),
        "forced_log_probability_mean": float(sum(log_probs) / 8),
        "full_logits_sha256": hashes, "steps": steps,
        "model_forward_calls": 8, "hook": "gpt_neox.layers[-1], output before final_layer_norm",
        "sampling": False, "cache_reused_across_sequences": False,
    }


def public32_arrays(embedding_layer, probe_ids):
    """Public-only Torch calculation matching the author's exact operation order."""
    import numpy as np
    import torch
    with torch.no_grad():
        vocab = embedding_layer.weight.detach().float()
        ids = torch.tensor([probe_ids], dtype=torch.long)
        e = embedding_layer(ids).float().reshape(-1, embedding_layer.embedding_dim)
        mean = e.mean(dim=0, keepdim=True)
        ec = e - mean
        identity = torch.eye(e.shape[1], device=e.device, dtype=e.dtype)
        covariance = (ec.T @ ec) / len(e) + 1e-3 * identity

        def power(exponent):
            a = (covariance + covariance.T) * 0.5
            values, vectors = torch.linalg.eigh(a)
            values = values.clamp_min(1e-6).pow(exponent)
            return (vectors * values.unsqueeze(0)) @ vectors.T

        inverse = power(-0.5)
        target_white = ec @ inverse
        root = power(0.5)
        target_norm = vocab.norm(dim=-1).mean().item()
    arrays = {
        "reference": e.detach().cpu().numpy().copy(),
        "mean": mean.detach().cpu().numpy().copy(),
        "covariance": covariance.detach().cpu().numpy().copy(),
        "root": root.detach().cpu().numpy().copy(),
        "target_white": target_white.detach().cpu().numpy().copy(),
    }
    parameters = {
        "target_norm": float(target_norm),
        "lam": float(np.float32(1e-3)),
        "anchor_gain": float(np.float32(0.3)),
        "complement": float(np.float32(1.0 - 0.3)),
        "covariance_floor": float(np.float32(1e-6)),
        "row_floor": float(np.float32(1e-6)),
        "cosine_floor": float(np.float32(1e-12)),
        "convention": "public32",
    }
    return arrays, parameters


class EvidenceWriter:
    """One atomic NPZ per record; an append-only manifest records every saved stage."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = (self.output / "array_manifest.jsonl").open("a", encoding="utf-8")

    def close(self):
        self.manifest.close()

    def bundle(self, relative, arrays, stage):
        import numpy as np
        relative = str(relative)
        if (not re.fullmatch(r"(?:bundles/)?[A-Za-z0-9_-]+\.npz", relative)
                or relative.startswith("/")):
            raise ValueError("unsafe_bundle_path")
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("refusing_to_overwrite_array_bundle:" + relative)
        canonical, entries = {}, {}
        for name, value in sorted(arrays.items()):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError("unsafe_npz_member")
            a, payload = raw_array(value)
            canonical[name] = a
            entries[name] = {"dtype": a.dtype.str, "shape": list(a.shape),
                             "raw_bytes": len(payload), "raw_sha256": sha256_bytes(payload)}
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(temporary, **canonical)
        os.replace(temporary, path)
        receipt = {"bundle": relative, "stage": stage, **file_info(path), "arrays": entries}
        append_jsonl(self.manifest, receipt)
        return {name: {"bundle": relative, "member": name, **meta}
                for name, meta in entries.items()}


def verify_execution(output):
    """Bind the scientific child to the reviewed first-run supervisor and source."""
    output = Path(output)
    if output.is_symlink():
        raise RuntimeError("output_symlink")
    output.mkdir(parents=True, exist_ok=True)
    allowed = {"launcher.json", "launch_source_manifest.json", "worker.log"}
    if any(p.name not in allowed or p.is_symlink() for p in output.iterdir()):
        raise RuntimeError("refusing_existing_scientific_output")
    launcher = json.loads((output / "launcher.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "launch_source_manifest.json").read_text(encoding="utf-8"))
    env = os.environ
    if (env.get("GITHUB_ACTIONS") != "true"
            or env.get("GITHUB_REPOSITORY") != "ConfusedDuckCollectibles/Research3"
            or env.get("GITHUB_REF") != "refs/heads/main"
            or env.get("GITHUB_RUN_NUMBER") != "1" or env.get("GITHUB_RUN_ATTEMPT") != "1"):
        raise RuntimeError("only_first_reviewed_workflow_permitted")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=10).stdout.strip()
    for document in (launcher, manifest):
        if (document.get("github_commit") != commit or commit != env.get("GITHUB_SHA")
                or str(document.get("github_run_id")) != env.get("GITHUB_RUN_ID")
                or document.get("frozen_design_commit") != DESIGN_COMMIT
                or document.get("design_sha256") != DESIGN_SHA256):
            raise RuntimeError("launcher_source_identity_mismatch")
    if launcher.get("budget_seconds") != 600:
        raise RuntimeError("unexpected_scientific_budget")
    checked = {}
    for name, expected in manifest.get("files", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (ROOT / relative).is_symlink():
            raise RuntimeError("unsafe_source_inventory_path")
        actual = file_info(ROOT / relative)
        if actual != expected:
            raise RuntimeError("source_changed_after_launcher:" + name)
        checked[name] = actual
    frozen = {DESIGN_PATH: DESIGN_SHA256, THEORY_PATH: THEORY_SHA256,
              "latent_bridge/alignment_core.py": CORE_SHA256,
              "third_party/statebridge/state_bridge.py": VENDOR_SHA256}
    for name, expected in frozen.items():
        if file_info(ROOT / name)["sha256"] != expected:
            raise RuntimeError("frozen_source_mismatch:" + name)
    if git_blob_sha(LOCK_PATH.read_bytes()) != LOCK_BLOB:
        raise RuntimeError("model_lock_changed")
    if git_blob_sha((ROOT / "third_party/statebridge/LICENSE").read_bytes()) != LICENSE_BLOB:
        raise RuntimeError("author_license_changed")
    return {"schema_version": 1, "github_commit": commit,
            "github_run_id": env["GITHUB_RUN_ID"], "frozen_design_commit": DESIGN_COMMIT,
            "design_sha256": DESIGN_SHA256, "files": checked, "all_match": True}


def initialize_runtime():
    import numpy as np
    import torch
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if platform.python_version() != "3.11.18" or observed != EXPECTED_PACKAGES:
        raise RuntimeError("frozen_runtime_versions_mismatch:" + repr(observed))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise RuntimeError("thread_environment_mismatch:" + name)
    cpu_models = []
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if ":" in line and line.split(":", 1)[0].strip() in {"model name", "Hardware"}:
                value = line.split(":", 1)[1].strip()
                if value not in cpu_models:
                    cpu_models.append(value)
    except OSError:
        pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    build_output = io.StringIO()
    with contextlib.redirect_stdout(build_output):
        np.show_config()
    return {
        "python": platform.python_version(), "packages": observed,
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "cpu_models": cpu_models, "affinity": affinity,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "numpy_build": build_output.getvalue(),
        "thread_environment": {name: os.environ[name] for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }


def download_model(directory, output):
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if (lock.get("id") != MODEL_ID or lock.get("revision") != MODEL_REVISION
            or set(lock.get("files", {})) != REQUIRED_MODEL_FILES
            or lock.get("hub_credentials_used") is not False
            or lock.get("remote_code_allowed") is not False):
        raise RuntimeError("invalid_pinned_model_lock")
    directory = Path(directory)
    manifest = {"id": MODEL_ID, "revision": MODEL_REVISION, "files": {},
                "hub_credentials_used": False, "remote_code_allowed": False,
                "status": "downloading"}
    write_json(Path(output) / "model_manifest.json", manifest)
    for name in sorted(lock["files"]):
        expected = lock["files"][name]
        started = time.perf_counter()
        temporary = directory / (name + ".part")
        destination = directory / name
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}?download=true"
        count, digest = 0, hashlib.sha256()
        request = urllib.request.Request(url, headers={"User-Agent": "Research3-contextual-information/1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > expected["bytes"]:
                        raise RuntimeError("model_download_larger_than_pinned_file:" + name)
                    digest.update(block)
                    stream.write(block)
            actual = {"bytes": count, "sha256": digest.hexdigest()}
            if actual != expected:
                raise RuntimeError("model_download_digest_mismatch:" + name)
            os.replace(temporary, destination)
            manifest["files"][name] = {**actual, "download_seconds": time.perf_counter() - started}
            write_json(Path(output) / "model_manifest.json", manifest)
            print("Pinned model file verified: " + name, flush=True)
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failed_file"] = name
            manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
            write_json(Path(output) / "model_manifest.json", manifest)
            raise
    manifest["status"] = "verified"
    write_json(Path(output) / "model_manifest.json", manifest)
    return manifest


def prepare_public_contexts(model, tokenization, writer, output):
    """All inverse-side arrays are fixed before the first private extraction."""
    import numpy as np
    import torch
    from . import alignment_core as core
    from .information_recovery import PublicContext
    layer = model.get_input_embeddings()
    if layer.embedding_dim != 128 or layer.weight.dtype != torch.float32:
        raise PreflightError("unexpected_model_width_or_dtype")
    vocab32 = layer.weight.detach().cpu().numpy().copy()
    if not np.isfinite(vocab32).all():
        raise PreflightError("nonfinite_public_vocabulary")
    vocab64 = vocab32.astype(np.float64)
    output_rows = model.get_output_embeddings().weight.shape[0]
    all_ids = (tokenization["color_ids"]
               + tokenization["probe_ids"]["unique"]
               + [v for values in tokenization["prompt_ids"].values() for v in values])
    if max(all_ids) >= len(vocab32) or max(all_ids) >= output_rows:
        raise PreflightError("token_outside_actual_model_tables")
    with torch.no_grad():
        norms32 = layer.weight.detach().float().norm(dim=-1).cpu().numpy().copy()
    norms64 = np.linalg.norm(vocab64, axis=1)
    table_refs = writer.bundle(
        "table.npz", {"vocabulary_f32": vocab32, "vocabulary_norms_f32": norms32,
                      "vocabulary_norms_f64": norms64}, "public_before_extraction")
    groups, manifest = {}, {"schema_version": 1, "prepared_before_private_extraction": True,
                            "table": table_refs, "probes": {}}
    for probe in PROBE_NAMES:
        ids = tokenization["probe_ids"][probe]
        arrays32, parameters32 = public32_arrays(layer, ids)
        r = vocab64[np.asarray(ids, dtype=np.int64)]
        mean = r.mean(axis=0)
        centered = r - mean
        covariance = centered.T @ centered / len(r) + 1e-3 * np.eye(r.shape[1])
        values, vectors = core._eigh(covariance, [])
        values = np.maximum(values, 1e-6)
        root = core._from_values(vectors, np.sqrt(values))
        target_white = centered @ core._from_values(vectors, values ** -0.5)
        arrays64 = {"reference": r, "mean": mean.reshape(1, -1),
                    "covariance": covariance, "root": root, "target_white": target_white}
        parameters64 = {
            "target_norm": float(norms64.mean()), "lam": 1e-3,
            "anchor_gain": 0.3, "complement": 1.0 - 0.3,
            "covariance_floor": 1e-6, "row_floor": 1e-6,
            "cosine_floor": 1e-12, "convention": "public64",
        }
        context32 = PublicContext(vocab32, ids, arrays32["mean"], arrays32["root"],
                                  **parameters32)
        context64 = PublicContext(vocab64, ids, arrays64["mean"], arrays64["root"],
                                  **parameters64)
        retained = {"probe_ids": np.asarray(ids, dtype=np.int64)}
        retained.update({"public32_" + key: value for key, value in arrays32.items()})
        retained.update({"public64_" + key: value for key, value in arrays64.items()})
        refs = writer.bundle("public_" + probe + ".npz", retained, "public_before_extraction")
        groups[probe] = {
            "context32": context32, "context64": context64,
            "arrays32": arrays32, "arrays64": arrays64, "refs": refs,
            "parameters32": parameters32, "parameters64": parameters64,
        }
        manifest["probes"][probe] = {
            "arrays": refs, "public32": context32.metadata(), "public64": context64.metadata(),
            "public32_tensor_grouping": "exact author Torch float32 operations",
            "public64_tensor_grouping": "frozen NumPy dense-reference operations",
        }
    write_json(Path(output) / "public_manifest.json", manifest)
    return groups, vocab32, vocab64, manifest


def _prefix_filename(record_id, suffix=""):
    if not re.fullmatch(r"F0[0-6][0-9]/(?:original|swapped)/(?:unique|repeated)", record_id):
        raise ValueError("unexpected_record_id")
    return "bundles/" + record_id.replace("/", "_") + suffix + ".npz"


def _recovery_record(result, truth, arrays, prefix, actual_anchor_ids):
    """Truth is used only after source-free recovery has returned."""
    import numpy as np
    from .information_recovery import array_errors
    for name, value in result["arrays"].items():
        arrays[prefix + "_" + name] = value
    gram = result["gram"]
    if gram is not None:
        arrays[prefix + "_gram"] = gram
    metrics = array_errors(gram, truth) if gram is not None else None
    derived = result["arrays"].get("anchor_ids")
    disagreements = (int(np.count_nonzero(derived != actual_anchor_ids))
                     if derived is not None else None)
    return {
        "status": result["status"], "reason": result["reason"],
        "diagnostics": result["diagnostics"], "gram_error": metrics,
        "actual_anchor_disagreements": disagreements,
        "accuracy_comparison_after_inverse_return": True,
    }


def _finite_error(error, threshold):
    if not isinstance(error, dict) or error.get("finite") is not True:
        return False
    value = error.get("relative_frobenius")
    return isinstance(value, (int, float)) and value <= threshold


def process_record(planned, family, tokenization, model, author_function, author_context,
                   public, vocab64, writer, first_states, pair_cache):
    """One numerical record; inverse rejection is a completed observation."""
    import numpy as np
    import torch
    from . import alignment_core as core
    from .information_recovery import source_grams, recover_final, array_errors, rank_diagnostics

    record_id = planned["record_id"]
    probe = planned["probe"]
    world = planned["world"]
    key = planned["family_id"] + "/" + world
    row = {**planned, "status": "error", "arrays": {}, "validation": {},
           "phase_seconds": {}, "recovery": {}, "operational_errors": []}
    arrays = {}
    public_group = public[probe]
    pair_entry = {"record_id": record_id, "answer_index": planned["answer_index"]}
    pair_cache.setdefault((planned["family_id"], probe), {})[world] = pair_entry
    began = time.perf_counter()
    phase = "extraction"
    try:
        started = time.perf_counter()
        extracted = extract_sequence(
            model, tokenization["prompt_ids"][key],
            tokenization["probe_ids"][probe], tokenization["color_ids"])
        row["phase_seconds"]["extraction"] = time.perf_counter() - started
        source = extracted["source"]
        if source.shape != (8, 128) or source.dtype != np.dtype("float32"):
            raise RuntimeError("unexpected_pretrained_source_shape_or_dtype")
        pair_entry["source"] = source
        source_arrays = {name: extracted[name] for name in
                         ("source", "first_color_logits", "color_probabilities",
                          "forced_logits", "forced_log_probabilities")}
        row["extraction"] = {name: value for name, value in extracted.items()
                             if name not in source_arrays}
        row["extraction"]["native_correct"] = (
            extracted["native_prediction_index"] == planned["answer_index"])
        row["extraction"]["source_shape"] = list(source.shape)
        source_started = time.perf_counter()
        row["arrays"].update(writer.bundle(
            _prefix_filename(record_id, "_source"), source_arrays, "source_checkpoint"))
        row["phase_seconds"]["source_retention"] = time.perf_counter() - source_started
        if probe == "unique":
            first_states[key] = (source[0].copy(), extracted["first_color_logits"].copy())
            row["validation"]["first_step_contract"] = "baseline_recorded"
        else:
            previous = first_states.get(key)
            same = (previous is not None and array_bits_equal(source[0], previous[0])
                    and array_bits_equal(extracted["first_color_logits"], previous[1]))
            row["validation"]["first_step_contract"] = bool(same)

        phase = "source_geometry"
        truth = source_grams(source)
        arrays.update({
            "source_centered_f64": truth["centered"],
            "source_reduced_gram_f64": truth["reduced"],
            "source_full_gram_f64": truth["full"],
            "direct_gram_f32": truth["reduced"].astype(np.float32),
        })
        pair_entry["reduced_gram_f64"] = truth["reduced"]
        pair_entry["direct_gram_f32"] = arrays["direct_gram_f32"]
        row["source_geometry"] = truth["diagnostics"]
        eig = np.asarray(truth["diagnostics"]["reduced_eigenvalues"], dtype=np.float64)
        row["source_geometry"]["inverse_condition_factors_public32"] = (
            1.0 + eig / (8 * public_group["parameters32"]["lam"])).tolist()
        row["source_geometry"]["inverse_condition_factors_public64"] = (
            1.0 + eig / (8 * public_group["parameters64"]["lam"])).tolist()
        row["direct_gram_error"] = array_errors(arrays["direct_gram_f32"], truth["reduced"])
        eligible = (
            probe == "unique"
            and truth["diagnostics"]["centered"].get("rank") == 7
            and public_group["context64"].metadata()["reference"].get("rank") == 8
            and public_group["context64"].metadata()["anchor_geometry_supported"])
        row["validation"]["f64_reference_eligible"] = bool(eligible)
        row["validation"]["f64_reference_recovery_valid"] = False if eligible else None

        phase = "author_forward"
        started = time.perf_counter()
        hidden = torch.from_numpy(source.copy()).unsqueeze(0)
        ids = torch.tensor([tokenization["probe_ids"][probe]], dtype=torch.long)
        with torch.no_grad():
            returned, captured = invoke_with_capture(
                author_function, author_context, hidden, ids)
        if returned.dtype != torch.float32 or tuple(returned.shape) != (1, 8, 128):
            raise RuntimeError("author_return_dtype_or_shape")
        if set(captured) != set(AUTHOR_LOCALS):
            raise RuntimeError("incomplete_author_return_capture")
        final32 = returned.detach().cpu().numpy().reshape(8, 128).copy()
        pair_entry["author_final_f32"] = final32
        arrays["author_final_f32"] = final32
        shared_names = {"mu_E": "mean", "Cov_E": "covariance",
                        "sqrt_Cov_E": "root", "E_w": "target_white"}
        checks, local_refs, local_arrays = {}, {}, {}
        for name, value in captured.items():
            actual = value.numpy().copy()
            local_arrays[name] = actual
            if name in shared_names:
                public_name = shared_names[name]
                equal = array_bits_equal(actual, public_group["arrays32"][public_name])
                checks[name] = equal
                if equal:
                    local_refs[name] = public_group["refs"]["public32_" + public_name]
                    continue
            arrays["author_" + name] = actual
            local_refs[name] = {"pending_member": "author_" + name}
        row["author"] = {
            "output_dtype": str(returned.dtype), "actual_return_bytes": True,
            "output_finite": bool(np.isfinite(final32).all()),
            "captured_locals": sorted(captured), "local_refs": local_refs,
            "public_contract_checks": checks,
            "pre_rank": rank_diagnostics(local_arrays["aligned_flat"]),
            "calibrated_rank": rank_diagnostics(local_arrays["flat"]),
            "pre_mean_residual": array_errors(
                local_arrays["aligned_flat"].astype(np.float64).mean(axis=0),
                public_group["arrays32"]["mean"].astype(np.float64).reshape(-1)),
            "rotation_orthogonality": array_errors(
                local_arrays["R"].astype(np.float64).T @ local_arrays["R"].astype(np.float64),
                np.eye(128)),
        }
        row["validation"]["public32_contract"] = all(checks.values()) and len(checks) == 4
        row["phase_seconds"]["author_forward_and_capture"] = time.perf_counter() - started

        phase = "reference_forward"
        started = time.perf_counter()
        reference = core.dense_reference(source, public_group["arrays64"]["reference"],
                                         lam=1e-3, eps=1e-6, details=True)
        processed = core.postprocess(reference.pre, vocab64, alpha=0.3)
        cast_final = processed["final"].astype(np.float32)
        arrays.update({
            "reference_pre_f64": reference.pre,
            "reference_white_f64": reference.white,
            "reference_whitened_source_gram_f64": reference.sender_gram,
            "reference_target_white_f64": reference.target_white,
            "reference_calibrated_f64": processed["calibrated"],
            "reference_final_f64": processed["final"],
            "reference_anchor_ids": processed["ids"],
            "reference_anchor_margins": processed["margins"],
            "reference_final_cast_f32": cast_final,
        })
        pair_entry["reference_final_cast_f32"] = cast_final
        row["reference"] = {
            "diagnostics": reference.diagnostics,
            "target_norm": processed["target_norm"],
            "unique_product_structural": (
                truth["diagnostics"]["centered"].get("rank") == 7
                and public_group["context64"].metadata()["centered_reference"].get("rank") == 7),
            "f32_f64_completion_equality_required": False,
        }
        row["vector_differences"] = {
            "author_vs_reference_final": array_errors(final32, processed["final"]),
            "author_vs_reference_pre": array_errors(local_arrays["aligned_flat"], reference.pre),
            "reference_cast_vs_reference_final": array_errors(cast_final, processed["final"]),
        }
        row["phase_seconds"]["reference_forward"] = time.perf_counter() - started

        phase = "recovery"
        started = time.perf_counter()
        routes = (
            ("author_matched_public32", final32, public_group["context32"], local_arrays["top_ids"]),
            ("author_ideal_public64", final32, public_group["context64"], local_arrays["top_ids"]),
            ("reference_f64", processed["final"], public_group["context64"], processed["ids"]),
            ("reference_f64_cast_f32", cast_final, public_group["context64"], processed["ids"]),
        )
        for name, received, public_context, true_ids in routes:
            result = recover_final(received, public_context)
            row["recovery"][name] = _recovery_record(
                result, truth["reduced"], arrays, "inverse_" + name,
                np.asarray(true_ids).reshape(-1))
        row["phase_seconds"]["recovery"] = time.perf_counter() - started
        if eligible:
            row["validation"]["f64_reference_recovery_valid"] = _finite_error(
                row["recovery"]["reference_f64"]["gram_error"], 1e-8)
        row["status"] = "completed"
    except Exception as error:
        row["operational_errors"].append({
            "phase": phase, "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=12),
        })
    finally:
        try:
            started = time.perf_counter()
            if arrays:
                final_refs = writer.bundle(_prefix_filename(record_id), arrays, row["status"])
                row["arrays"].update(final_refs)
                if "author" in row:
                    for name, ref in list(row["author"]["local_refs"].items()):
                        if "pending_member" in ref:
                            row["author"]["local_refs"][name] = final_refs[ref["pending_member"]]
            row["phase_seconds"]["numerical_retention"] = time.perf_counter() - started
        except Exception as error:
            row["status"] = "error"
            row["operational_errors"].append({
                "phase": "retention", "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(limit=12),
            })
        retained_channels = {
            "source": "source", "reduced_gram_f64": "source_reduced_gram_f64",
            "direct_gram_f32": "direct_gram_f32", "author_final_f32": "author_final_f32",
            "reference_final_cast_f32": "reference_final_cast_f32",
        }
        for channel, member in retained_channels.items():
            if member not in row["arrays"]:
                pair_entry.pop(channel, None)
        row["wall_seconds"] = time.perf_counter() - began
    return row


def analyze_pair(family, probe, entries, token_reference):
    """Exact bytes and exact dyadic geometry, independent of inverse success."""
    import numpy as np
    from .information_recovery import exact_centered_gram_pair, array_errors
    left, right = entries.get("original", {}), entries.get("swapped", {})
    row = {
        "pair_id": family["family_id"] + "/" + probe, "family_id": family["family_id"],
        "question": family["question"], "probe": probe,
        "answers": [family["worlds"][world]["answer_index"] for world in WORLD_NAMES],
        "status": "missing_or_invalid", "channels": {},
        "preexisting_source_collision": None,
        "alignment_created_collision": None, "exact_gram_loss_collision": None,
    }
    channels = {
        "author_f32": "author_final_f32", "reference_f64_cast_f32": "reference_final_cast_f32",
        "direct_gram_f32": "direct_gram_f32", "raw_state_f32": "source",
    }
    for channel, name in channels.items():
        a, b = left.get(name), right.get(name)
        valid = (a is not None and b is not None and np.asarray(a).shape == np.asarray(b).shape
                 and np.asarray(a).dtype == np.dtype("float32")
                 and np.asarray(b).dtype == np.dtype("float32")
                 and np.isfinite(a).all() and np.isfinite(b).all())
        row["channels"][channel] = {
            "valid_pair": bool(valid), "byte_collision": array_bits_equal(a, b) if valid else None,
        }
        if valid:
            row["channels"][channel]["difference"] = array_errors(a, b)
    row["channels"]["token_only_f32"] = {
        "valid_pair": True, "byte_collision": True,
        "public_reference_raw_sha256": sha256_bytes(raw_array(token_reference)[1]),
    }
    source_left, source_right = left.get("source"), right.get("source")
    if source_left is not None and source_right is not None:
        try:
            row["exact_source_comparison"] = exact_centered_gram_pair(source_left, source_right)
        except Exception as error:
            row["exact_source_comparison_error"] = {
                "type": type(error).__name__, "message": str(error)}
    if left.get("reduced_gram_f64") is not None and right.get("reduced_gram_f64") is not None:
        row["stored_reduced_gram_f64_byte_equal"] = array_bits_equal(
            left["reduced_gram_f64"], right["reduced_gram_f64"])
        row["reduced_gram_difference"] = array_errors(
            left["reduced_gram_f64"], right["reduced_gram_f64"])
    actual = row["channels"]["author_f32"]
    if actual["valid_pair"]:
        row["status"] = "compared"
        exact = row.get("exact_source_comparison")
        if exact is not None:
            created = (actual["byte_collision"] and row["answers"][0] != row["answers"][1]
                       and not exact["numeric_values_equal"])
            row["preexisting_source_collision"] = bool(
                actual["byte_collision"] and exact["numeric_values_equal"])
            row["alignment_created_collision"] = bool(created)
            row["exact_gram_loss_collision"] = bool(created and not exact["gram_equal"])
    return row


def collision_summary(rows, expected_pairs=64):
    """Missing pairs remain unknown; inverse rejection never filters a prefix."""
    names = ("author_f32", "reference_f64_cast_f32", "direct_gram_f32",
             "raw_state_f32", "token_only_f32")
    summaries = {}
    for name in names:
        c = sum(r.get("channels", {}).get(name, {}).get("byte_collision") is True for r in rows)
        d = sum(r.get("channels", {}).get(name, {}).get("byte_collision") is False for r in rows)
        missing = expected_pairs - c - d
        if missing < 0:
            raise ValueError("duplicate_collision_rows")
        summaries[name] = {
            "expected_pairs": expected_pairs, "confirmed_collisions": c,
            "confirmed_distinct": d, "missing_or_invalid": missing,
            "full_codebook_ceiling": 1 - c / (2 * expected_pairs) if missing == 0 else None,
            "full_codebook_ceiling_bounds": [
                1 - (c + missing) / (2 * expected_pairs), 1 - c / (2 * expected_pairs)],
            "observed_only_ceiling": 1 - c / (2 * (c + d)) if c + d else None,
            "observed_pair_denominator": c + d,
        }
    return {
        "channels": summaries,
        "preexisting_source_collision_ids": [
            r["pair_id"] for r in rows if r.get("preexisting_source_collision") is True],
        "preexisting_source_collision_count": sum(
            r.get("preexisting_source_collision") is True for r in rows),
        "alignment_created_witnesses": [
            r["pair_id"] for r in rows if r.get("alignment_created_collision") is True],
        "exact_gram_loss_witnesses": [
            r["pair_id"] for r in rows if r.get("exact_gram_loss_collision") is True],
        "exact_source_comparison_failures": [
            r["pair_id"] for r in rows if "exact_source_comparison_error" in r],
    }


def _error_summary(values):
    valid = [float(x) for x in values
             if isinstance(x, (int, float)) and math.isfinite(x)]
    return {"finite_count": len(valid), "minimum": min(valid) if valid else None,
            "median": float(statistics.median(valid)) if valid else None,
            "maximum": max(valid) if valid else None,
            "scope": "finite available estimates only; fixed-denominator counts are separate"}


def summarize_records(records, collisions, complete=False):
    """Fixed planned denominators; every rejection and missing slot stays visible."""
    recover_names = ("author_matched_public32", "author_ideal_public64",
                     "reference_f64", "reference_f64_cast_f32")
    output = {"schema_version": 1,
              "study_complete": bool(complete and len(records) == 256
                                     and all(r["status"] == "completed" for r in records)),
              "planned_records": 256, "recorded_records": len(records),
              "completed_records": sum(r["status"] == "completed" for r in records),
              "error_records": sum(r["status"] == "error" for r in records),
              "missing_records": 256 - len(records), "cohorts": {}}
    for probe in PROBE_NAMES:
        rows = [r for r in records if r["probe"] == probe]
        available = {r["record_id"]: r for r in rows}
        if len(available) != len(rows):
            raise ValueError("duplicate_numerical_record")
        recovery = {}
        for name in recover_names:
            statuses, reasons, errors = Counter(), Counter(), []
            counts = {str(t): 0 for t in (1e-6, 1e-4, 1e-3, 1e-2)}
            for row in rows:
                result = row.get("recovery", {}).get(name)
                if result is None:
                    statuses["not_observed"] += 1
                    continue
                statuses[result["status"]] += 1
                if result["reason"]:
                    reasons[result["reason"]] += 1
                error = result["gram_error"]
                if error is not None:
                    errors.append(error.get("relative_frobenius"))
                for t in (1e-6, 1e-4, 1e-3, 1e-2):
                    counts[str(t)] += int(_finite_error(error, t))
            statuses["not_observed"] += 128 - len(rows)
            recovery[name] = {
                "planned_denominator": 128, "statuses": dict(statuses),
                "rejection_reasons": dict(reasons), "success_counts": counts,
                "finite_relative_errors": _error_summary(errors),
            }
        direct_errors = [r.get("direct_gram_error", {}).get("relative_frobenius") for r in rows]
        native = [r for r in rows if "extraction" in r]
        native_correct = sum(r["extraction"]["native_correct"] for r in native)
        native_pairs = 0
        complete_native_pairs = 0
        for f in range(64):
            pair = [available.get(f"F{f:03d}/{world}/{probe}") for world in WORLD_NAMES]
            if all(r is not None and "extraction" in r for r in pair):
                complete_native_pairs += 1
                native_pairs += int(all(r["extraction"]["native_correct"] for r in pair))
        factors = [v for r in rows for v in
                   r.get("source_geometry", {}).get("inverse_condition_factors_public32", [])]
        forced = [r["extraction"]["forced_log_probability_mean"] for r in native]
        output["cohorts"][probe] = {
            "planned_records": 128, "recorded": len(rows),
            "completed": sum(r["status"] == "completed" for r in rows),
            "missing": 128 - len(rows), "recovery": recovery,
            "direct_gram_relative_error": _error_summary(direct_errors),
            "source_whitening_inverse_condition_factors": _error_summary(factors),
            "forced_token_mean_log_probability": _error_summary(forced),
            "native_readout": {
                "observed_records": len(native), "correct_records": native_correct,
                "planned_record_denominator": 128,
                "accuracy": native_correct / 128 if len(native) == 128 else None,
                "observed_complete_pairs": complete_native_pairs,
                "both_correct_pairs": native_pairs, "planned_pair_denominator": 64,
                "paired_both_correct": native_pairs / 64 if complete_native_pairs == 64 else None,
                "readouts_across_probes_are_duplicates_not_independent": True,
            },
            "collisions": collision_summary([r for r in collisions if r["probe"] == probe]),
        }
    primary = output["cohorts"]["unique"]
    successes = primary["recovery"]["author_matched_public32"]["success_counts"]["0.001"]
    target_available = primary["recorded"] == 128 and primary["completed"] == 128
    output["primary_numerical_target"] = {
        "relative_frobenius_threshold": 1e-3, "required_successes": 122,
        "planned_denominator": 128, "successes": successes,
        "success_fraction_planned": successes / 128,
        "met": successes >= 122 if target_available else None,
        "available": target_available,
        "interpretation": "fixed 0.1% numerical reporting target only",
    }
    eligible = [r for r in records
                if r.get("validation", {}).get("f64_reference_eligible") is True]
    f64_failures = [r["record_id"] for r in eligible
                    if r["validation"].get("f64_reference_recovery_valid") is not True]
    eligible_unknown = [r["record_id"] for r in records if r["probe"] == "unique"
                        and "f64_reference_eligible" not in r.get("validation", {})]
    primary_missing = 128 - sum(r["probe"] == "unique" for r in records)
    repeats = [r for r in records if r["probe"] == "repeated"]
    first_checks = [r for r in repeats
                    if isinstance(r.get("validation", {}).get("first_step_contract"), bool)]
    public_checks = [r for r in records
                     if isinstance(r.get("validation", {}).get("public32_contract"), bool)]
    exact_failures = [r["pair_id"] for r in collisions if "exact_source_comparison_error" in r]
    expected_pair_ids = {f"F{f:03d}/{probe}" for f in range(64) for probe in PROBE_NAMES}
    observed_pair_counts = Counter(r["pair_id"] for r in collisions)
    exact_pair_ids = {r["pair_id"] for r in collisions if "exact_source_comparison" in r}
    missing_exact_ids = sorted(expected_pair_ids - exact_pair_ids)
    extra_pair_ids = sorted(set(observed_pair_counts) - expected_pair_ids)
    duplicate_pair_ids = sorted(k for k, n in observed_pair_counts.items() if n > 1)
    checks = {
        "all_256_records_completed": len(records) == 256 and output["error_records"] == 0,
        "f64_reference": {
            "relative_threshold": 1e-8, "eligible_count": len(eligible),
            "failed_ids": f64_failures, "eligibility_unknown_ids": eligible_unknown,
            "ineligible_count": sum(r["probe"] == "unique" and
                r.get("validation", {}).get("f64_reference_eligible") is False for r in records),
            "missing_primary_records": primary_missing,
            "passed": bool(eligible) and not f64_failures and not eligible_unknown and primary_missing == 0,
            "failure_can_be_numerical_conditioning_not_theorem_or_code_refutation": True,
        },
        "first_state_and_logits": {
            "expected_checks": 128, "observed": len(first_checks),
            "failed_ids": [r["record_id"] for r in first_checks
                           if r["validation"]["first_step_contract"] is False],
            "passed": len(first_checks) == 128 and all(
                r["validation"]["first_step_contract"] for r in first_checks),
        },
        "public32_contract": {
            "expected_checks": 256, "observed": len(public_checks),
            "failed_ids": [r["record_id"] for r in public_checks
                           if r["validation"]["public32_contract"] is False],
            "passed": len(public_checks) == 256 and all(
                r["validation"]["public32_contract"] for r in public_checks),
        },
        "exact_pair_arithmetic": {
            "expected_pairs": 128, "recorded_pairs": len(collisions),
            "failed_ids": exact_failures, "missing_exact_comparison_ids": missing_exact_ids,
            "extra_pair_ids": extra_pair_ids, "duplicate_pair_ids": duplicate_pair_ids,
            "passed": (len(collisions) == 128 and not exact_failures
                       and not missing_exact_ids and not extra_pair_ids and not duplicate_pair_ids),
        },
    }
    output["required_validation"] = checks
    output["required_validation_passed"] = (
        checks["all_256_records_completed"] and checks["f64_reference"]["passed"]
        and checks["first_state_and_logits"]["passed"]
        and checks["public32_contract"]["passed"] and checks["exact_pair_arithmetic"]["passed"])
    return output


def report_text(results):
    def number(value, digits=6):
        if not isinstance(value, (int, float)):
            return "unavailable"
        return f"{value:.{digits}g}"
    target = results["primary_numerical_target"]
    lines = [
        "# Contextual information recovery: original worker report",
        "",
        "This is a frozen-model numerical assay with known public reference tokens. "
        "It contains teacher-forced prediction-step states, not generated agent messages.",
        "",
        f"Recorded {results['recorded_records']}/256 numerical records; "
        f"{results['completed_records']} completed and {results['error_records']} operational errors.",
        f"Required validation passed: **{results['required_validation_passed']}**.",
        f"The fixed primary 0.1% recovery target: **{target['met']}** "
        f"({target['successes']}/128 successes; 122 required). Missing or unsupported "
        "records are not removed from the planned denominator.",
        "",
        "## Recovery from final outputs",
        "",
        "| Probe | Inverse/forward convention | <=1e-6 | <=1e-4 | <=1e-3 | <=1e-2 | Finite-error median | Finite-error maximum |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for probe in PROBE_NAMES:
        for name, data in results["cohorts"][probe]["recovery"].items():
            counts, error = data["success_counts"], data["finite_relative_errors"]
            lines.append(
                f"| {probe} | {name} | {counts['1e-06']}/128 | {counts['0.0001']}/128 | "
                f"{counts['0.001']}/128 | {counts['0.01']}/128 | "
                f"{number(error['median'])} | {number(error['maximum'])} |")
    lines.extend([
        "", "Error medians and maxima describe finite available estimates only; "
        "the success counts retain all 128 planned records in each cohort.",
        "",
        "## Exact finite-codebook collisions",
        "",
        "| Probe | Channel | Confirmed collisions | Confirmed distinct | Missing/invalid | Full ceiling | Full ceiling bounds |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for probe in PROBE_NAMES:
        coll = results["cohorts"][probe]["collisions"]
        for name, data in coll["channels"].items():
            bounds = data["full_codebook_ceiling_bounds"]
            lines.append(
                f"| {probe} | {name} | {data['confirmed_collisions']} | "
                f"{data['confirmed_distinct']} | {data['missing_or_invalid']} | "
                f"{number(data['full_codebook_ceiling'])} | "
                f"[{number(bounds[0])}, {number(bounds[1])}] |")
    for probe in PROBE_NAMES:
        coll = results["cohorts"][probe]["collisions"]
        lines.extend([
            "",
            f"{probe}: preexisting identical-numerical-source collision IDs: "
            + (", ".join(coll["preexisting_source_collision_ids"]) or "none observed") + ".",
            f"{probe}: alignment-created witness IDs: "
            + (", ".join(coll["alignment_created_witnesses"]) or "none observed") + ".",
            f"{probe}: exact centered-Gram-loss witness IDs: "
            + (", ".join(coll["exact_gram_loss_witnesses"]) or "none observed") + ".",
        ])
    lines.extend(["", "## Input regime", ""])
    for probe in PROBE_NAMES:
        cohort = results["cohorts"][probe]
        native = cohort["native_readout"]
        lines.append(
            f"- {probe}: native first-state color accuracy {number(native['accuracy'])}; "
            f"paired both correct {number(native['paired_both_correct'])}; "
            "the two probes' first-state readouts are duplicates. "
            f"Observed source-whitening inverse-condition-factor maximum "
            f"{number(cohort['source_whitening_inverse_condition_factors']['maximum'])}.")
    lines.extend([
        "", "## Interpretation and scope", "",
        "An exact answer-changing collision with different source values establishes "
        "a limitation on that retained pair under its identical public conditions. "
        "The ceiling is a finite paired-codebook statement, not population accuracy.",
        "",
        "An absent collision does not establish useful semantic transfer. Failed "
        "inversion or small distance does not prove absence of information. Positive "
        "regularization permits exact recovery under the stated conditions while "
        "the numerical inverse can be poorly conditioned.",
        "",
        "The actual author float32 return is distinguished from the NumPy float64 "
        "mathematical reference and the latter's single float32 cast. Different "
        "nonunique repeated-ID completions are not prefix-equality failures. "
        "Float64 validation failure may itself reflect conditioning.",
        "",
        "The forced tokens may be outside the original generation protocol's top-p "
        "support. There is no trained adapter, receiver generation, timing-performance "
        "claim, or established novelty claim.",
        "",
        "All private truths and captured anchor choices are used only after the "
        "public-input inverse returns. Raw record and array manifests retain "
        "rejections, numerical residuals, exact comparisons and partial evidence.",
        "",
    ])
    return "\n".join(lines)


def run_study(output):
    """Called only by the reviewed supervisor; the parent owns the hard deadline."""
    output = Path(output)
    beginning = time.perf_counter()
    identity = verify_execution(output)
    cohort = build_cohort()
    plan = build_plan(cohort)
    write_json(output / "plan.json", plan)
    write_json(output / "cohort.json", cohort)
    write_json(output / "source_manifest.json", identity)
    write_json(output / "status.json", {"status": "initializing", "recorded_records": 0})
    records, collisions, first_states, pair_cache = [], [], {}, {}
    writer = EvidenceWriter(output)
    records_file = (output / "records.jsonl").open("a", encoding="utf-8")
    collisions_file = (output / "collisions.jsonl").open("a", encoding="utf-8")
    phase_times = {}
    final_results = None
    try:
        runtime = initialize_runtime()
        write_json(output / "runtime_manifest.json", runtime)
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        author_function, author_metadata = load_author_method()
        write_json(output / "author_method_manifest.json", author_metadata)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="contextual-model-", dir=output.parent) as model_directory:
            model_manifest = download_model(model_directory, output)
            phase_times["download_and_hash_verification"] = time.perf_counter() - started
            started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(
                model_directory, local_files_only=True, trust_remote_code=False)
            tokenization = tokenize_cohort(tokenizer, cohort)
            write_json(output / "tokenization.json", tokenization)
            model = AutoModelForCausalLM.from_pretrained(
                model_directory, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation="eager")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if model.config.hidden_size != 128 or model.config._attn_implementation != "eager":
                raise PreflightError("model_config_width_or_attention")
            phase_times["tokenizer_model_loading_and_preflight"] = time.perf_counter() - started
            started = time.perf_counter()
            public, vocab32, vocab64, public_manifest = prepare_public_contexts(
                model, tokenization, writer, output)
            phase_times["public_setup_and_retention"] = time.perf_counter() - started
            author_context = SimpleNamespace(
                embedding_layer=model.get_input_embeddings(), device=torch.device("cpu"),
                dtype=torch.float32, vocab_embeds=model.get_input_embeddings().weight.detach().float(),
                target_norm=public["unique"]["parameters32"]["target_norm"],
                adaptive_reg=1e-3, snap_ratio=0.3)
            if author_context.target_norm != public["repeated"]["parameters32"]["target_norm"]:
                raise RuntimeError("public_target_norm_changed")
            families = {f["family_id"]: f for f in cohort["families"]}
            for planned in plan["planned_records"]:
                family = families[planned["family_id"]]
                print("Record " + planned["record_id"] + " started", flush=True)
                row = process_record(
                    planned, family, tokenization, model, author_function, author_context,
                    public, vocab64, writer, first_states, pair_cache)
                append_jsonl(records_file, row)
                records.append(row)
                if planned["world"] == "swapped":
                    key = (planned["family_id"], planned["probe"])
                    paired = analyze_pair(
                        family, planned["probe"], pair_cache.pop(key),
                        public[planned["probe"]]["arrays32"]["reference"])
                    append_jsonl(collisions_file, paired)
                    collisions.append(paired)
                snapshot = summarize_records(records, collisions, complete=False)
                snapshot.update({
                    "provenance": identity, "scientific_wall_seconds_so_far": time.perf_counter() - beginning,
                    "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                    "phase_seconds": phase_times, "empirical_retries": 0,
                    "native_model_training": False, "receiver_generation": False,
                })
                write_json(output / "results.json", snapshot)
                write_json(output / "status.json", {
                    "status": "running", "recorded_records": len(records),
                    "completed_records": snapshot["completed_records"],
                    "last_record_id": row["record_id"], "last_record_status": row["status"],
                })
                print("Record " + row["record_id"] + ": " + row["status"], flush=True)
                if row["status"] == "error":
                    raise RuntimeError("Stopped after retained operational record error: " + row["record_id"])
            final_results = summarize_records(records, collisions, complete=True)
            final_results.update({
                "provenance": identity, "scientific_wall_seconds": time.perf_counter() - beginning,
                "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                "phase_seconds": phase_times, "empirical_retries": 0,
                "native_model_training": False, "receiver_generation": False,
                "retained_complete_extraction_forward_calls": sum(
                    r.get("extraction", {}).get("model_forward_calls", 0) for r in records),
                "forward_call_count_is_exact": all("extraction" in r for r in records),
                "expected_model_forward_calls": 2048,
                "public_contexts_prepared_before_private_extraction": True,
            })
        # Temporary model-file removal is part of this scientific child.
        final_results["scientific_wall_seconds"] = time.perf_counter() - beginning
        final_results["worker_wall_time_scope"] = (
            "Elapsed before final summary/status writes and process teardown; "
            "supervisor execution.json records the complete scientific-child duration.")
        write_json(output / "results.json", final_results)
        (output / "report.md").write_text(report_text(final_results), encoding="utf-8")
        write_json(output / "status.json", {
            "status": "completed" if final_results["required_validation_passed"] else "completed_with_validation_failure",
            "recorded_records": len(records), "completed_records": final_results["completed_records"],
            "required_validation_passed": final_results["required_validation_passed"],
            "primary_numerical_target_met": final_results["primary_numerical_target"]["met"],
        })
        print("Contextual assay complete: " + json.dumps({
            "records": len(records), "validation": final_results["required_validation_passed"],
            "primary_target": final_results["primary_numerical_target"],
            "wall_seconds": final_results["scientific_wall_seconds"]}, sort_keys=True), flush=True)
        return 0 if final_results["required_validation_passed"] else 1
    except Exception as error:
        failure = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=20),
            "recorded_records": len(records), "scientific_wall_seconds": time.perf_counter() - beginning,
        }
        write_json(output / "worker_failure.json", failure)
        snapshot = summarize_records(records, collisions, complete=False)
        snapshot.update({"provenance": identity, "failure": failure,
                         "phase_seconds": phase_times, "empirical_retries": 0})
        write_json(output / "results.json", snapshot)
        (output / "report.md").write_text(report_text(snapshot), encoding="utf-8")
        write_json(output / "status.json", {
            "status": "unsupported_preflight" if isinstance(error, PreflightError) else "failed",
            "recorded_records": len(records), "failure": failure})
        raise
    finally:
        records_file.close()
        collisions_file.close()
        writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_study(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
