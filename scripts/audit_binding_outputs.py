"""Read-only evidence audit of one extracted, frozen binding-study artifact.

No model imports, training, inference, tokenizer calls, or study helpers are
used. All labels, quartet metrics, whole-family bootstrap draws, gates, order
hashes, and NPZ tensor digests are rebuilt here. The receipt must be outside
the input tree. An integrity pass on partial evidence is not study completion.

Usage:
  python scripts/audit_binding_outputs.py \
    --artifact /path/to/checkpoints/binding-study \
    --receipt /path/outside/artifact/binding_audit.json \
    --source-root /path/to/research3
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time

import numpy as np

DESIGN_COMMIT = "dfce06b5dfb2860414af166ef4441eefd9548afc"
HELPER_COMMIT = "9eb4478fae1ad373234eb2a725345b66c8c63c57"
FROZEN_BLOBS = {
    "docs/binding_design.md": "c07f9d384e0dd607d482218a8a93a4e71f52e470",
    "latent_bridge/binding_data.py": "2b5ecaa8ea1f302ccd269cef1e76105e80e725eb",
    "latent_bridge/binding_core.py": "118001d3c50724ec8c6d6c75877ac20ea8308bff",
    "latent_bridge/binding_study.py": "025fbb2eca2d8fa11aa2e2892cf8de9feb964e31",
}
OBJECTS = ("apple", "book", "cup", "lamp", "key", "plate", "shoe", "ball")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "purple", "orange")
ROLES = ("O", "T", "N", "R")
SPLITS = {"train": 64, "validation": 32, "test": 64}
SEEDS = (101, 202, 303)
ARMS = ("aware_ce", "aware_aux", "blind_ce", "private_constant", "text", "symbolic")
INTERVENTIONS = ("wrong_source", "wrong_question", "zero")
CONDITIONS = ARMS + tuple(a + "_" + c for a in ("aware_ce", "aware_aux") for c in INTERVENTIONS) + (
    "frozen_no_channel", "rank_codec_oracle")
FIT_ORDER = [(s, "text") for s in SEEDS] + [(s, a) for s in SEEDS for a in ARMS if a != "text"]
FRESH_ORDER = [(s, c) for s in SEEDS for c in CONDITIONS]
SCHEMA = "eight-object-bijective-binding-v1"
PACKAGES = {"torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
            "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2"}
MODELS = {
    "sender": {"id": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa"},
    "receiver": {"id": "EleutherAI/pythia-14m", "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac"}}
CONFIG = {
    "seeds": list(SEEDS), "epochs": 8, "quartets_per_batch": 4,
    "records_per_batch": 16, "slots": 2, "bottleneck": 8, "max_length": 192,
    "learning_rate": .001, "weight_decay": .01, "gradient_clip_norm": 1.,
    "auxiliary_weight": 1., "margin": .25, "message_coordinates": 16,
    "receiver_separator": "\n", "query_normalization_epsilon": 1e-5,
    "private_control_seed_offset": 9841, "fixed_question_seed_offset": 17291,
    "bootstrap_seed": 66173, "bootstrap_resamples": 2000, "threads": 2}
STATE_SHAPES = {
    "queries": (2, 768), "gates": (2,),
    "compressor.0.weight": (768,), "compressor.0.bias": (768,),
    "compressor.1.weight": (8, 768), "compressor.1.bias": (8,),
    "expander.weight": (128, 8), "expander.bias": (128,)}
LIMITS = [
    "Choice correctness, labels, CE averages and statistics are independently recomputed; logits and per-record CE are retained model outputs, not independently reproduced inference.",
    "Raw per-example messages and frozen feature tensors are not retained. Distance/loss summaries are checked for bounds and algebraic consistency, not rederived from activations.",
    "NPZ tensor contents and hashes are checked, but seeded PyTorch random tensors and actual optimizer updates are not re-executed. Shared initialization/update chronology are evidence checks.",
    "Tokenizer-ID bag checks, downloaded model weights and cache-content hashes are recorded preflight claims. This numpy/stdlib auditor does not retokenize or download model files.",
    "Selection-before-test chronology is checked through manifests, completed-fit/order evidence and source hashes; file contents alone cannot independently prove wall-clock execution order.",
    "Bootstrap intervals are descriptive over the fixed families conditional on three fixed model seeds. Integrity and any engineering gates imply no novelty or breakthrough claim.",
]


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def metadata(path):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def bad_number(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_number)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def equivalent(a, b, tolerance=1e-10):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equivalent(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equivalent(x, y, tolerance) for x, y in zip(a, b))
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    if finite(a) and finite(b):
        return abs(a - b) <= tolerance * max(1., abs(a), abs(b))
    return type(a) is type(b) and a == b


def safe_path(root, name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or p.is_absolute() or not p.parts or ".." in p.parts or "\\" in name:
        raise ValueError("unsafe relative artifact path")
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes input tree")
    return result


def swap(world, i, j):
    result = list(world)
    result[i], result[j] = result[j], result[i]
    return tuple(result)


def reservation(world):
    return {world} | {swap(world, i, j) for i in range(8) for j in range(i + 1, 8)}


def world_id(world):
    return sha(compact({"schema": SCHEMA, "objects": OBJECTS, "colors": COLORS, "world": world}))


def render(world, order, symbolic=False):
    return "".join((OBJECTS[i] + "=" + COLORS[world[i]] + ";\n") if symbolic else
                   ("The " + OBJECTS[i] + " is " + COLORS[world[i]] + ".\n") for i in order)


def frozen_data():
    """Independent implementation from the public finite-world specification."""
    candidates = list(itertools.permutations(range(8)))
    random.Random(20261127).shuffle(candidates)
    bases, occupied, examined = [], set(), 0
    for examined, base in enumerate(candidates, 1):
        ball = reservation(base)
        if not ball.intersection(occupied):
            bases.append(base)
            occupied |= ball
            if len(bases) == 160:
                break
    if len(bases) != 160 or len(occupied) != 4640 or examined != 340:
        raise ValueError("independent fixed-cohort feasibility differs")
    # Anchors came from a separate JavaScript MT19937 implementation before fits.
    anchors = {0: "45103726", 63: "51607324", 64: "26014357", 95: "50317642",
               96: "23501764", 159: "52147063"}
    if any("".join(map(str, bases[i])) != text for i, text in anchors.items()):
        raise ValueError("independent frozen-cohort anchors differ")
    orders, edits = random.Random(20261128), random.Random(20261129)
    partitions, families = {s: [] for s in SPLITS}, []
    source_hashes, frequencies = {}, {}
    for number, base in enumerate(bases):
        split = "train" if number < 64 else "validation" if number < 96 else "test"
        family = "binding-family-" + str(number).zfill(3)
        order = list(range(8))
        orders.shuffle(order)
        reorder = list(range(8))
        orders.shuffle(reorder)
        while reorder == order:
            reorder = list(range(8))
            orders.shuffle(reorder)
        donors = list(range(8))
        edits.shuffle(donors)
        while any(i == donor for i, donor in enumerate(donors)):
            donors = list(range(8))
            edits.shuffle(donors)
        pairs = [sorted(edits.sample([j for j in range(8) if j != i], 2)) for i in range(8)]
        ball = sorted(reservation(base))
        families.append({"family": family, "partition": split, "base_world": list(base),
            "base_world_id": world_id(base), "order": order, "reorder": reorder,
            "donors": donors, "nuisance_pairs": pairs,
            "reserved_worlds": [list(w) for w in ball],
            "reserved_world_ids": [world_id(w) for w in ball]})
        for qi in range(8):
            worlds = (base, swap(base, qi, donors[qi]), swap(base, *pairs[qi]), base)
            members = []
            for role, world in zip(ROLES, worlds):
                view = reorder if role == "R" else order
                english = render(world, view)
                members.append({"variant": role, "world": list(world), "world_id": world_id(world),
                    "order": list(view), "source": english, "symbolic": render(world, view, True),
                    "answer": COLORS[world[qi]]})
                source_hashes[world_id(world) + ":" + ",".join(map(str, view))] = sha(english.encode())
            partitions[split].append({"family": family, "query_index": qi,
                "query": "The color of the " + OBJECTS[qi] + " is", "partner": donors[qi],
                "nuisance_pair": pairs[qi], "members": members})
    for split, rows in partitions.items():
        frequency = {str(q): {r: [0] * 8 for r in ROLES} for q in range(8)}
        for row in rows:
            for member in row["members"]:
                frequency[str(row["query_index"])][member["variant"]][COLORS.index(member["answer"])] += 1
        frequencies[split] = frequency
    audit = {"family_counts": dict(SPLITS),
        "quartet_counts": {s: n * 8 for s, n in SPLITS.items()},
        "record_counts": {s: n * 32 for s, n in SPLITS.items()},
        "reserved_world_count": 4640, "answer_frequencies_by_query_member": frequencies,
        "unique_source_order_identities": len(source_hashes),
        "source_identity_sha256": sha(compact(source_hashes))}
    return partitions, families, occupied, audit


def tensor_hash(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode() + b"\0")
        h.update(str(a.dtype).encode() + b"\0")
        h.update(json.dumps(list(a.shape)).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def order_hashes(seed):
    result = []
    for epoch in range(1, 9):
        order = list(range(512))
        random.Random(seed + 1009 * epoch).shuffle(order)
        result.append(sha(json.dumps(order, separators=(",", ":")).encode()))
    return result


def summary_stats(values):
    values = list(values)
    return {"count": len(values), "min": min(values), "mean": float(np.mean(values)),
            "max": max(values), "sum": sum(values)}


def source_path_consistency(base_rows, substituted_rows):
    """Compare an intervention with the identical retained donor computation."""
    lookup = {(r["seed"], r["family"], r["query_index"], r["variant"]): r for r in base_rows}
    counts = {"matched_rows": 0, "unmatched_rows": 0, "same_label_ce_rows": 0,
              "choice_mismatches": 0, "source_label_mismatches": 0, "same_label_ce_mismatches": 0}
    for row in substituted_rows:
        key = (row["seed"], row["source_family"], row["query_index"], row["variant"])
        supplied = lookup.get(key)
        if supplied is None:
            counts["unmatched_rows"] += 1
            continue
        counts["matched_rows"] += 1
        counts["choice_mismatches"] += row["choice"] != supplied["choice"]
        counts["source_label_mismatches"] += row["source_gold"] != supplied["gold"]
        if row["gold"] == row["source_gold"]:
            counts["same_label_ce_rows"] += 1
            counts["same_label_ce_mismatches"] += not equivalent(
                row["restricted_ce"], supplied["restricted_ce"], tolerance=2e-7)
    return counts


class AuditFailure(Exception):
    pass


class Audit:
    def __init__(self, root, source_root=None):
        self.root, self.source_root = root, source_root
        self.checks, self.error_count, self.errors, self.notes = 0, 0, [], []
        self.evidence, self.stage_status = {}, {}
        self.inventory, self.jsonl_tails, self.prediction_blocks = {}, {}, {}
        self.result = self.read("results.json") or {}
        self.status = self.read("status.json") or {}
        self.execution = self.read("execution.json") or {}
        self.full = self.result.get("execution_status") == "executed_binding_competence_study"
        self.stopped = self.result.get("execution_status") == "receiver_limited_binding_study"
        self.terminal = self.full or self.stopped
        self.fits = self.read("fits.json") or []
        self.partitions, self.families, self.reserved, self.expected_data_audit = frozen_data()
        self.cases = {
            s: [(q["family"], q["query_index"], m["variant"], m["world"][q["query_index"]], m, q["query"])
                for q in rows for m in q["members"]]
            for s, rows in self.partitions.items()}
        self.validation, self.fresh, self.initial, self.gate = {}, {}, {}, None
        self.history_rows, self.selection_verified = [], False

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.error_count += 1
            if len(self.errors) < 100:
                self.errors.append(label)
        return bool(condition)

    def need(self, condition, label):
        if not self.check(condition, label):
            raise AuditFailure(label)

    def same(self, actual, expected, label, tolerance=1e-10):
        return self.check(equivalent(actual, expected, tolerance), label)

    def step(self, name, function):
        before = self.error_count
        try:
            function()
        except AuditFailure:
            pass
        except Exception as error:
            self.check(False, name + ": " + type(error).__name__ + ": " + str(error))
        self.stage_status[name] = "passed" if self.error_count == before else "failed"

    def read(self, name, required=False):
        path = safe_path(self.root, name)
        if not path.is_file():
            if required:
                self.need(False, "missing " + name)
            return None
        return strict_json(path.read_bytes())

    def rows(self, name, required=False):
        path = safe_path(self.root, name)
        if not path.is_file():
            if required:
                self.need(False, "missing " + name)
            return []
        raw, rows = path.read_bytes(), []
        segments = raw.split(b"\n")
        tail = segments.pop()
        self.jsonl_tails[name] = len(tail)
        self.check(not self.terminal or not tail, name + ": terminal result has unterminated raw line")
        if tail:
            self.notes.append(name + ": retained " + str(len(tail)) + " trailing bytes; not treated as a complete row")
        for number, line in enumerate(segments, 1):
            value = strict_json(line)
            self.check(compact(value) == line, name + ": noncanonical JSONL row " + str(number))
            rows.append(value)
        return rows

    def arrays(self, name):
        path = safe_path(self.root, name)
        self.need(path.is_file(), "missing tensor archive " + name)
        with np.load(path, allow_pickle=False) as archive:
            self.need(len(archive.files) == len(set(archive.files)), "duplicate NPZ keys in " + name)
            return {key: np.array(archive[key], copy=True) for key in archive.files}

    def checkpoint(self, record, expected_name, initial=False):
        self.need(record.get("path") == expected_name, "checkpoint path differs: " + expected_name)
        path = safe_path(self.root, expected_name)
        actual = metadata(path)
        self.same({k: record.get(k) for k in ("sha256", "bytes")}, actual, "checkpoint file digest " + expected_name)
        arrays = self.arrays(expected_name)
        self.need(set(arrays) == set(STATE_SHAPES), "checkpoint tensor names " + expected_name)
        for name, shape in STATE_SHAPES.items():
            self.check(arrays[name].shape == shape and arrays[name].dtype == np.dtype("float32")
                       and bool(np.isfinite(arrays[name]).all()), "tensor shape/dtype/finiteness " + expected_name + "/" + name)
        digest = tensor_hash(arrays)
        self.check(record.get("state_sha256") == digest, "checkpoint state digest " + expected_name)
        self.check(sum(v.size for v in arrays.values()) == 10378, "bridge parameter count " + expected_name)
        if initial:
            self.check(bool((arrays["gates"] == 0).all()), "initial gates must be exactly zero")
            self.check(bool((arrays["compressor.0.weight"] == 1).all())
                       and bool((arrays["compressor.0.bias"] == 0).all()), "initial LayerNorm parameters")
        return arrays, digest

    def retention(self):
        for path in sorted(self.root.rglob("*")):
            self.need(not path.is_symlink(), "symlink is outside this artifact audit's file contract")
            if path.is_file():
                self.inventory[path.relative_to(self.root).as_posix()] = metadata(path)
        self.need(bool(self.inventory), "empty artifact directory")
        expected = self.execution.get("retained_files")
        if expected is None:
            self.check(not self.terminal, "terminal outcome is missing wrapper retention index")
            self.notes.append("Wrapper retention index absent; inventory was hashed, but there is no independent retained-file list.")
        else:
            for name in expected:
                safe_path(self.root, name)
            actual = {k: v for k, v in self.inventory.items() if PurePosixPath(k).name != "execution.json"}
            self.same(expected, actual, "all retained file names, byte counts and SHA256 digests")
        receipts = self.execution.get("prediction_receipts", {})
        for name in ("predictions.jsonl", "validation_predictions.jsonl"):
            path, compressed = self.root / name, self.root / (name + ".gz")
            if not path.is_file():
                self.check(not compressed.exists(), "gzip exists without its raw file: " + name)
                self.check(name not in receipts, "compression receipt exists without raw file: " + name)
                continue
            if not compressed.is_file():
                self.need(not self.terminal and name not in receipts,
                          "missing preserved or claimed gzip: " + name)
                self.notes.append(name + ": raw evidence retained without gzip; parent compression/finalization is incomplete.")
                continue
            raw, zipped = path.read_bytes(), compressed.read_bytes()
            self.check(gzip.decompress(zipped) == raw, "exact gzip/raw byte equivalence: " + name)
            self.check(zipped[:3] == b"\x1f\x8b\x08" and zipped[4:8] == b"\0" * 4
                       and not (zipped[3] & 8), "deterministic no-filename/mtime gzip header: " + name)
            if name in receipts:
                self.same(receipts[name], {"raw": self.inventory[name],
                    "gzip": self.inventory[name + ".gz"], "round_trip_verified": True},
                    "wrapper gzip receipt: " + name)
            else:
                self.check(not self.terminal, "terminal wrapper gzip receipt missing: " + name)
        if self.execution:
            self.check(self.execution.get("frozen_design_commit") == DESIGN_COMMIT, "wrapper frozen design commit")
            self.check(str(self.execution.get("github_run_attempt")) == "1"
                       and str(self.execution.get("github_run_number")) == "1", "single authorized empirical run/attempt")
            self.check(self.execution.get("budget_seconds") == 1800, "unchanged study process budget")
            self.same(self.execution.get("study_status"), self.status or None, "wrapper copied final scientific status")
            for name in ("study_wall_seconds_including_downloads", "wrapper_wall_seconds_including_retention"):
                self.check(finite(self.execution.get(name)) and self.execution[name] >= 0, "finite wrapper timing: " + name)
            if self.terminal:
                self.check(self.execution.get("status") == "worker_finished"
                           and self.execution.get("worker_returncode") == 0
                           and self.execution.get("failure") is None, "terminal scientific result has successful wrapper")
        self.evidence["retention"] = {
            "files": len(self.inventory), "bytes": sum(v["bytes"] for v in self.inventory.values()),
            "execution_index_present": expected is not None, "inventory": self.inventory}

    def provenance(self):
        code = self.read("code_manifest.json", self.terminal)
        runtime = self.read("runtime_manifest.json", self.terminal)
        models = self.read("model_manifest.json", self.terminal)
        if code is not None:
            self.check(code.get("frozen_design_commit") == DESIGN_COMMIT
                       and code.get("historical_helper_commit") == HELPER_COMMIT
                       and code.get("all_match") is True, "immutable design/helper provenance")
            self.check(code.get("design_sha256") == code.get("expected_design_sha256"), "declared expected design bytes")
            self.need(set(FROZEN_BLOBS).issubset(code.get("files", {})), "required scientific source entries")
            for name, entry in code.get("historical_helpers", {}).items():
                self.check(entry.get("expected_sha256") == entry.get("actual_sha256"), "historical helper digest " + name)
            if self.source_root is not None:
                for name, expected in code["files"].items():
                    path = safe_path(self.source_root, name)
                    self.same(metadata(path), expected, "checked-out source hash " + name)
                for name, expected_blob in FROZEN_BLOBS.items():
                    raw = safe_path(self.source_root, name).read_bytes()
                    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                    self.check(git_blob == expected_blob, "independently pinned scientific Git blob " + name)
                self.check(metadata(self.source_root / "docs/binding_design.md")["sha256"] == code["design_sha256"],
                           "frozen design SHA256")
            else:
                self.notes.append("No --source-root supplied: code hashes checked for manifest consistency only, not against source bytes.")
            if self.execution:
                self.check(code.get("github_commit") == self.execution.get("github_commit")
                           and str(code.get("github_run_id")) == str(self.execution.get("github_run_id")),
                           "wrapper/source run identity")
        if runtime is not None:
            self.check(runtime.get("python") == runtime.get("expected_python") == "3.11.18", "frozen Python version")
            self.same(runtime.get("actual_packages"), PACKAGES, "actual package versions")
            self.same(runtime.get("expected_packages"), PACKAGES, "expected package versions")
            if "packages" in runtime:
                self.same(runtime["packages"], PACKAGES, "runner package versions")
            self.check(runtime.get("torch_threads") == 2, "frozen CPU thread count")
            self.check(isinstance(runtime.get("platform"), str) and bool(runtime["platform"]), "host platform retained")
            if code:
                self.check(runtime.get("github_commit") == code.get("github_commit")
                           and str(runtime.get("github_run_id")) == str(code.get("github_run_id")), "runtime/source run identity")
        if models is not None:
            self.check(models.get("remote_code_allowed") is False and models.get("hub_credentials_used") is False,
                       "public pinned model download contract")
            if self.terminal:
                self.check(models.get("status") == "downloaded_pinned_public_pretrained_models", "completed model manifest")
                self.need(set(models.get("models", {})) == set(MODELS), "both model provenance entries")
            for role, entry in models.get("models", {}).items():
                self.need(role in MODELS, "unknown model role")
                self.same({k: entry.get(k) for k in ("id", "revision")}, MODELS[role], "pinned model " + role)
                self.check("model.safetensors" in entry.get("files", {}), "safetensors model recorded " + role)
                for name, m in entry.get("files", {}).items():
                    self.check(type(m.get("bytes")) is int and m["bytes"] > 0
                               and isinstance(m.get("sha256"), str) and len(m["sha256"]) == 64,
                               "model file digest shape " + role + "/" + name)
            if code:
                self.check(models.get("design_sha256") == code.get("design_sha256")
                           and models.get("github_commit") == code.get("github_commit")
                           and str(models.get("github_run_id")) == str(code.get("github_run_id")), "model/source identity")
        for key, value in (("code_manifest", code), ("model_manifest", models), ("runtime", runtime)):
            if key in self.result:
                self.same(self.result[key], value, "result copies " + key)
        for value, label in ((self.result, "result"), (self.status, "status")):
            if "config" in value:
                self.same(value["config"], CONFIG, label + " frozen configuration")
        self.evidence["provenance"] = {"scientific_source_rehashed": code is not None and self.source_root is not None,
            "source_commit": None if code is None else code.get("github_commit"),
            "run_id": None if code is None else code.get("github_run_id"),
            "runtime": runtime, "frozen_design_commit": DESIGN_COMMIT}

    def dataset(self):
        manifest = self.read("data/manifest.json", self.terminal)
        copied = self.read("dataset_manifest.json", self.terminal)
        if manifest is None:
            self.notes.append("Dataset was not fully generated before the recorded interruption.")
            return
        for name, expected in (
            ("schema", SCHEMA), ("data_seed", 20261127), ("order_seed", 20261128),
            ("edit_seed", 20261129), ("objects", list(OBJECTS)), ("colors", list(COLORS)),
            ("members", list(ROLES)), ("split_family_counts", SPLITS),
            ("reserved_world_count", 4640), ("families", self.families), ("audit", self.expected_data_audit),
            ("reserved_id_set_sha256", sha(compact(sorted(world_id(w) for w in self.reserved))))):
            self.same(manifest.get(name), expected, "independently regenerated dataset manifest " + name)
        self.check(self.expected_data_audit["unique_source_order_identities"] == 2340, "independent source/view count")
        for split, expected_rows in self.partitions.items():
            name = "data/" + split + ".jsonl"
            path = self.root / name
            if not path.exists():
                self.check(not self.terminal, "missing terminal dataset partition " + split)
                continue
            raw = path.read_bytes()
            expected_bytes = b"".join(compact(row) + b"\n" for row in expected_rows)
            self.check(raw == expected_bytes, "exact independent frozen JSONL bytes " + split)
            self.same(manifest.get("files", {}).get(split + ".jsonl"), metadata(path), "dataset file metadata " + split)
        if copied is not None:
            self.same(copied, manifest, "copied full dataset manifest")
        if "data_manifest_sha256" in self.result:
            self.check(self.result["data_manifest_sha256"] == metadata(self.root / "dataset_manifest.json")["sha256"],
                       "result dataset manifest byte hash")
        self.evidence["data"] = {
            "accepted_families": 160, "examined_shuffled_candidates": 340, "reserved_worlds": 4640,
            "families_per_split": SPLITS, "source_order_identities": 2340,
            "all_partition_labels_and_renderings_independently_regenerated": all(
                (self.root / "data" / (s + ".jsonl")).is_file() for s in SPLITS)}

    def initial_states(self):
        initial = self.read("initialization_manifest.json", self.terminal)
        if initial is None:
            return
        self.need(set(initial) == {str(s) for s in SEEDS}, "three seed initialization records")
        controls = self.arrays("control_tensors.npz")
        expected_names = {f"seed{s}__{name}" for s in SEEDS for name in ("hidden", "mask", "question")}
        self.need(set(controls) == expected_names, "control tensor keys")
        digests = []
        for seed in SEEDS:
            record = initial[str(seed)]
            arrays, digest = self.checkpoint(record["checkpoint"], f"initial/{seed}.npz", initial=True)
            self.check(record.get("state_sha256") == digest, "initial selected state digest")
            self.check(record.get("trainable_parameters") == 10378, "initial parameter budget")
            self.same(record.get("epoch_order_sha256"), order_hashes(seed), "independent initial epoch orders")
            c = {name: controls[f"seed{seed}__{name}"] for name in ("hidden", "mask", "question")}
            for name, shape in (("hidden", (1, 8, 768)), ("mask", (1, 8)), ("question", (768,))):
                self.check(c[name].shape == shape and c[name].dtype == np.dtype("float32")
                           and bool(np.isfinite(c[name]).all()), "control tensor shape/dtype " + name)
            self.check(bool((c["mask"] == 1).all()) and bool(np.any(c["hidden"] != 0)),
                       "fixed nonzero private control and all-valid mask")
            self.check(abs(float(c["question"].mean())) < 1e-5
                       and .99 < float(c["question"].var()) <= 1.00001, "fixed normalized public-question control")
            self.check(record.get("control_state_sha256") == tensor_hash(c), "complete control state digest")
            self.check(record.get("fixed_private_sha256") == tensor_hash({k: c[k] for k in ("hidden", "mask")}),
                       "private control digest")
            self.check(record.get("fixed_question_sha256") == tensor_hash({"question": c["question"]}),
                       "question control digest")
            self.same(record.get("fixed_question_norm"), float(np.linalg.norm(c["question"])),
                      "question norm from actual NPZ", tolerance=2e-6)
            self.initial[seed] = digest
            digests.append(digest)
        self.check(len(set(digests)) == 3, "distinct initialization for the three declared seeds")
        self.evidence["initialization"] = {"state_hashes": {str(k): v for k, v in self.initial.items()},
                                         "parameters_per_arm": 10378}

    def loss_terms(self, terms, arm, label):
        needed = {"nuisance_distance", "reorder_distance", "target_distance",
                  "restricted_ce", "hinge", "auxiliary", "objective"}
        self.need(set(terms) == needed, label + ": exact logged loss-term fields")
        self.need(all(finite(terms[k]) for k in needed), label + ": finite terms")
        for name in ("nuisance_distance", "reorder_distance", "target_distance"):
            self.check(-1e-7 <= terms[name] <= 4.000001, label + ": bounded raw distance " + name)
        self.check(terms["restricted_ce"] >= -1e-7 and -1e-7 <= terms["hinge"] <= .250001,
                   label + ": CE and hinge bounds")
        self.check(terms["hinge"] >= max(0., .25 - terms["target_distance"]) - 2e-6,
                   label + ": averaged hinge respects convexity")
        aux = .5 * (terms["nuisance_distance"] + terms["reorder_distance"]) + terms["hinge"]
        self.same(terms["auxiliary"], aux, label + ": auxiliary identity", tolerance=2e-6)
        self.same(terms["objective"], terms["restricted_ce"] + (aux if arm == "aware_aux" else 0.),
                  label + ": objective identity", tolerance=2e-6)

    def histories(self):
        rows = self.rows("history.jsonl", self.terminal)
        self.history_rows = rows
        expected_sequence = [(s, a, e) for s, a in FIT_ORDER for e in range(1, 9)]
        self.check(len(rows) <= len(expected_sequence), "history cannot exceed frozen total epochs")
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            key = (row.get("seed"), row.get("arm"), row.get("epoch"))
            self.check(i < len(expected_sequence) and key == expected_sequence[i], "fixed fit/epoch history chronology " + str(i))
            self.check(row.get("updates") == 128 * row["epoch"], "cumulative updates at epoch " + str(i))
            self.need(finite(row.get("validation_ce")) and row["validation_ce"] >= 0, "finite validation CE")
            self.same(row.get("validation_ce"), row["validation"].get("restricted_ce"), "history validation selection CE")
            self.loss_terms(row["training"], row["arm"], "training epoch " + str(i))
            self.loss_terms(row["validation"], row["arm"], "validation epoch " + str(i))
            self.check(len(row.get("query_gates", [])) == 2 and all(finite(v) for v in row["query_gates"]),
                       "history query gate values")
            for name in ("training_seconds", "validation_seconds"):
                self.check(finite(row.get(name)) and row[name] >= 0, "nonnegative epoch timing")
            grouped[(row["seed"], row["arm"])].append(row)
        self.need(isinstance(self.fits, list), "fits.json must be a list")
        self.check(len(self.fits) <= 18, "completed fit count cannot exceed budget")
        for index, fit in enumerate(self.fits):
            key = (fit.get("seed"), fit.get("arm"))
            self.check(index < len(FIT_ORDER) and key == FIT_ORDER[index], "completed fit order")
            self.check(fit.get("complete") is True and fit.get("epochs") == 8 and fit.get("updates") == 1024,
                       "exact completed fit budget")
            history = grouped[key]
            self.need(len(history) == 8, "completed fit retains all eight epochs")
            self.same(fit.get("history"), history, "fit history exactly matches append-only raw history")
            selected = min(range(8), key=lambda i: (history[i]["validation_ce"], history[i]["epoch"]))
            self.check(fit.get("selected_epoch") == selected + 1, "earliest minimum validation-CE epoch")
            self.same(fit.get("epoch_order_sha256"), order_hashes(key[0]), "independently regenerated quartet orders")
            self.check(fit.get("initial_state_sha256") == self.initial.get(key[0]), "shared initialization across arms")
            arrays, digest = self.checkpoint(fit["checkpoint"], f"selected/{key[0]}/{key[1]}.npz")
            self.check(fit.get("state_sha256") == digest, "selected-fit tensor digest")
            self.same(fit.get("query_gates"), arrays["gates"].tolist(), "selected gates from actual NPZ")
            self.same(fit.get("query_gates"), history[selected]["query_gates"], "selected-epoch gates")
            self.check(fit.get("parameters") == 10378, "equal trained parameter budget")
            self.loss_terms(fit["validation_terms"], key[1], "selected validation " + str(key))
            self.same(fit["validation_terms"]["restricted_ce"], history[selected]["validation_ce"],
                      "selected re-evaluation CE", tolerance=2e-7)
            for aggregate, field in (("training_seconds", "training_seconds"), ("validation_seconds", "validation_seconds")):
                self.same(fit.get(aggregate), sum(r[field] for r in history), "summed fit timing " + aggregate)
            self.check(finite(fit.get("selected_validation_seconds")) and fit["selected_validation_seconds"] >= 0,
                       "selected validation timing")
            progress = self.root / "in_progress" / f"{key[0]}-{key[1]}.npz"
            self.need(progress.is_file(), "completed fit retains its last best in-progress checkpoint")
            self.check(tensor_hash(self.arrays(progress.relative_to(self.root).as_posix())) == digest,
                       "best in-progress state equals final selection")
        if self.terminal:
            self.check(len(rows) == (144 if self.full else 24), "terminal epoch count")
            self.check(len(self.fits) == (18 if self.full else 3), "terminal completed-fit count")
        if "fits" in self.result:
            self.same(self.result["fits"], self.fits, "result copied completed fit records")
        if "completed_fits" in self.status:
            expected_completed = [{"seed": f["seed"], "arm": f["arm"]} for f in self.fits]
            claimed = self.status["completed_fits"]
            self.same(claimed, expected_completed if self.terminal else expected_completed[:len(claimed)],
                      "status completed fits (partial status may lag a final atomic fit write)")
        self.evidence["training_evidence"] = {"completed_fits": len(self.fits), "retained_epoch_records": len(rows),
            "completed_updates": sum(f.get("updates", 0) for f in self.fits),
            "selections": [{"seed": f["seed"], "arm": f["arm"], "epoch": f["selected_epoch"],
                            "state_sha256": f["state_sha256"]} for f in self.fits]}

    def prediction_rows(self, name, split, planned, required):
        rows = self.rows(name, required)
        cases, size = self.cases[split], len(self.cases[split])
        self.check(len(rows) <= size * len(planned), name + ": no excess records")
        families = sorted({case[0] for case in cases})
        successor = dict(zip(families, families[1:] + families[:1]))
        case_map = {(f, q, v): gold for f, q, v, gold, _, _ in cases}
        blocks, seen = defaultdict(list), set()
        for number, row in enumerate(rows):
            block, offset = divmod(number, size)
            self.need(block < len(planned), name + ": unexpected block")
            seed, condition = planned[block]
            family, query, role, gold, _, _ = cases[offset]
            identity = (row.get("seed"), row.get("condition"), row.get("family"),
                        row.get("query_index"), row.get("variant"))
            self.check(identity == (seed, condition, family, query, role),
                       name + ": fixed seed/condition/family/query/member ordering at " + str(number))
            self.check(identity not in seen, name + ": duplicate prediction identity")
            seen.add(identity)
            self.check(type(row.get("seed")) is int and type(row.get("query_index")) is int,
                       name + ": integer seed/query identity")
            self.check(type(row.get("choice")) is int and 0 <= row["choice"] < 8,
                       name + ": valid restricted color choice")
            self.check(type(row.get("gold")) is int and row["gold"] == gold, name + ": independent correct binding label")
            expected_keys = {"seed", "condition", "family", "query_index", "variant", "choice", "gold", "restricted_ce"}
            if condition == "rank_codec_oracle":
                expected_keys.add("payload_bytes")
                self.check(row.get("restricted_ce") is None and row.get("payload_bytes") == 2
                           and row.get("choice") == gold, name + ": deterministic complete-world oracle")
            else:
                ce = row.get("restricted_ce")
                self.check(finite(ce) and ce >= -1e-7, name + ": finite nonnegative retained CE")
                if finite(ce):
                    # These necessary probability bounds independently catch
                    # some inconsistent choice/CE records without logits.
                    self.check((row.get("choice") == gold and ce <= math.log(8) + 1e-5)
                               or (row.get("choice") != gold and ce >= math.log(2) - 1e-5),
                               name + ": choice and CE probability bound")
            if condition.endswith("_wrong_source"):
                expected_keys |= {"source_family", "source_gold", "source_answer_unchanged"}
                source_family = successor[family]
                source_gold = case_map[(source_family, query, role)]
                self.check(row.get("source_family") == source_family
                           and type(row.get("source_gold")) is int and row["source_gold"] == source_gold
                           and type(row.get("source_answer_unchanged")) is bool
                           and row["source_answer_unchanged"] == (source_gold == gold),
                           name + ": value-blind cyclic-source mapping and supplied label")
            self.check(set(row) == expected_keys, name + ": exact prediction schema")
            blocks[(seed, condition)].append(row)
        complete = {}
        for key, block in blocks.items():
            if len(block) == size:
                complete[key] = block
            else:
                self.check(not self.terminal, name + ": incomplete terminal condition")
        self.prediction_blocks[name] = {
            "complete_rows": len(rows), "expected_rows_for_all_blocks": size * len(planned),
            "complete_blocks": [{"seed": s, "condition": c, "records": len(block)} for (s, c), block in complete.items()],
            "partial_blocks": [{"seed": s, "condition": c, "records": len(block)}
                               for (s, c), block in blocks.items() if len(block) != size],
            "trailing_unparsed_bytes": self.jsonl_tails.get(name, 0)}
        return complete, rows

    def block_metrics(self, rows, split):
        """Array formulas from choices/independent gold, without study helpers."""
        n = SPLITS[split]
        choices = np.asarray([row["choice"] for row in rows], dtype=np.int64).reshape(n, 8, 4)
        gold = np.asarray([case[3] for case in self.cases[split]], dtype=np.int64).reshape(n, 8, 4)
        correct = choices == gold
        scores = {
            "record_accuracy": correct.mean(axis=(1, 2)),
            "target_paired_both_correct": (correct[:, :, 0] & correct[:, :, 1]).mean(axis=1),
            "target_answer_change": (choices[:, :, 0] != choices[:, :, 1]).mean(axis=1),
            "nuisance_stability": (choices[:, :, 0] == choices[:, :, 2]).mean(axis=1),
            "nuisance_both_correct": (correct[:, :, 0] & correct[:, :, 2]).mean(axis=1),
            "reorder_stability": (choices[:, :, 0] == choices[:, :, 3]).mean(axis=1),
            "reorder_both_correct": (correct[:, :, 0] & correct[:, :, 3]).mean(axis=1)}
        if all(row["restricted_ce"] is not None for row in rows):
            ce = np.asarray([row["restricted_ce"] for row in rows], dtype=np.float64).reshape(n, 8, 4)
            scores["restricted_ce"] = ce.mean(axis=(1, 2))
        if all("source_gold" in row for row in rows):
            source = np.asarray([row["source_gold"] for row in rows], dtype=np.int64).reshape(n, 8, 4)
            source_correct = choices == source
            scores.update(
                supplied_source_record_accuracy=source_correct.mean(axis=(1, 2)),
                supplied_source_target_paired_both_correct=(source_correct[:, :, 0] & source_correct[:, :, 1]).mean(axis=1),
                supplied_source_answer_unchanged_fraction=(source == gold).mean(axis=(1, 2)))
        counts = {
            "target_paired_both_correct": int((correct[:, :, 0] & correct[:, :, 1]).sum()),
            "nuisance_stable": int((choices[:, :, 0] == choices[:, :, 2]).sum()),
            "reorder_stable": int((choices[:, :, 0] == choices[:, :, 3]).sum()),
            "record_correct": int(correct.sum())}
        aggregate = {"quartets": n * 8, "records": n * 32,
            "metrics": {name: float(values.mean()) for name, values in scores.items()}, "counts": counts}
        return scores, aggregate

    def validation_and_gate(self):
        self.validation, rows = self.prediction_rows("validation_predictions.jsonl", "validation", FIT_ORDER, self.terminal)
        fit_keys = {(fit["seed"], fit["arm"]) for fit in self.fits}
        self.check(fit_keys <= set(self.validation), "every completed fit has complete retained selected-validation rows")
        if self.terminal:
            self.check(set(self.validation) == fit_keys, "terminal validation blocks exactly match completed fits")
        validation = {}
        for key, block in self.validation.items():
            _, stats = self.block_metrics(block, "validation")
            validation[key] = stats
        for fit in self.fits:
            key = (fit["seed"], fit["arm"])
            if key not in validation:
                continue
            self.same(fit.get("validation"), validation[key], "independently rebuilt selected-validation metrics " + str(key))
            self.same(fit["validation_terms"]["restricted_ce"], validation[key]["metrics"]["restricted_ce"],
                      "selected-validation CE from raw records", tolerance=2e-7)
        gate = self.read("competence_gate.json", self.terminal)
        if gate is not None:
            text = {f["seed"]: f for f in self.fits if f["arm"] == "text"}
            self.need(set(text) == set(SEEDS) and all((s, "text") in validation for s in SEEDS),
                      "text gate requires every completed text fit and its raw validation")
            per_seed = []
            for seed in SEEDS:
                stats = validation[(seed, "text")]
                # Integer thresholds avoid silently changing the denominator.
                counts = stats["counts"]
                passed = (counts["target_paired_both_correct"] >= math.ceil(.8 * 256)
                          and counts["nuisance_stable"] >= math.ceil(.9 * 256)
                          and counts["reorder_stable"] >= math.ceil(.9 * 256))
                per_seed.append({"seed": seed, "passed": passed, "validation": stats,
                    "selected_epoch": text[seed]["selected_epoch"], "state_sha256": text[seed]["state_sha256"]})
            expected = {"passed": all(r["passed"] for r in per_seed), "per_seed": per_seed,
                "failed_seeds": [r["seed"] for r in per_seed if not r["passed"]],
                "thresholds": {"target_paired_both_correct": .8, "nuisance_stability": .9, "reorder_stability": .9},
                "fresh_test_scoring_started": False, "other_arms_started": False}
            self.same(gate, expected, "competence gate independently rebuilt from raw validation")
            self.gate = expected
            if "competence_gate" in self.result:
                self.same(self.result["competence_gate"], expected, "result competence gate")
        if any(f["arm"] != "text" for f in self.fits) or any(r["arm"] != "text" for r in self.history_rows):
            self.check(self.gate is not None and self.gate["passed"],
                       "any completed or partial non-text training requires reconstructed passed text gate")
        self.evidence["validation"] = {
            "records": len(rows), "blocks": len(self.validation), "competence_gate": self.gate,
            "per_fit": [{"seed": s, "arm": a, **value} for (s, a), value in validation.items()]}

    def selection(self):
        selected = self.read("selection_frozen.json", self.full)
        if selected is None:
            if self.stopped:
                self.check(not (self.root / "selection_frozen.json").exists(), "receiver-limited study has no full selection")
            return
        self.need(len(self.fits) == 18 and self.gate is not None and self.gate["passed"],
                  "persisted full selection requires all 18 fits and passed competence gate")
        fields = ("seed", "arm", "selected_epoch", "state_sha256", "checkpoint", "query_gates",
                  "initial_state_sha256", "updates", "epoch_order_sha256")
        expected = {"all_required_fits_complete": True, "fresh_test_model_cases_loaded": False,
            "config": CONFIG, "competence_gate": self.gate,
            "checkpoints": [{k: f[k] for k in fields} for f in sorted(self.fits, key=lambda f: (f["seed"], f["arm"]))]}
        self.selection_verified = self.same(selected, expected, "full validation-only frozen checkpoint selection")
        if "selection" in self.result:
            self.same(self.result["selection"], expected, "result selection copy")
        self.evidence["selection"] = {"all_18_fits_verified": True, "retained_manifest": metadata(self.root / "selection_frozen.json")}

    def fresh_predictions(self):
        self.fresh, rows = self.prediction_rows("predictions.jsonl", "test", FRESH_ORDER, self.full)
        if rows or self.status.get("fresh_test_model_cases_loaded") is True or self.result.get("fresh_test_model_cases_loaded") is True:
            self.check(self.gate is not None and self.gate["passed"] and len(self.fits) == 18
                       and self.selection_verified,
                       "any fresh model access requires all 18 fits and a verified persisted selection")
        if self.full:
            self.need(set(self.fresh) == set(FRESH_ORDER) and len(rows) == 86016, "complete 14-by-3 fresh grid")
        if self.stopped:
            for name in ("predictions.jsonl", "predictions.jsonl.gz", "selection_frozen.json",
                         "source_intervention_manifest.json", "evaluation_conditions.json"):
                self.check(not (self.root / name).exists(), "receiver-limited study must not retain " + name)
            self.check(len(self.fits) == 3 and all(f["arm"] == "text" for f in self.fits)
                       and self.gate is not None and not self.gate["passed"], "valid receiver-limited stopping branch")
            self.check(self.status.get("fresh_test_model_cases_loaded") is False
                       and self.result.get("fresh_test_model_cases_loaded") is False
                       and self.status.get("other_arms_started") is False, "receiver gate stops before latent/fresh work")
            self.check("summary" not in self.result and "selection" not in self.result
                       and "advancement" not in self.result, "receiver-limited study makes no fresh primary claim")
        completed = [{"seed": s, "condition": c, "records": len(v)} for (s, c), v in self.fresh.items()]
        for value, label in ((self.result, "result"), (self.status, "status")):
            if "completed_conditions" in value:
                claimed = value["completed_conditions"]
                self.same(claimed, completed if self.terminal else completed[:len(claimed)],
                          label + " completed-condition records")
        if self.terminal:
            self.check(self.status.get("complete") is True and self.status.get("phase") == ("complete" if self.full else "receiver_limited"),
                       "correct terminal scientific status")
        mapping = self.read("source_intervention_manifest.json", self.full)
        if mapping is not None:
            expected_indices = [((number // 8 + 1) % 64) * 8 + number % 8 for number in range(512)]
            self.same(mapping.get("quartet_successor_indices"), expected_indices, "value-blind cyclic source index mapping")
            self.check(mapping.get("same_for_all_seeds_and_arms") is True and mapping.get("uses_answer_values") is False,
                       "declared source mapping boundaries")
        integrity = self.result.get("prediction_integrity")
        if integrity is not None:
            path = self.root / "predictions.jsonl"
            self.need(path.is_file(), "prediction integrity without raw predictions")
            raw = path.read_bytes()
            self.same(integrity, {"records": raw.count(b"\n"), "raw_bytes": len(raw), "raw_sha256": sha(raw),
                "compressed": metadata(self.root / "predictions.jsonl.gz")}, "raw/compressed prediction integrity in results")
        for (seed, condition), block in self.fresh.items():
            if condition in ("private_constant", "aware_ce_zero", "aware_aux_zero", "frozen_no_channel"):
                choices = np.asarray([row["choice"] for row in block]).reshape(64, 8, 4)
                self.check(bool((choices == choices[0, :, 0][None, :, None]).all()),
                           "information-free choices constant conditional on actual public question " + condition + "/" + str(seed))
            if condition in ("frozen_no_channel", "rank_codec_oracle") and seed != SEEDS[0] and (SEEDS[0], condition) in self.fresh:
                first = self.fresh[(SEEDS[0], condition)]
                self.same([(r["choice"], r["restricted_ce"]) for r in block],
                          [(r["choice"], r["restricted_ce"]) for r in first],
                          "seed-independent frozen/oracle outputs " + condition, tolerance=2e-7)
        source_checks = []
        for arm in ("aware_ce", "aware_aux"):
            for seed in SEEDS:
                source_key, base_key = (seed, arm + "_wrong_source"), (seed, arm)
                if source_key not in self.fresh or base_key not in self.fresh:
                    continue  # Incomplete blocks cannot invent missing paired evidence.
                counts = source_path_consistency(self.fresh[base_key], self.fresh[source_key])
                self.check(counts["matched_rows"] == 2048 and counts["unmatched_rows"] == 0,
                           "complete source-path matched-row coverage " + arm + "/" + str(seed))
                self.check(counts["choice_mismatches"] == 0 and counts["source_label_mismatches"] == 0,
                           "wrong-source choice equals identical donor base computation " + arm + "/" + str(seed))
                self.check(counts["same_label_ce_mismatches"] == 0,
                           "wrong-source CE equals donor CE when gold labels coincide " + arm + "/" + str(seed))
                source_checks.append({"seed": seed, "arm": arm, **counts})
        self.evidence["source_intervention_consistency"] = {
            "scope": "all available complete base/wrong-source blocks; no inference for missing partial blocks",
            "per_seed_arm": source_checks}
        evaluations = self.read("evaluation_conditions.json", self.full)
        if evaluations is not None:
            expected_keys = [(s, c) for s, c in FRESH_ORDER if c != "rank_codec_oracle"]
            actual_keys = [(r["seed"], r["condition"]) for r in evaluations]
            self.same(actual_keys, expected_keys if self.full else expected_keys[:len(actual_keys)], "evaluation term order")
            for row in evaluations:
                key = (row["seed"], row["condition"])
                self.check(finite(row.get("seconds")) and row["seconds"] >= 0, "condition elapsed time")
                if key not in self.fresh:
                    self.check(not self.terminal, "evaluation terms without complete raw block")
                    continue
                ce = float(np.mean([r["restricted_ce"] for r in self.fresh[key]]))
                self.same(row["terms"].get("restricted_ce"), ce, "condition mean CE from raw rows")
                if key[1] == "frozen_no_channel":
                    self.check(set(row["terms"]) == {"restricted_ce"}, "no-channel has no fabricated message-distance terms")
                else:
                    arm = next((a for a in ARMS if key[1] == a or key[1].startswith(a + "_")), None)
                    self.loss_terms(row["terms"], arm, "evaluation " + str(key))
        self.evidence["prediction_coverage"] = self.prediction_blocks

    def bootstrap_summary(self):
        if not self.full:
            self.evidence["primary_complete_study_claim_permitted"] = False
            return
        self.need(set(self.fresh) == set(FRESH_ORDER), "no primary comparison from an incomplete fresh grid")
        metrics = {key: self.block_metrics(rows, "test")[0] for key, rows in self.fresh.items()}
        # Same fixed whole-family draws for every metric and paired contrast.
        draws = np.random.Generator(np.random.PCG64(66173)).integers(0, 64, size=(2000, 64))
        aggregate, conditions = {}, []
        for condition in CONDITIONS:
            values = {}
            for metric in metrics[(SEEDS[0], condition)]:
                matrix = np.stack([metrics[(seed, condition)][metric] for seed in SEEDS], axis=0)
                family = matrix.mean(axis=0)
                seed_means = matrix.mean(axis=1)
                values[metric] = {
                    "mean": float(family.mean()),
                    "family_cluster_95_interval": np.percentile(family[draws].mean(axis=1), [2.5, 97.5], method="linear").tolist(),
                    "seed_means": {str(seed): float(seed_means[i]) for i, seed in enumerate(SEEDS)},
                    "seed_range": [float(seed_means.min()), float(seed_means.max())]}
                aggregate[(condition, metric)] = family
            conditions.append({"condition": condition, "metrics": values})
        pairs = [("primary_aux_minus_ce", "aware_aux", "aware_ce"),
                 ("aux_minus_blind", "aware_aux", "blind_ce")]
        pairs += [(a + "_minus_" + c, a, a + "_" + c) for a in ("aware_ce", "aware_aux") for c in INTERVENTIONS]
        comparisons, primary_metric = [], "target_paired_both_correct"
        for name, a, b in pairs:
            difference = aggregate[(a, primary_metric)] - aggregate[(b, primary_metric)]
            per_seed = {str(s): float(metrics[(s, a)][primary_metric].mean() - metrics[(s, b)][primary_metric].mean())
                        for s in SEEDS}
            comparisons.append({"contrast": name, "a": a, "b": b, "mean_difference": float(difference.mean()),
                "family_cluster_95_interval": np.percentile(difference[draws].mean(axis=1), [2.5, 97.5], method="linear").tolist(),
                "seed_differences": per_seed,
                "seed_difference_range": [min(per_seed.values()), max(per_seed.values())]})
        expected = {"conditions": conditions, "paired_comparisons": comparisons, "family_count": 64,
                    "bootstrap_resamples": 2000, "bootstrap_seed": 66173,
                    "interval_scope": "descriptive family intervals conditional on three fixed seed pairs"}
        self.same(self.result.get("summary"), expected, "all independent fresh statistics and whole-family intervals")
        by_condition = {r["condition"]: r["metrics"] for r in conditions}
        contrast = {r["contrast"]: r for r in comparisons}
        aux = by_condition["aware_aux"]
        components = {
            "text_validation_gate": bool(self.gate and self.gate["passed"]),
            "fresh_paired_at_least_80_percent": aux["target_paired_both_correct"]["mean"] >= .8,
            "fresh_nuisance_stability_at_least_90_percent": aux["nuisance_stability"]["mean"] >= .9,
            "fresh_reorder_stability_at_least_90_percent": aux["reorder_stability"]["mean"] >= .9,
            "positive_mean_correct_vs_wrong_source_pair_difference": contrast["aware_aux_minus_wrong_source"]["mean_difference"] > 0}
        practical = all(components.values())
        positive = contrast["primary_aux_minus_ce"]["family_cluster_95_interval"][0] > 0
        gates = {"components": components, "practical_advance": practical,
                 "primary_interval_strictly_positive": positive,
                 "supported_comparative_improvement": practical and positive, "breakthrough_claim": False}
        self.same(self.result.get("advancement"), gates, "all separately reported advancement gates")
        diagnostics = []
        for arm in ("aware_ce", "aware_aux"):
            for intervention in INTERVENTIONS:
                for seed in SEEDS:
                    original = self.fresh[(seed, arm)]
                    other = self.fresh[(seed, arm + "_" + intervention)]
                    diagnostics.append({"arm": arm, "intervention": intervention, "seed": seed,
                        "changed_choices": sum(a["choice"] != b["choice"] for a, b in zip(original, other)),
                        "records": 2048})
        self.same(self.result.get("intervention_diagnostics"), diagnostics, "raw paired intervention choice-change counts")
        self.check(self.result.get("complete_study") is True
                   and self.result.get("fresh_test_model_cases_loaded") is True
                   and self.status.get("fresh_test_model_cases_loaded") is True, "complete study flags")
        self.evidence["fresh_summary"] = expected
        self.evidence["advancement"] = gates
        self.evidence["bootstrap"] = {
            "family_indices_sha256_little_endian_int64": sha(np.asarray(draws, dtype="<i8").tobytes()),
            "family_order": sorted({c[0] for c in self.cases["test"]}),
            "point_and_intervals_rebuilt_without_binding_helpers": True}

    def accounting(self):
        tokens = self.read("tokenization_manifest.json", self.terminal)
        features = self.read("feature_manifest.json", self.terminal)
        if tokens is not None:
            self.check(tokens.get("passed") is True
                       and tokens.get("private_and_complete_conditional_token_multisets_equal") is True,
                       "recorded tokenization preflight passed")
            self.same(tokens.get("candidate_colors"), list(COLORS), "fixed candidate order")
            ids = tokens.get("candidate_token_ids", [])
            self.check(len(ids) == 8 and all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == 8,
                       "distinct recorded single-token color IDs")
            self.check(tokens.get("receiver_separator") == "\n", "fixed receiver separator")
            for role in ("sender", "receiver"):
                for split, count in SPLITS.items():
                    lengths = tokens.get("token_lengths", {}).get(role + "/" + split, {})
                    self.need(set(lengths) == {"private_text", "private_symbolic", "query", "latent_receiver",
                                              "text_receiver", "symbolic_receiver", "frozen_no_channel"},
                              "recorded token length categories")
                    for name, value in lengths.items():
                        self.check(value["count"] == 32 * count
                                   and 0 < value["min"] <= value["mean"] <= value["max"] <= 192,
                                   "recorded token sequence count/bounds " + role + "/" + split + "/" + name)
                        self.same(value["sum"], value["mean"] * value["count"], "token-count mean/sum")
            if "candidate_token_ids" in self.result:
                self.same(self.result["candidate_token_ids"], ids, "result candidate IDs")
                self.same(self.result.get("candidate_colors"), list(COLORS), "result candidate colors")
        if not self.terminal:
            self.check(not self.result.get("complete_study"), "incomplete run cannot claim complete study")
            self.check("summary" not in self.result, "incomplete run cannot provide a complete-study summary")
            return
        self.need(features is not None, "terminal feature accounting manifest")
        used = ("train", "validation", "test") if self.full else ("train", "validation")
        sources = {m["source"] for s in used for q in self.partitions[s] for m in q["members"]}
        source_count = len(sources) if self.full else 0
        text_receiver_count = sum(len(self.cases[s]) for s in used)
        receiver_count = 8 + text_receiver_count * (2 if self.full else 1)
        self.check(features["source"]["unique_entries"] == source_count, "actual unique private encoder cache entry count")
        self.check(features["question"]["unique_entries"] == 8, "actual unique public question cache entry count")
        self.check(features["receiver_embeddings"]["unique_entries"] == receiver_count, "actual unique receiver embedding cache count")
        self.same(features.get("actual_encoded_sequences"), {"source": source_count, "question": 8},
                  "actual encoded source/question sequence counts")
        source_calls = 0
        if self.full:
            fitting_sources = {m["source"] for s in ("train", "validation") for q in self.partitions[s] for m in q["members"]}
            fresh_sources = {m["source"] for q in self.partitions["test"] for m in q["members"]}
            source_calls = math.ceil(len(fitting_sources) / 16) + math.ceil(len(fresh_sources) / 16)
        self.same(features.get("actual_encoder_forward_calls"), {"source": source_calls, "question": 1},
                  "batched frozen encoder forward-call count")
        logical = {"source": 285696, "question": 451584, "receiver_embedding": 540672} if self.full else {
            "source": 0, "question": 76800, "receiver_embedding": 76800}
        self.same(features.get("logical_cache_requests"), logical, "uncached logical requests from the fixed update/evaluation budget")
        self.check(features.get("logical_requests_are_not_latency_measurements") is True,
                   "logical requests distinguished from measured latency")
        self.check(features["question"]["bytes"] == 8 * 768 * 4, "question-cache float32 bytes")
        if tokens is not None:
            lengths = tokens["token_lengths"]
            sender_source_tokens = lengths["sender/train"]["private_text"]["min"]
            self.check(all(lengths["sender/" + s]["private_text"]["min"] ==
                           lengths["sender/" + s]["private_text"]["max"] == sender_source_tokens for s in SPLITS),
                       "recorded private equal token-bag lengths consistent across partitions")
            self.check(features["source"]["bytes"] == source_count * sender_source_tokens * 768 * 4,
                       "private hidden-cache bytes from recorded token lengths")
            receiver_tokens = 0
            for split in used:
                fields = ("text_receiver", "symbolic_receiver") if self.full else ("text_receiver",)
                for field in fields:
                    receiver_tokens += lengths["receiver/" + split][field]["sum"] - 2 * len(self.cases[split])
            receiver_tokens += lengths["receiver/train"]["query"]["sum"] // (64 * 4)
            self.check(features["receiver_embeddings"]["bytes"] == receiver_tokens * 128 * 4,
                       "receiver-cache float32 bytes from recorded token lengths")
        for category in ("source", "question", "receiver_embeddings"):
            value = features[category]
            self.check(type(value["bytes"]) is int and value["bytes"] >= 0
                       and isinstance(value.get("ordered_content_sha256"), str)
                       and len(value["ordered_content_sha256"]) == 64, "recorded cache digest/byte shape")
        expected_questions = {"The color of the " + o + " is" for o in OBJECTS}
        self.check(set(features.get("question_norms", {})) == expected_questions
                   and all(finite(v) and 27 < v < 28 for v in features["question_norms"].values()),
                   "recorded normalized question-vector norms")
        account = self.result.get("accounting", {})
        byte_stats = {}
        for split, cases in self.cases.items():
            byte_stats[split] = {
                "source_text_utf8_bytes": summary_stats(len(c[4]["source"].encode()) for c in cases),
                "symbolic_utf8_bytes": summary_stats(len(c[4]["symbolic"].encode()) for c in cases),
                "public_question_utf8_bytes": summary_stats(len(c[5].encode()) for c in cases)}
        self.same(account.get("utf8_bytes_by_partition"), byte_stats, "independent UTF8 byte accounting")
        for key, expected in (("latent_payload_float32_bytes", 64), ("exact_complete_world_rank_payload_bytes", 2),
                              ("latent_to_rank_payload_ratio", 32), ("rank_packet_reusable_across_queries", True),
                              ("network_energy_production_speed_measured", False)):
            self.same(account.get(key), expected, "payload/cost distinction " + key)
        self.same(account.get("cache"), features, "result final feature accounting")
        self.check(account.get("selected_checkpoint_bytes") == sum(f["checkpoint"]["bytes"] for f in self.fits),
                   "selected portable checkpoint byte total")
        timing = self.result.get("timings_seconds", {})
        self.same(timing.get("training"), sum(f["training_seconds"] for f in self.fits), "summed training timing")
        self.same(timing.get("validation"), sum(f["validation_seconds"] + f["selected_validation_seconds"] for f in self.fits),
                  "summed validation timing")
        self.evidence["accounting"] = {"payload_float32_bytes": 64, "complete_world_rank_bytes": 2,
            "selected_checkpoint_bytes": sum(f["checkpoint"]["bytes"] for f in self.fits),
            "source_cache_entries": source_count, "question_cache_entries": 8,
            "receiver_embedding_entries": receiver_count, "fixed_budget_logical_requests": logical}

    def run(self):
        for name in ("retention", "provenance", "dataset", "initial_states", "histories",
                     "validation_and_gate", "selection", "fresh_predictions", "bootstrap_summary", "accounting"):
            self.step(name, getattr(self, name))
        if self.error_count:
            classification = "invalid_or_inconsistent_evidence"
        elif self.full:
            classification = "complete_study"
        elif self.stopped:
            classification = "valid_receiver_limited_stop"
        else:
            classification = "incomplete_study"
        self.evidence["primary_complete_study_claim_permitted"] = classification == "complete_study"
        return {
            "audit_status": "passed" if not self.error_count else "failed",
            "classification": classification, "checks": self.checks,
            "error_count": self.error_count, "first_errors": self.errors,
            "errors_truncated": self.error_count > len(self.errors), "notes": self.notes,
            "stage_status": self.stage_status, "evidence": self.evidence,
            "declared_scientific_status": self.result.get("execution_status"),
            "primary_complete_study_claim_permitted": classification == "complete_study",
            "benchmark_invocations": 0, "model_imports_or_inference": 0,
            "limits": LIMITS,
        }


def self_test():
    """Small deterministic arithmetic/protocol tests; no empirical study run."""
    partitions, families, reserved, data_audit = frozen_data()
    assert len(families) == 160 and len(reserved) == 4640 and data_audit["unique_source_order_identities"] == 2340
    checker = object.__new__(Audit)
    checker.cases = {
        split: [(q["family"], q["query_index"], m["variant"], m["world"][q["query_index"]], m, q["query"])
                for q in records for m in q["members"]]
        for split, records in partitions.items()}
    for split in ("validation", "test"):
        oracle = [{"choice": case[3], "gold": case[3], "restricted_ce": None} for case in checker.cases[split]]
        scores, aggregate = checker.block_metrics(oracle, split)
        assert aggregate["metrics"] == {
            "record_accuracy": 1., "target_paired_both_correct": 1., "target_answer_change": 1.,
            "nuisance_stability": 1., "nuisance_both_correct": 1., "reorder_stability": 1., "reorder_both_correct": 1.}
        assert aggregate["counts"]["target_paired_both_correct"] == SPLITS[split] * 8
        constant = [{"choice": 0, "gold": case[3], "restricted_ce": math.log(8)} for case in checker.cases[split]]
        scores, aggregate = checker.block_metrics(constant, split)
        assert aggregate["metrics"]["record_accuracy"] == .125
        assert aggregate["metrics"]["target_paired_both_correct"] == 0.
        assert aggregate["metrics"]["target_answer_change"] == 0.
        assert aggregate["metrics"]["nuisance_stability"] == 1.
        assert np.allclose(scores["restricted_ce"], math.log(8), rtol=0, atol=1e-14)
    base_row = {"seed": 101, "family": "donor", "query_index": 0, "variant": "O",
                "choice": 2, "gold": 2, "restricted_ce": .4}
    supplied_row = {"seed": 101, "family": "original", "source_family": "donor",
                    "query_index": 0, "variant": "O", "choice": 2, "gold": 2,
                    "source_gold": 2, "restricted_ce": .4}
    matched = source_path_consistency([base_row], [supplied_row])
    assert matched == {"matched_rows": 1, "unmatched_rows": 0, "same_label_ce_rows": 1,
                       "choice_mismatches": 0, "source_label_mismatches": 0, "same_label_ce_mismatches": 0}
    assert source_path_consistency([base_row], [{**supplied_row, "choice": 3}])["choice_mismatches"] == 1
    assert source_path_consistency([base_row], [{**supplied_row, "restricted_ce": .6}])["same_label_ce_mismatches"] == 1
    other_label = {**supplied_row, "gold": 1, "restricted_ce": 3.}
    assert source_path_consistency([base_row], [other_label])["same_label_ce_rows"] == 0
    assert source_path_consistency([], [supplied_row])["unmatched_rows"] == 1
    left = np.array([[1., 2.]], dtype=np.float32)
    assert tensor_hash({"x": left}) == tensor_hash({"x": left.copy()})
    assert tensor_hash({"x": left}) != tensor_hash({"x": left.astype(np.float64)})
    assert tensor_hash({"x": left}) != tensor_hash({"x": left.reshape(2, 1)})
    assert not equivalent(True, 1)
    assert len(order_hashes(101)) == len(set(order_hashes(101))) == 8
    try:
        strict_json('{"a": 1, "a": 2}')
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate keys accepted")
    # A hard kill can leave valid raw evidence before parent compression.
    # Absence is different from a contradictory/corrupt retained gzip.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = b'{"synthetic_audit_fixture":true}\n'
        (root / "validation_predictions.jsonl").write_bytes(raw)
        partial = Audit(root)
        partial.step("retention", partial.retention)
        assert partial.error_count == 0
        assert any("without gzip" in note for note in partial.notes)
        zipped = root / "validation_predictions.jsonl.gz"
        zipped.write_bytes(gzip.compress(raw, mtime=0))
        preserved = Audit(root)
        preserved.step("retention", preserved.retention)
        assert preserved.error_count == 0
        zipped.write_bytes(gzip.compress(b"different retained bytes\n", mtime=0))
        corrupt = Audit(root)
        corrupt.step("retention", corrupt.retention)
        assert corrupt.error_count > 0
        zipped.unlink()
        (root / "execution.json").write_text(json.dumps({
            "prediction_receipts": {"validation_predictions.jsonl": {"raw": metadata(root / "validation_predictions.jsonl")}}
        }))
        claimed = Audit(root)
        claimed.step("retention", claimed.retention)
        assert claimed.error_count > 0
    print(json.dumps({"audit_self_test": "passed", "model_calls": 0, "benchmark_invocations": 0}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(v is not None for v in (args.artifact, args.receipt, args.source_root)):
            parser.error("--self-test takes no artifact/output arguments")
        self_test()
        return
    if args.artifact is None or args.receipt is None:
        parser.error("--artifact and --receipt are required")
    artifact, receipt_path = args.artifact.resolve(), args.receipt.resolve()
    if not artifact.is_dir():
        parser.error("--artifact must be an extracted study directory")
    if receipt_path.is_relative_to(artifact):
        parser.error("--receipt must be outside the audited artifact tree")
    started = time.perf_counter()
    try:
        checker = Audit(artifact, None if args.source_root is None else args.source_root.resolve())
        result = checker.run()
    except Exception as error:
        result = {"audit_status": "failed", "classification": "unreadable_or_invalid_evidence",
                  "error_count": 1, "first_errors": [type(error).__name__ + ": " + str(error)],
                  "primary_complete_study_claim_permitted": False,
                  "benchmark_invocations": 0, "model_imports_or_inference": 0, "limits": LIMITS}
    result["audit_runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started,
        "script_sha256": metadata(Path(__file__))["sha256"]}
    result["artifact_directory"] = str(artifact)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_status": result["audit_status"], "classification": result["classification"],
                      "checks": result.get("checks", 0), "error_count": result["error_count"],
                      "primary_complete_study_claim_permitted": result["primary_complete_study_claim_permitted"],
                      "benchmark_invocations": 0, "receipt": str(receipt_path)}, sort_keys=True))
    if result["audit_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
