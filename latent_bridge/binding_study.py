"""Execute the fixed binding study, stopping before latent fits if text is limited."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import time
import numpy as np

from . import binding_core as core
from .binding_data import (COLORS, MEMBERS, OBJECTS, decode_rank, encode_rank,
    file_metadata, load_partitions, parse_question, parse_source, validate_partitions)
from .transformers_runner import fingerprint_directory


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, values):
    with Path(path).open("a", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def verify_manifests(args, output):
    root = Path(__file__).resolve().parents[1]
    code = json.loads(Path(args.code_manifest).read_text())
    models = json.loads(Path(args.model_manifest).read_text())
    write_json(output / "code_manifest.json", code)
    write_json(output / "model_manifest.json", models)
    if code.get("frozen_design_commit") != core.DESIGN_COMMIT or not code.get("all_match"):
        raise ValueError("code manifest does not verify the frozen design")
    if code.get("historical_helper_commit") != core.HISTORICAL_HELPER_COMMIT:
        raise ValueError("historical helper reference differs")
    needed = {"latent_bridge/binding_data.py", "latent_bridge/binding_core.py",
              "latent_bridge/binding_study.py", "docs/binding_design.md"}
    if not needed.issubset(code.get("files", {})):
        raise ValueError("code manifest omits a required study source")
    for name, expected in code["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe code-manifest path")
        if file_metadata(root / name) != expected:
            raise ValueError("implementation file bytes differ: " + name)
    helpers = code.get("historical_helpers", {})
    if not helpers:
        raise ValueError("historical helper checks are missing")
    for name, values in helpers.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe helper path")
        actual = file_metadata(root / name)["sha256"]
        if values.get("expected_sha256") != actual or values.get("actual_sha256") != actual:
            raise ValueError("historical helper changed: " + name)
    design_hash = file_metadata(root / "docs/binding_design.md")["sha256"]
    if (code.get("design_sha256") != design_hash or code.get("expected_design_sha256") != design_hash
        or models.get("design_sha256") != design_hash):
        raise ValueError("design digest differs between manifests and implementation")
    if code.get("github_commit") != models.get("github_commit") or str(
        code.get("github_run_id")) != str(models.get("github_run_id")):
        raise ValueError("model/code run identities differ")
    runtime_path = output / "runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    actual_packages = {name: importlib.metadata.version(name) for name in core.PINNED_PACKAGES}
    runtime.update(
        python=platform.python_version(), expected_python="3.11.18",
        packages=actual_packages, actual_packages=actual_packages,
        expected_packages=core.PINNED_PACKAGES, platform=platform.platform(),
        cpu_count=os.cpu_count(), torch_threads=args.threads,
        github_commit=code.get("github_commit"), github_run_id=code.get("github_run_id"))
    write_json(runtime_path, runtime)
    if runtime["python"] != "3.11.18" or runtime["packages"] != core.PINNED_PACKAGES:
        raise ValueError("installed runtime differs from frozen versions")
    for role, directory in (("sender", args.sender_model), ("receiver", args.receiver_model)):
        entry = models.get("models", {}).get(role, {})
        if any(entry.get(name) != value for name, value in core.PINNED_MODELS[role].items()):
            raise ValueError("model identifier or revision differs")
        if not (Path(directory) / "model.safetensors").is_file():
            raise ValueError("pinned safetensors file is absent")
        if fingerprint_directory(directory) != entry.get("files"):
            raise ValueError("local model files differ from their manifest")
    return code, models, runtime


class FrozenFeatures:
    """Only private text and public question strings enter learned messages."""

    def __init__(self, sender, receiver, source_tokenizer, receiver_tokenizer, ids, controls):
        import torch
        self.torch = torch
        self.sender, self.receiver = sender, receiver
        self.source_tokenizer, self.receiver_tokenizer = source_tokenizer, receiver_tokenizer
        self.ids, self.controls = ids, controls
        self.sources, self.questions, self.static = {}, {}, {}
        self.encoding_seconds = {"source": 0., "question": 0., "receiver_embedding": 0.}
        self.encoding_sequences = {"source": 0, "question": 0}
        self.encoding_forward_calls = {"source": 0, "question": 0}
        self.logical_requests = {"source": 0, "question": 0, "receiver_embedding": 0}

    def populate_sender(self, texts, kind):
        torch = self.torch
        cache = self.sources if kind == "source" else self.questions
        missing = sorted(set(texts) - set(cache))
        for offset in range(0, len(missing), 16):
            batch = missing[offset:offset + 16]
            tokens = [self.source_tokenizer.encode(text, add_special_tokens=False) for text in batch]
            if any(not ids or len(ids) > core.MAX_LENGTH for ids in tokens):
                raise ValueError("empty or oversized sender input")
            lengths = torch.tensor([len(ids) for ids in tokens])
            input_ids = torch.zeros((len(tokens), int(lengths.max())), dtype=torch.long)
            for index, ids in enumerate(tokens):
                input_ids[index, :len(ids)] = torch.tensor(ids)
            mask = (torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]).long()
            before = time.perf_counter()
            with torch.no_grad():
                hidden = self.sender(input_ids=input_ids, attention_mask=mask,
                                     use_cache=False, return_dict=True).last_hidden_state
                for index, text in enumerate(batch):
                    state = hidden[index, :len(tokens[index])].detach().clone()
                    cache[text] = state if kind == "source" else core.normalize_question(state.mean(dim=0))
            self.encoding_seconds[kind] += time.perf_counter() - before
            self.encoding_sequences[kind] += len(batch)
            self.encoding_forward_calls[kind] += 1

    def populate_static(self, texts):
        torch = self.torch
        for text in sorted(set(texts) - set(self.static)):
            tokens = self.receiver_tokenizer.encode(text, add_special_tokens=False)
            if not tokens or len(tokens) + core.SLOTS > core.MAX_LENGTH:
                raise ValueError("receiver sequence exceeds maximum length")
            before = time.perf_counter()
            with torch.no_grad():
                self.static[text] = self.receiver.get_input_embeddings()(torch.tensor(tokens)).detach()
            self.encoding_seconds["receiver_embedding"] += time.perf_counter() - before

    def prepare(self, cases, private, receiver_arms):
        self.populate_sender([case["query"] for case in cases], "question")
        if private:
            self.populate_sender([case["member"]["source"] for case in cases], "source")
        self.populate_static([core.receiver_text(case, arm) for case in cases for arm in receiver_arms])

    def message(self, bridge, seed, arm, cases, source_cases=None, wrong_question=False):
        torch = self.torch
        from torch.nn.utils.rnn import pad_sequence
        fixed_hidden, fixed_mask, fixed_question = self.controls[seed]
        if arm in ("aware_ce", "aware_aux", "blind_ce"):
            sources = cases if source_cases is None else source_cases
            states = [self.sources[case["member"]["source"]] for case in sources]
            hidden = pad_sequence(states, batch_first=True)
            lengths = torch.tensor([len(state) for state in states])
            mask = (torch.arange(hidden.shape[1])[None, :] < lengths[:, None]).long()
            self.logical_requests["source"] += len(cases)
        else:
            hidden = fixed_hidden.expand(len(cases), -1, -1)
            mask = fixed_mask.expand(len(cases), -1)
        if arm == "blind_ce":
            question = fixed_question[None, :].expand(len(cases), -1)
        else:
            questions = [core.wrong_public_question(case["query"]) if wrong_question else case["query"]
                         for case in cases]
            question = torch.stack([self.questions[text] for text in questions])
            self.logical_requests["question"] += len(cases)
        hidden, mask, question = core.route_inputs(
            arm, hidden, mask, question, fixed_hidden, fixed_mask, fixed_question)
        return bridge.encode(hidden, mask, question)

    def score(self, bridge, seed, arm, cases, source_cases=None, wrong_question=False, zero=False):
        torch = self.torch
        from torch.nn.utils.rnn import pad_sequence
        if arm == "frozen_no_channel":
            message, prefix = None, None
        else:
            message = self.message(bridge, seed, arm, cases, source_cases, wrong_question)
            if zero:
                message = torch.zeros_like(message)
            prefix = bridge.decode(message)
        sequences = []
        for index, case in enumerate(cases):
            static = self.static[core.receiver_text(case, arm)]
            sequences.append(static if prefix is None else torch.cat((prefix[index], static), dim=0))
        self.logical_requests["receiver_embedding"] += len(cases)
        embeddings = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        if int(lengths.max()) > core.MAX_LENGTH:
            raise ValueError("actual receiver prefix plus text exceeds limit")
        mask = (torch.arange(embeddings.shape[1])[None, :] < lengths[:, None]).long()
        return core.selected_logits(self.receiver, embeddings, mask, self.ids), message

    def manifest(self):
        def describe(cache):
            digest = hashlib.sha256()
            size = 0
            for text, value in sorted(cache.items()):
                array = np.ascontiguousarray(value.detach().cpu().numpy())
                digest.update(text.encode() + b"\0")
                digest.update(json.dumps(list(array.shape)).encode() + b"\0")
                digest.update(array.tobytes())
                size += array.nbytes
            return {"unique_entries": len(cache), "bytes": size, "ordered_content_sha256": digest.hexdigest()}
        return {"source": describe(self.sources), "question": describe(self.questions),
                "receiver_embeddings": describe(self.static), "encoding_seconds": self.encoding_seconds,
                "actual_encoded_sequences": self.encoding_sequences,
                "actual_encoder_forward_calls": self.encoding_forward_calls,
                "logical_cache_requests": self.logical_requests,
                "logical_requests_are_not_latency_measurements": True,
                "question_norms": {text: float(value.norm()) for text, value in sorted(self.questions.items())}}


def evaluate(features, bridge, seed, arm, cases, condition, source_cases=None,
             wrong_question=False, zero=False):
    import torch
    from torch.nn import functional as F
    rows, totals = [], {}
    if bridge is not None:
        bridge.eval()
    with torch.no_grad():
        for offset in range(0, len(cases), 16):
            batch = cases[offset:offset + 16]
            supplied = None if source_cases is None else source_cases[offset:offset + 16]
            logits, message = features.score(bridge, seed, arm, batch, supplied, wrong_question, zero)
            gold = torch.from_numpy(core.target_indices(batch))
            losses = F.cross_entropy(logits, gold, reduction="none")
            if not bool(torch.isfinite(losses).all()):
                raise ValueError("non-finite evaluation CE")
            choices = logits.argmax(dim=-1).tolist()
            if message is not None:
                _, terms = core.quartet_loss(logits, gold, message, arm == "aware_aux")
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.) + float(value) * len(batch)
            for index, (case, choice, ce) in enumerate(zip(batch, choices, losses.tolist())):
                row = {"seed": seed, "condition": condition, "family": case["family"],
                       "query_index": case["query_index"], "variant": case["variant"],
                       "choice": choice, "gold": int(gold[index]), "restricted_ce": ce}
                if supplied is not None:
                    row["source_family"] = supplied[index]["family"]
                    row["source_gold"] = COLORS.index(supplied[index]["member"]["answer"])
                    row["source_answer_unchanged"] = row["source_gold"] == row["gold"]
                rows.append(row)
    terms = {name: value / len(cases) for name, value in totals.items()}
    terms["restricted_ce"] = float(np.mean([row["restricted_ce"] for row in rows]))
    return rows, terms


def compress_predictions(output):
    raw_path = output / "predictions.jsonl"
    if not raw_path.is_file():
        return None
    raw = raw_path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    if gzip.decompress(compressed) != raw:
        raise ValueError("prediction compression changed bytes")
    (output / "predictions.jsonl.gz").write_bytes(compressed)
    return {"records": raw.count(b"\n"), "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed": file_metadata(output / "predictions.jsonl.gz")}


def data_preflight(data_dir, source_tokenizer, receiver_tokenizer):
    partitions = load_partitions(data_dir, include_test=True)
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    if validate_partitions(partitions) != manifest["audit"]:
        raise ValueError("data audit differs from its manifest")
    tokens = core.tokenizer_preflight(
        partitions, {"sender": source_tokenizer, "receiver": receiver_tokenizer})
    byte_counts = {}
    for split, quartets in partitions.items():
        cases = core.flattened_cases(quartets)
        byte_counts[split] = {
            "source_text_utf8_bytes": core.stats(len(case["member"]["source"].encode()) for case in cases),
            "symbolic_utf8_bytes": core.stats(len(case["member"]["symbolic"].encode()) for case in cases),
            "public_question_utf8_bytes": core.stats(len(case["query"].encode()) for case in cases),
        }
    # Only manifest/statistics leave this data-only scope. No test model cases
    # or hidden states are retained for fitting.
    return manifest, tokens, byte_counts


def make_report(result):
    status = result["execution_status"]
    lines = ["# Binding-sensitive message study", "", "Status: **" + status + "**.", "",
             "The frozen eight-object bijection task removes color-presence shortcuts. "
             "The main arms use identical edited examples and quartet CE; only aware_aux receives "
             "the fixed raw-message invariance/separation loss.", ""]
    gate = result.get("competence_gate")
    if gate is not None:
        lines += ["## Full-text validation gate", "",
                  "| Seed | Both correct | Nuisance stability | Reorder stability | Passed |",
                  "|---|---:|---:|---:|---|"]
        for row in gate["per_seed"]:
            m = row["validation"]["metrics"]
            lines.append(f"| {row['seed']} | {m['target_paired_both_correct']:.2%} | "
                         f"{m['nuisance_stability']:.2%} | {m['reorder_stability']:.2%} | {row['passed']} |")
    if status == "receiver_limited_binding_study":
        lines += ["", "The predeclared competence gate failed. Only the three text fits were completed. "
                  "No latent/symbolic fits or fresh test model scoring occurred. "
                  "This is a valid stopped study, not evidence that useful latent communication is impossible."]
    elif "summary" in result:
        primary = next(row for row in result["summary"]["paired_comparisons"]
                       if row["contrast"] == "primary_aux_minus_ce")
        lo, hi = primary["family_cluster_95_interval"]
        lines += ["", "## Fresh outcomes", "",
                  f"Primary aware_aux minus aware_ce target-pair difference: "
                  f"**{100 * primary['mean_difference']:+.4f} percentage points**, "
                  f"descriptive 95% family interval **[{100 * lo:+.4f}, {100 * hi:+.4f}] points**.", "",
                  "| Condition | Target pair both correct | Record accuracy | Nuisance stability | Reorder stability |",
                  "|---|---:|---:|---:|---:|"]
        for row in result["summary"]["conditions"]:
            m = row["metrics"]
            lines.append(f"| {row['condition']} | {m['target_paired_both_correct']['mean']:.2%} | "
                         f"{m['record_accuracy']['mean']:.2%} | {m['nuisance_stability']['mean']:.2%} | "
                         f"{m['reorder_stability']['mean']:.2%} |")
        lines += ["", "Predeclared advancement components: " +
                  json.dumps(result["advancement"], sort_keys=True) + "."]
    lines += ["", "## Interpretation and costs", "",
              "The 64-byte latent packet is 32 times the exact two-byte permutation-rank payload. "
              "The latter is a hand-written task-specific codec, separate from the trained symbolic LM arm. "
              "A query-aware request also carries its public-question UTF-8 bytes. "
              "Equal trainable parameters and updates do not imply equal input lengths, FLOPs or runtime.",
              "",
              "Stable wrong predictions are not successful invariance. A private-constant target pair "
              "is structurally unable to answer both distinct labels correctly. Source-following metrics "
              "score wrong-source predictions against the supplied world's answers as well as original labels.",
              "",
              "The joint auxiliary comparison does not isolate invariance from separation. It uses "
              "task-dependent relational supervision, fixed templates, sparse edits and one frozen model pair. "
              "Intervals are descriptive over 64 whole families conditional on three fixed seeds. "
              "No new method, universal language, network speedup, production-cost saving or breakthrough is claimed.",
              "",
              "Selected NPZ states, histories, manifests, validation outputs and any completed fresh prediction "
              "blocks are retained. An incomplete run is not a complete-study primary result."]
    return "\n".join(lines) + "\n"


def run(args):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in (
        "history.jsonl", "fits.json", "predictions.jsonl", "results.json", "selection_frozen.json"
    )):
        raise ValueError("refusing a silent restart or overwrite of binding outcomes")
    if args.threads != 2:
        raise ValueError("the frozen study requires two PyTorch threads")
    started = time.perf_counter()
    fits, completed_conditions, all_predictions = [], [], []
    timings = {}
    state = {"phase": "manifest_preflight", "complete": False,
             "fresh_test_model_cases_loaded": False, "completed_fits": [],
             "completed_conditions": [], "config": core.CONFIG}
    def status(phase, **extra):
        state.update(phase=phase, **extra)
        state["completed_fits"] = [{"seed": fit["seed"], "arm": fit["arm"]} for fit in fits]
        state["completed_conditions"] = list(completed_conditions)
        write_json(output / "status.json", state)
    status("manifest_preflight")
    code_manifest, model_manifest, runtime = verify_manifests(args, output)
    torch.set_num_threads(args.threads)
    before = time.perf_counter()
    options = {"local_files_only": True, "trust_remote_code": False}
    source_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, **options)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, **options)
    sender = AutoModel.from_pretrained(args.sender_model, use_safetensors=True, **options).float().eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, use_safetensors=True, **options).float().eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    sender_dim = sender.config.hidden_size
    receiver_dim = receiver.get_input_embeddings().weight.shape[-1]
    if (sender_dim, receiver_dim) != (768, 128):
        raise ValueError("pinned model hidden dimensions differ")
    timings["model_loading"] = time.perf_counter() - before
    status("complete_data_tokenization_preflight")
    before = time.perf_counter()
    data_manifest, token_manifest, byte_counts = data_preflight(
        args.data_dir, source_tokenizer, receiver_tokenizer)
    write_json(output / "dataset_manifest.json", data_manifest)
    write_json(output / "tokenization_manifest.json", token_manifest)
    timings["data_and_tokenization_preflight"] = time.perf_counter() - before
    fitting = load_partitions(args.data_dir, include_test=False)
    train_cases = core.flattened_cases(fitting["train"])
    validation_cases = core.flattened_cases(fitting["validation"])
    initial_states, controls, initial_records, control_arrays = {}, {}, {}, {}
    for seed in core.SEEDS:
        torch.manual_seed(seed)
        bridge = core.make_query_bridge(sender_dim, receiver_dim, core.BOTTLENECK, core.SLOTS)
        initial_states[seed] = copy.deepcopy(bridge.state_dict())
        if bool((bridge.gates != 0).any()):
            raise ValueError("query gates must start at zero")
        initial = core.save_state(output / "initial" / f"{seed}.npz", initial_states[seed])
        initial["path"] = str(Path(initial["path"]).relative_to(output))
        fixed_hidden, fixed_mask = core.constant_observation(sender_dim, seed)
        fixed_question = core.constant_question(sender_dim, seed)
        controls[seed] = (fixed_hidden, fixed_mask, fixed_question)
        control_state = {"hidden": fixed_hidden, "mask": fixed_mask, "question": fixed_question}
        initial_records[str(seed)] = {
            "checkpoint": initial, "state_sha256": core.state_hash(initial_states[seed]),
            "control_state_sha256": core.state_hash(control_state),
            "fixed_private_sha256": core.state_hash({"hidden": fixed_hidden, "mask": fixed_mask}),
            "fixed_question_sha256": core.state_hash({"question": fixed_question}),
            "fixed_question_norm": float(fixed_question.norm()),
            "trainable_parameters": sum(parameter.numel() for parameter in bridge.parameters()),
            "epoch_order_sha256": core.order_hashes(core.epoch_orders(seed))}
        for name, value in control_state.items():
            control_arrays[f"seed{seed}__{name}"] = value.numpy()
    np.savez_compressed(output / "control_tensors.npz", **control_arrays)
    write_json(output / "initialization_manifest.json", initial_records)
    features = FrozenFeatures(sender, receiver, source_tokenizer, receiver_tokenizer,
                              torch.tensor(token_manifest["candidate_token_ids"]), controls)
    status("prepare_text_fitting_features")
    before = time.perf_counter()
    features.prepare(train_cases + validation_cases, private=False, receiver_arms=("text", "aware_ce"))
    timings["text_fitting_feature_preparation"] = time.perf_counter() - before
    write_json(output / "feature_manifest.json", features.manifest())
    gate = None

    def load_checked(fit):
        path = output / fit["checkpoint"]["path"]
        metadata = file_metadata(path)
        if metadata["sha256"] != fit["checkpoint"]["sha256"] or metadata["bytes"] != fit["checkpoint"]["bytes"]:
            raise ValueError("selected NPZ bytes changed")
        selected = core.load_state(path)
        if core.state_hash(selected) != fit["state_sha256"]:
            raise ValueError("selected NPZ tensors differ from the selected state")
        model = core.make_query_bridge(sender_dim, receiver_dim, core.BOTTLENECK, core.SLOTS)
        model.load_state_dict(selected, strict=True)
        return model.eval()

    def fit_arm(seed, arm):
        core.authorize_fit(arm, fits, gate)
        status("fitting", active_seed=seed, active_arm=arm, epoch=0)
        model = core.make_query_bridge(sender_dim, receiver_dim, core.BOTTLENECK, core.SLOTS)
        model.load_state_dict(initial_states[seed], strict=True)
        initial_hash = core.state_hash(model.state_dict())
        if initial_hash != initial_records[str(seed)]["state_sha256"]:
            raise ValueError("arm initialization differs from its seed peers")
        optimizer = torch.optim.AdamW(model.parameters(), lr=core.LEARNING_RATE, weight_decay=core.WEIGHT_DECAY)
        history, best_loss, best_state, best_epoch, updates = [], float("inf"), None, None, 0
        orders = core.epoch_orders(seed)
        training_seconds = validation_seconds = 0.
        for epoch, order in enumerate(orders, 1):
            model.train()
            before = time.perf_counter()
            totals = {}
            for offset in range(0, len(order), core.QUARTETS_PER_BATCH):
                indices = order[offset:offset + core.QUARTETS_PER_BATCH]
                cases = [train_cases[4 * index + member] for index in indices for member in range(4)]
                optimizer.zero_grad(set_to_none=True)
                logits, messages = features.score(model, seed, arm, cases)
                gold = torch.from_numpy(core.target_indices(cases))
                loss, terms = core.quartet_loss(logits, gold, messages, auxiliary=arm == "aware_aux")
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite binding training objective")
                loss.backward()
                if updates == 0 and any(parameter.grad is not None for base in (sender, receiver)
                                        for parameter in base.parameters()):
                    raise ValueError("a frozen base model acquired parameter gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(), core.CLIP_NORM)
                optimizer.step()
                updates += 1
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.) + float(value.detach()) * len(indices)
            epoch_training = time.perf_counter() - before
            training_seconds += epoch_training
            before = time.perf_counter()
            validation_rows, validation_terms = evaluate(
                features, model, seed, arm, validation_cases, arm)
            epoch_validation = time.perf_counter() - before
            validation_seconds += epoch_validation
            validation_ce = validation_terms["restricted_ce"]
            row = {"seed": seed, "arm": arm, "epoch": epoch, "updates": updates,
                   "training": {name: value / 512 for name, value in totals.items()},
                   "validation": validation_terms, "validation_ce": validation_ce,
                   "query_gates": model.gates.detach().tolist(),
                   "training_seconds": epoch_training, "validation_seconds": epoch_validation}
            history.append(row)
            append_jsonl(output / "history.jsonl", [row])
            if validation_ce < best_loss:
                best_loss, best_state, best_epoch = validation_ce, copy.deepcopy(model.state_dict()), epoch
                core.save_state(output / "in_progress" / f"{seed}-{arm}.npz", best_state)
            status("fitting", active_seed=seed, active_arm=arm, epoch=epoch, updates=updates)
            print(json.dumps({"phase": "fitting", "seed": seed, "arm": arm,
                              "epoch": epoch, "validation_ce": validation_ce, "updates": updates}), flush=True)
        if best_epoch != core.select_epoch(history) or updates != 1024:
            raise ValueError("checkpoint selection or update budget differs")
        model.load_state_dict(best_state)
        checkpoint = core.save_state(output / "selected" / str(seed) / (arm + ".npz"), best_state)
        checkpoint["path"] = str(Path(checkpoint["path"]).relative_to(output))
        before = time.perf_counter()
        selected_rows, selected_terms = evaluate(features, model, seed, arm, validation_cases, arm)
        selected_validation_seconds = time.perf_counter() - before
        record = {
            "seed": seed, "arm": arm, "complete": True, "epochs": core.EPOCHS,
            "updates": updates, "initial_state_sha256": initial_hash,
            "epoch_order_sha256": core.order_hashes(orders),
            "selected_epoch": best_epoch, "state_sha256": core.state_hash(best_state),
            "checkpoint": checkpoint, "query_gates": model.gates.detach().tolist(),
            "validation": core.validation_metrics(selected_rows), "validation_terms": selected_terms,
            "history": history, "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds, "validation_seconds": validation_seconds,
            "selected_validation_seconds": selected_validation_seconds}
        load_checked(record)
        append_jsonl(output / "validation_predictions.jsonl", selected_rows)
        fits.append(record)
        write_json(output / "fits.json", fits)
        status("fit_complete", active_seed=seed, active_arm=arm)
        return record

    # The text competence controls are all trained and selected first.
    for seed in core.SEEDS:
        fit_arm(seed, "text")
    gate = core.competence_gate(fits)
    write_json(output / "competence_gate.json", gate)
    write_json(output / "feature_manifest.json", features.manifest())

    def base_result(execution_status):
        return {
            "execution_status": execution_status, "config": core.CONFIG,
            "competence_gate": gate, "fits": fits, "data_manifest_sha256": file_metadata(output / "dataset_manifest.json")["sha256"],
            "code_manifest": code_manifest, "model_manifest": model_manifest, "runtime": runtime,
            "candidate_colors": list(COLORS), "candidate_token_ids": token_manifest["candidate_token_ids"],
            "timings_seconds": {**timings, "training": sum(row["training_seconds"] for row in fits),
                               "validation": sum(row["validation_seconds"] + row["selected_validation_seconds"] for row in fits),
                               "study_runner_so_far": time.perf_counter() - started},
            "accounting": {
                "utf8_bytes_by_partition": byte_counts, "latent_payload_float32_bytes": 64,
                "exact_complete_world_rank_payload_bytes": 2, "latent_to_rank_payload_ratio": 32,
                "public_question_feedback_charge": "full query UTF-8 bytes for every query-aware request",
                "rank_packet_reusable_across_queries": True,
                "selected_checkpoint_bytes": sum(row["checkpoint"]["bytes"] for row in fits),
                "model_download_seconds": {role: entry.get("download_seconds")
                                           for role, entry in model_manifest["models"].items()},
                "cache": features.manifest(), "network_energy_production_speed_measured": False},
            "completed_conditions": list(completed_conditions),
            "fresh_test_model_cases_loaded": state["fresh_test_model_cases_loaded"],
            "limits": ["One frozen model pair, fixed eight-object grammar and finite bijective task.",
                       "Known quartet relationships provide additional task-dependent supervision.",
                       "Equal adaptation steps and parameters do not imply equal FLOPs.",
                       "Wrong-source replacement includes another family's rendering order; its effect is not a pure binding-only intervention.",
                       "A private-constant target pair has structurally zero both-correct accuracy.",
                       "Stable wrong predictions do not establish robustness.",
                       "Family intervals are conditional on three fixed seeds.",
                       "No new method or breakthrough claim is implied."]}
    if not gate["passed"]:
        status("receiver_limited", complete=True, scientific_status="receiver_limited",
               other_arms_started=False, fresh_test_model_cases_loaded=False)
        result = base_result("receiver_limited_binding_study")
        write_json(output / "results.json", result)
        (output / "report.md").write_text(make_report(result), encoding="utf-8")
        print(json.dumps({"status": "receiver_limited", "failed_seeds": gate["failed_seeds"]}), flush=True)
        return result

    status("prepare_remaining_fitting_features")
    before = time.perf_counter()
    features.prepare(train_cases + validation_cases, private=True, receiver_arms=("symbolic", "aware_ce"))
    timings["remaining_fitting_feature_preparation"] = time.perf_counter() - before
    write_json(output / "feature_manifest.json", features.manifest())
    for seed in core.SEEDS:
        for arm in core.ARMS:
            if arm != "text":
                fit_arm(seed, arm)
    selection = core.freeze_selection(fits, gate)
    write_json(output / "selection_frozen.json", selection)
    core.require_test_authorized(output / "selection_frozen.json", fits, gate)
    # Verify every saved checkpoint, not only in-memory selections, before
    # loading fresh cases or computing any fresh model state.
    selected_models = {(fit["seed"], fit["arm"]): load_checked(fit) for fit in fits}
    status("fresh_selection_verified")

    test_quartets = load_partitions(args.data_dir, include_test=True)["test"]
    test_cases = core.flattened_cases(test_quartets)
    status("prepare_fresh_features", fresh_test_model_cases_loaded=True)
    before = time.perf_counter()
    features.prepare(test_cases, private=True, receiver_arms=("text", "symbolic", "aware_ce"))
    timings["fresh_feature_preparation"] = time.perf_counter() - before
    write_json(output / "feature_manifest.json", features.manifest())
    successor = core.source_successor_map(test_quartets)
    source_cases = core.flattened_cases([test_quartets[index] for index in successor])
    write_json(output / "source_intervention_manifest.json", {
        "rule": "cyclic successor of sorted families; preserve public query and quartet member",
        "quartet_successor_indices": successor,
        "same_for_all_seeds_and_arms": True,
        "uses_answer_values": False})
    fresh_started = time.perf_counter()
    evaluation_terms = []
    for seed in core.SEEDS:
        for condition in core.MODEL_CONDITIONS:
            status("fresh_evaluation", active_seed=seed, active_condition=condition)
            base_arm = condition
            intervention = None
            for suffix in ("wrong_source", "wrong_question", "zero"):
                if condition.endswith("_" + suffix):
                    base_arm = condition[:-(len(suffix) + 1)]
                    intervention = suffix
                    break
            model = None if base_arm == "frozen_no_channel" else selected_models[(seed, base_arm)]
            before = time.perf_counter()
            rows, terms = evaluate(
                features, model, seed, base_arm, test_cases, condition,
                source_cases=source_cases if intervention == "wrong_source" else None,
                wrong_question=intervention == "wrong_question", zero=intervention == "zero")
            append_jsonl(output / "predictions.jsonl", rows)
            all_predictions.extend(rows)
            completed_conditions.append({"seed": seed, "condition": condition, "records": len(rows)})
            evaluation_terms.append({"seed": seed, "condition": condition, "terms": terms,
                                     "seconds": time.perf_counter() - before})
            write_json(output / "evaluation_conditions.json", evaluation_terms)
            status("fresh_condition_complete", active_seed=seed, active_condition=condition)
            print(json.dumps({"phase": "fresh_condition_complete", "seed": seed,
                              "condition": condition, "records": len(rows)}), flush=True)
        oracle_rows = []
        for case in test_cases:
            # This explicit hand-written oracle is isolated from every neural arm.
            private_world = parse_source(case["member"]["source"])
            packet = encode_rank(private_world)
            reconstructed = decode_rank(packet)
            query_index = parse_question(case["query"])
            oracle_rows.append({
                "seed": seed, "condition": "rank_codec_oracle", "family": case["family"],
                "query_index": case["query_index"], "variant": case["variant"],
                "choice": reconstructed[query_index],
                "gold": COLORS.index(case["member"]["answer"]), "restricted_ce": None,
                "payload_bytes": len(packet)})
        append_jsonl(output / "predictions.jsonl", oracle_rows)
        all_predictions.extend(oracle_rows)
        completed_conditions.append({"seed": seed, "condition": "rank_codec_oracle", "records": len(oracle_rows)})
        status("fresh_condition_complete", active_seed=seed, active_condition="rank_codec_oracle")
    timings["fresh_evaluation"] = time.perf_counter() - fresh_started
    summary = core.summarize_predictions(all_predictions)
    metric_map = {row["condition"]: row["metrics"] for row in summary["conditions"]}
    primary = next(row for row in summary["paired_comparisons"] if row["contrast"] == "primary_aux_minus_ce")
    source_contrast = next(row for row in summary["paired_comparisons"]
                           if row["contrast"] == "aware_aux_minus_wrong_source")
    auxiliary = metric_map["aware_aux"]
    components = {
        "text_validation_gate": gate["passed"],
        "fresh_paired_at_least_80_percent": auxiliary["target_paired_both_correct"]["mean"] >= .8,
        "fresh_nuisance_stability_at_least_90_percent": auxiliary["nuisance_stability"]["mean"] >= .9,
        "fresh_reorder_stability_at_least_90_percent": auxiliary["reorder_stability"]["mean"] >= .9,
        "positive_mean_correct_vs_wrong_source_pair_difference": source_contrast["mean_difference"] > 0,
    }
    advance = all(components.values())
    positive_interval = primary["family_cluster_95_interval"][0] > 0
    integrity = compress_predictions(output)
    if integrity["records"] != len(all_predictions):
        raise ValueError("retained prediction count differs")
    expected_raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                           for row in all_predictions).encode()
    if hashlib.sha256(expected_raw).hexdigest() != integrity["raw_sha256"]:
        raise ValueError("retained prediction contents differ from completed records")
    choice_lookup = {(row["condition"], row["seed"], row["family"], row["query_index"], row["variant"]): row["choice"]
                     for row in all_predictions}
    intervention_diagnostics = []
    for arm in ("aware_ce", "aware_aux"):
        for intervention in ("wrong_source", "wrong_question", "zero"):
            for seed in core.SEEDS:
                matched = [row for row in all_predictions if row["condition"] == arm and row["seed"] == seed]
                changed = sum(
                    row["choice"] != choice_lookup[(arm + "_" + intervention, seed, row["family"],
                                                   row["query_index"], row["variant"])]
                    for row in matched)
                intervention_diagnostics.append({"arm": arm, "intervention": intervention, "seed": seed,
                                                 "changed_choices": changed, "records": len(matched)})
    result = base_result("executed_binding_competence_study")
    result.update(
        complete_study=True, summary=summary, selection=selection,
        prediction_integrity=integrity, intervention_diagnostics=intervention_diagnostics,
        advancement={"components": components, "practical_advance": advance,
                     "primary_interval_strictly_positive": positive_interval,
                     "supported_comparative_improvement": advance and positive_interval,
                     "breakthrough_claim": False},
        oracle={"type": "hand-written exact permutation-rank codec and lookup",
                "deterministic_rows_repeated_for_paired_seed_summary": True,
                "distinct_from_trained_symbolic_lm": True})
    write_json(output / "feature_manifest.json", features.manifest())
    write_json(output / "results.json", result)
    (output / "report.md").write_text(make_report(result), encoding="utf-8")
    status("complete", complete=True, scientific_status=result["execution_status"])
    print(json.dumps({"status": "complete", "prediction_records": len(all_predictions),
                      "primary_difference": primary["mean_difference"],
                      "advancement": result["advancement"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sender-model", "receiver-model", "data-dir", "output",
                 "model-manifest", "code-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    existing_output = Path(args.output)
    if any((existing_output / name).exists() for name in (
        "history.jsonl", "fits.json", "predictions.jsonl", "results.json", "selection_frozen.json"
    )):
        parser.exit(2, "Refusing to modify existing binding outcomes or retry a partial run.\n")
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        status_path = output / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        previous.update(phase="stopped_with_recorded_failure", complete=False,
                        failure={"type": type(error).__name__, "message": str(error)})
        write_json(status_path, previous)
        integrity = compress_predictions(output)
        if not (output / "results.json").exists():
            failure_result = {
                "execution_status": "stopped_with_recorded_failure", "complete_study": False,
                "config": core.CONFIG, "status": previous, "prediction_integrity": integrity,
                "primary_complete_study_claim_permitted": False}
            write_json(output / "results.json", failure_result)
            (output / "report.md").write_text(
                "# Binding study stopped\n\nThe fixed run stopped with a recorded failure. "
                "Completed fits and prediction blocks remain retained. "
                "No complete-study primary claim is permitted.\n\n" +
                json.dumps(previous["failure"], indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
