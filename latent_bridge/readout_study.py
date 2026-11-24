"""Run the fixed frozen-message readout diagnostic after reconstruction passes."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .counterfactual_runner import candidate_ids, flatten, selected_logits, source_derangement
from .private_counterfactual import COLORS, OBJECTS, VIEWS
from .query_conditioned import EVALUATED_ARMS, TRAINED_ARMS, constant_question, make_query_bridge, normalize_question, wrong_public_question
from .query_conditioned_data import load_partitions as load_original
from .readout_data import load_partitions
from .readout_probes import (LINEAR_ARMS, MESSAGE_ARMS, fit_preprocessing, flatten_source, freeze_selection,
    label_derangement, make_probe, pool_before_compressor, preprocess, public_head, secondary_trigger, state_hash, summarize)
from .readout_reconstruction import verify_reconstruction
from .transformers_runner import constant_observation, fingerprint_directory


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_metadata(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def true_targets(cases):
    return np.asarray([COLORS.index(case["member"]["answer"]) for case in cases], dtype=np.int64)


def validation_pair_accuracy(cases, choices):
    pairs = {}
    for case, choice in zip(cases, choices):
        key = (case["family"], public_head(case["query"]))
        pairs.setdefault(key, {})[case["variant"]] = int(choice) == COLORS.index(case["member"]["answer"])
    if any(set(pair) != {"original", "counterfactual"} for pair in pairs.values()):
        raise ValueError("validation decision requires complete original/target pairs")
    return float(np.mean([pair["original"] and pair["counterfactual"] for pair in pairs.values()]))


def make_report(result):
    primary = next(row for row in result["summary"]["paired_comparisons"] if row["view"] == "iid" and row["contrast"] == "primary_accessibility_gap")
    interval = primary["paired_family_cluster_95_percentile_interval"]
    lines = ["# Frozen-message readout diagnostic", "",
        "Reconstruction passed the predeclared behavioral gate. Historical message byte identity cannot be established because old final checkpoint hashes were absent.", "",
        f"The primary linear-aware probe minus original receiver difference is **{100*primary['mean_difference']:+.4f} percentage points**, with descriptive family interval **[{100*interval[0]:+.4f}, {100*interval[1]:+.4f}] points**.", "",
        f"Validation-only secondary MLP triggered: **{result['selection']['trigger']['triggered']}**. The linear primary was retained.", "",
        "The probe receives extra supervised readout fitting and an explicit parser of public question text. Larger source/pool probes have more parameters. Results measure accessibility to these readers, not a superior agent, universal language or new method.", "",
        "| View | Condition | Target pair both correct | Target record accuracy | Nuisance both correct | Nuisance stability |",
        "|---|---|---:|---:|---:|---:|"]
    for row in result["summary"]["conditions"]:
        metric = row["metrics"]
        lines.append(f"| {row['view']} | {row['condition']} | {metric['paired_both_correct']['mean']:.2%} | {metric['record_accuracy']['mean']:.2%} | {metric['nuisance_both_correct']['mean']:.2%} | {metric['nuisance_invariance']['mean']:.2%} |")
    lines += ["", "## Limits and retention", "",
        "The 32 fresh families exclude all 2,002 prior pretrained-study worlds. Probe training/validation explicitly reuse the preceding study's fitting families. Four views and three paired seeds are averaged within family clusters, not counted as independent worlds.", "",
        f"Full-source linear validation competence (80% record accuracy per seed): {result['full_source_validation_competence']}. Failed seeds are retained.", "",
        "Reliable message-probe recovery with poor original-receiver performance shows information accessible to that additional reader. A failed linear or small MLP probe does not prove information is absent. Larger raw/pool controls cannot uniquely attribute a gap to irreversible compression loss because their capacity and positional access differ.", "",
        "Use the prespecified 80% fresh paired accuracy and 90% nuisance stability targets together with wrong-source controls when assessing practical utility. These targets do not define scientific novelty. The shuffled-training-label control and all negative outcomes remain reported.", "",
        "Compressed bridge tensors, 16-coordinate feature caches and small-probe weights are exported for durable repository publication. Larger checkpoints and normalization statistics are retained in the workflow artifact; its 90-day retention is finite. Full-source caches are represented by hashes and a regeneration recipe rather than a large repository upload.", "",
        "No paid API, production-latency measurement or bandwidth claim is part of this diagnostic."]
    return "\n".join(lines) + "\n"


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists() or (output / "results.json").exists():
        raise ValueError("output already contains fresh results; silent restart/overwrite is forbidden")
    started = time.perf_counter()
    gate = verify_reconstruction(args.reference_dir, args.reconstruction_dir, args.source_manifest, Path(__file__).resolve().parents[1])
    (output / "reconstruction_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    if not gate["passed"]:
        raise ValueError("reconstruction gate failed; no probe fitting or fresh scoring is permitted")
    partitions = load_partitions(args.data_dir)
    dataset_manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    if dataset_manifest["generation_seed"] != 20261126 or any(len(partitions["test_" + view]) != 128 for view in VIEWS):
        raise ValueError("fresh test must use the frozen seed and 32 families")
    original = load_original(args.original_data_dir)
    if any(partitions[name] != original[name] for name in ("train", "validation")):
        raise ValueError("provided fitting data do not match the fresh-data manifest")
    reconstruction = json.loads((Path(args.reconstruction_dir) / "results.json").read_text())
    config = reconstruction["config"]
    if args.seeds != [101, 202, 303] or config["seeds"] != args.seeds:
        raise ValueError("this fixed diagnostic requires paired seeds 101, 202 and 303 in order")
    if (args.epochs, args.batch_size, args.learning_rate, args.weight_decay, args.max_source_positions, args.threads) != (80, 64, 0.001, 0.01, 64, 2):
        raise ValueError("probe settings differ from the frozen design")
    import torch
    import transformers
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    load_start = time.perf_counter()
    options = {"local_files_only": True, "trust_remote_code": False}
    source_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, **options)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, **options)
    sender = AutoModel.from_pretrained(args.sender_model, **options).float().eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, **options).float().eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    actual_model_files = {"sender": fingerprint_directory(args.sender_model), "receiver": fingerprint_directory(args.receiver_model)}
    if actual_model_files != reconstruction["model_files"]:
        raise ValueError("local pinned model files differ from the accepted reconstruction")
    ids = torch.tensor(candidate_ids(receiver_tokenizer))
    if ids.tolist() != reconstruction["candidate_token_ids"]:
        raise ValueError("receiver candidate tokens differ from reconstruction")
    exports = output / "exports"
    exports.mkdir(exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    bridges, fixed, bridge_export, bridge_manifest = {}, {}, {}, {}
    for seed in args.seeds:
        bridges[seed] = {}
        fixed_private, fixed_mask = constant_observation(config["sender_dim"], seed)
        fixed[seed] = (fixed_private, fixed_mask, constant_question(config["sender_dim"], seed))
        recorded_run = next(row for row in reconstruction["runs"] if row["seed"] == seed)
        for arm in TRAINED_ARMS:
            source_path = Path(args.reconstruction_dir) / "checkpoints" / str(seed) / (arm + ".pt")
            state = torch.load(source_path, map_location="cpu", weights_only=True)
            bridge = make_query_bridge(config["sender_dim"], config["receiver_dim"], config["bottleneck"], config["slots"])
            bridge.load_state_dict(state)
            bridge.eval()
            for parameter in bridge.parameters():
                parameter.requires_grad_(False)
            if bridge.gates.tolist() != recorded_run["selected_query_gates"][arm]:
                raise ValueError("selected checkpoint gates differ from reconstruction output")
            bridges[seed][arm] = bridge
            destination = checkpoint_dir / "reconstructed" / str(seed) / (arm + ".pt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source_path.read_bytes())
            bridge_manifest[f"{seed}/{arm}"] = {"tensor_sha256": state_hash(state), **file_metadata(destination)}
            for name, value in state.items():
                bridge_export[f"seed{seed}__{arm}__{name}"] = value.numpy()
    np.savez_compressed(exports / "bridge_weights.npz", **bridge_export)
    (output / "bridge_manifest.json").write_text(json.dumps(bridge_manifest, indent=2) + "\n")
    model_loading_seconds = time.perf_counter() - load_start
    source_cache, question_cache, static_cache, representation_cache, full_source_cache = {}, {}, {}, {}, {}
    source_seconds = question_seconds = 0.0

    def source_states(text):
        nonlocal source_seconds
        if text not in source_cache:
            before = time.perf_counter()
            tokens = source_tokenizer(text, return_tensors="pt", truncation=False)
            if tokens["input_ids"].shape[1] > config["max_length"]:
                raise ValueError("sender context overflow; truncation forbidden")
            with torch.no_grad():
                source_cache[text] = sender(**tokens).last_hidden_state.squeeze(0).detach()
            source_seconds += time.perf_counter() - before
        return source_cache[text]

    def question_vector(text):
        nonlocal question_seconds
        if text not in question_cache:
            before = time.perf_counter()
            tokens = source_tokenizer(text, return_tensors="pt", truncation=False)
            with torch.no_grad():
                hidden = sender(**tokens).last_hidden_state.squeeze(0)
                question_cache[text] = normalize_question(hidden.mean(dim=0)).detach()
            question_seconds += time.perf_counter() - before
        return question_cache[text]

    def representation(seed, case, arm, wrong_question=False):
        source, question = case["member"]["source"], case["query"]
        if arm == "message_aware_shuffled_labels":
            arm = "message_aware"
        if arm == "query_only":
            key = (seed, arm)
            if key not in representation_cache:
                representation_cache[key] = np.random.default_rng(seed + 56039).standard_normal(16).astype(np.float32)
            return representation_cache[key]
        if arm == "full_source":
            if source not in full_source_cache:
                full_source_cache[source] = flatten_source(source_states(source).numpy(), args.max_source_positions)
            return full_source_cache[source]
        mapping = {"message_aware": "query_aware", "message_blind": "query_blind", "message_private_constant": "private_constant",
                   "pooled_aware": "query_aware", "pooled_blind": "query_blind", "receiver_text": "text", "receiver_symbolic": "symbolic"}
        bridge_arm = mapping[arm]
        if wrong_question:
            question = wrong_public_question(question)
        effective_source = "fixed" if bridge_arm in ("private_constant", "text", "symbolic") else source
        effective_question = "fixed" if bridge_arm == "query_blind" else question
        key = (seed, arm, effective_source, effective_question)
        if key not in representation_cache:
            bridge = bridges[seed][bridge_arm]
            fixed_private, fixed_mask, fixed_query = fixed[seed]
            if bridge_arm in ("private_constant", "text", "symbolic"):
                hidden, mask = fixed_private, fixed_mask
            else:
                hidden = source_states(source)[None]
                mask = torch.ones(hidden.shape[:2])
            query = fixed_query if bridge_arm == "query_blind" else question_vector(question)
            with torch.no_grad():
                value = pool_before_compressor(bridge, hidden, mask, query[None]) if arm.startswith("pooled_") else bridge.encode(hidden, mask, query[None])
            representation_cache[key] = value.reshape(-1).numpy().copy()
        return representation_cache[key]

    def prepare(rows, variants):
        cases = flatten(rows, variants)
        for case in cases:
            # Reporting identity is derived from the same public parser.
            case["query_key"] = OBJECTS[public_head(case["query"])]
            source_states(case["member"]["source"])
            question_vector(case["query"])
        return cases

    preparation_start = time.perf_counter()
    cases_by_split = {"train": prepare(partitions["train"], ("original",)),
                      "validation": prepare(partitions["validation"], ("original", "counterfactual"))}
    cases_by_split.update({view: prepare(partitions["test_" + view], ("original", "counterfactual", "nuisance")) for view in VIEWS})
    train_cases, validation_cases = cases_by_split["train"], cases_by_split["validation"]
    shuffle_maps = {view: source_derangement(cases_by_split[view]) for view in VIEWS}
    feature_manifest = {"bridge_checkpoints": bridge_manifest, "representations": {}, "case_order": {},
        "scope": "Frozen inputs only; public-question head selection parsed from text; source/pooled caches represented by hashes and reconstruction recipe."}
    message_export = {}
    for split, cases in cases_by_split.items():
        feature_manifest["case_order"][split] = [{key: case[key] for key in ("family", "query", "query_key", "view", "variant")} for case in cases]
    # Extract fixed features before fitting, without reading fresh labels into
    # any training or selection function. This is not fresh outcome scoring.
    for seed in args.seeds:
        for split, cases in cases_by_split.items():
            for arm in LINEAR_ARMS:
                matrix = np.stack([representation(seed, case, arm) for case in cases])
                name = f"seed{seed}__{split}__{arm}"
                feature_manifest["representations"][name] = {"shape": list(matrix.shape), "float32_bytes": matrix.nbytes, "sha256_float32": array_hash(matrix)}
                if arm in ("message_aware", "message_blind", "message_private_constant", "query_only"):
                    message_export[name] = matrix
                del matrix
            if split in VIEWS:
                name = f"seed{seed}__{split}__message_aware_wrong_question"
                matrix = np.stack([representation(seed, case, "message_aware", True) for case in cases])
                message_export[name] = matrix
                feature_manifest["representations"][name] = {"shape": list(matrix.shape), "float32_bytes": matrix.nbytes, "sha256_float32": array_hash(matrix)}
    np.savez_compressed(exports / "message_features.npz", **message_export)
    del message_export
    feature_manifest["frozen_source_cache"] = {hashlib.sha256(text.encode()).hexdigest(): {"shape": list(value.shape), "sha256_float32": array_hash(value.numpy())} for text, value in source_cache.items()}
    feature_manifest["source_cache_recipe"] = "Regenerate data by manifests; encode each source alone using the pinned frozen sender under no_grad, retain last_hidden_state, pad right to 64 for full_source. Reconstruct bridges from durable exported state tensors to regenerate pooled features. No gold-selected positions."
    (output / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2) + "\n")
    (output / "dataset_manifest.json").write_bytes((Path(args.data_dir) / "manifest.json").read_bytes())
    preparation_seconds = time.perf_counter() - preparation_start
    selected, selected_states, statistics_by_arm, fit_records = {}, {}, {}, []
    selected_validation_pair = {}
    all_predictions = []
    train_targets = true_targets(train_cases)
    validation_targets = true_targets(validation_cases)
    shuffled_train_targets = train_targets[label_derangement(train_cases)]
    train_heads = torch.tensor([public_head(case["query"]) for case in train_cases])
    validation_heads = torch.tensor([public_head(case["query"]) for case in validation_cases])
    orders_by_seed = {}
    for seed in args.seeds:
        rng = random.Random(seed + 4242)
        orders = []
        for _ in range(args.epochs):
            order = list(range(len(train_cases)))
            rng.shuffle(order)
            orders.append(order)
        orders_by_seed[seed] = orders

    def probe_evaluate(model, features, heads, targets):
        total_loss, choices = 0.0, []
        with torch.no_grad():
            for offset in range(0, len(features), args.batch_size):
                logits = model(features[offset:offset + args.batch_size], heads[offset:offset + args.batch_size])
                target = torch.as_tensor(targets[offset:offset + args.batch_size])
                losses = F.cross_entropy(logits, target, reduction="none")
                if not bool(torch.isfinite(losses).all()):
                    raise ValueError("non-finite probe evaluation")
                total_loss += float(losses.sum())
                choices.extend(logits.argmax(dim=-1).tolist())
        return {"restricted_ce": total_loss / len(features), "record_accuracy": float(np.mean(np.asarray(choices) == targets)), "choices": choices}

    def fit_probe(seed, kind, arm):
        before = time.perf_counter()
        training = np.stack([representation(seed, case, arm) for case in train_cases])
        validation = np.stack([representation(seed, case, arm) for case in validation_cases])
        stats_key = (seed, "message_aware" if arm == "message_aware_shuffled_labels" else arm)
        if stats_key not in statistics_by_arm:
            statistics_by_arm[stats_key] = fit_preprocessing(training)
        statistics = statistics_by_arm[stats_key]
        x_train, x_validation = torch.from_numpy(preprocess(training, statistics)), torch.from_numpy(preprocess(validation, statistics))
        del training, validation
        stats_path = output / "normalization" / str(seed) / (arm + ".npz")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stats_path, mean=statistics["mean"], scale=statistics["scale"])
        preprocessing_seconds = time.perf_counter() - before
        y_training = shuffled_train_targets if arm == "message_aware_shuffled_labels" else train_targets
        torch.manual_seed(seed + 4242)
        model = make_probe(x_train.shape[1], kind)
        initial_hash = state_hash(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_loss, best_state, best_epoch, history = float("inf"), None, None, []
        training_seconds = validation_seconds = 0.0
        for epoch, order in enumerate(orders_by_seed[seed], 1):
            epoch_start = time.perf_counter()
            for offset in range(0, len(order), args.batch_size):
                batch = torch.tensor(order[offset:offset + args.batch_size])
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x_train[batch], train_heads[batch]), torch.as_tensor(y_training)[batch])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite probe training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            training_seconds += time.perf_counter() - epoch_start
            validation_start = time.perf_counter()
            val = probe_evaluate(model, x_validation, validation_heads, validation_targets)
            validation_seconds += time.perf_counter() - validation_start
            history.append({"epoch": epoch, "validation_ce": val["restricted_ce"], "validation_record_accuracy": val["record_accuracy"]})
            with (output / "training_history.jsonl").open("a") as stream:
                stream.write(json.dumps({"seed": seed, "reader": kind, "arm": arm, **history[-1]}) + "\n")
            if val["restricted_ce"] < best_loss:
                best_loss, best_epoch, best_state = val["restricted_ce"], epoch, copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({"phase": "probe_fitting", "seed": seed, "reader": kind, "arm": arm, **history[-1]}), flush=True)
        model.load_state_dict(best_state)
        val = probe_evaluate(model, x_validation, validation_heads, validation_targets)
        train_assigned = probe_evaluate(model, x_train, train_heads, y_training)
        actual_train_accuracy = float(np.mean(np.asarray(train_assigned["choices"]) == train_targets))
        pair_accuracy = validation_pair_accuracy(validation_cases, val["choices"])
        key = (seed, kind, arm)
        selected_states[key] = best_state
        selected[key] = {"selected_epoch": best_epoch, "state_sha256": state_hash(best_state)}
        if kind == "linear" and arm == "message_aware":
            selected_validation_pair[seed] = pair_accuracy
        destination = checkpoint_dir / "probes" / str(seed) / kind / (arm + ".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, destination)
        fit_records.append({"seed": seed, "probe_seed": seed + 4242, "reader": kind, "arm": arm,
            "dimension": x_train.shape[1], "parameters": sum(p.numel() for p in model.parameters()),
            "updates": args.epochs * math.ceil(len(train_cases) / args.batch_size), "initial_state_sha256": initial_hash,
            "epoch_order_sha256": [hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest() for order in orders_by_seed[seed]],
            **selected[key], "validation": {"restricted_ce": val["restricted_ce"], "record_accuracy": val["record_accuracy"], "paired_both_correct": pair_accuracy},
            "training": {"assigned_label_ce": train_assigned["restricted_ce"], "assigned_label_accuracy": train_assigned["record_accuracy"], "true_label_accuracy": actual_train_accuracy},
            "history": history, "checkpoint": {"path": str(destination.relative_to(output)), **file_metadata(destination)},
            "normalization": {"path": str(stats_path.relative_to(output)), **file_metadata(stats_path)},
            "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
            "validation_seconds": validation_seconds, "total_fitting_seconds": time.perf_counter() - before})
        (output / "fit_records.json").write_text(json.dumps(fit_records, indent=2) + "\n")

    # Every linear arm/seed is selected before the validation-only trigger.
    for seed in args.seeds:
        for arm in LINEAR_ARMS:
            fit_probe(seed, "linear", arm)
    trigger = secondary_trigger(selected_validation_pair, args.seeds)
    (output / "secondary_trigger.json").write_text(json.dumps(trigger, indent=2) + "\n")
    print(json.dumps({"phase": "secondary_decision", **trigger}), flush=True)
    if trigger["triggered"]:
        for seed in args.seeds:
            for arm in MESSAGE_ARMS:
                fit_probe(seed, "mlp", arm)
    # This persistent boundary is written before the first fresh outcome.
    selection = freeze_selection(selected, args.seeds, trigger)
    (output / "selection_frozen.json").write_text(json.dumps(selection, indent=2) + "\n")
    for seed in args.seeds:
        for kind in ("linear", "mlp"):
            records = [row for row in fit_records if row["seed"] == seed and row["reader"] == kind and row["arm"] in MESSAGE_ARMS]
            if records and (len({row["initial_state_sha256"] for row in records}) != 1 or len({row["updates"] for row in records}) != 1):
                raise ValueError("matched message probes do not share initialization/update budgets")

    small_export, normalization_export = {}, {}
    for (seed, kind, arm), state in selected_states.items():
        if arm in MESSAGE_ARMS:
            for name, value in state.items():
                small_export[f"seed{seed}__{kind}__{arm}__{name}"] = value.numpy()
    for (seed, arm), stats in statistics_by_arm.items():
        for key in ("mean", "scale"):
            normalization_export[f"seed{seed}__{arm}__{key}"] = stats[key]
            if arm in MESSAGE_ARMS:
                small_export[f"seed{seed}__normalization__{arm}__{key}"] = stats[key]
    np.savez_compressed(exports / "small_probe_weights.npz", **small_export)
    np.savez_compressed(output / "normalization.npz", **normalization_export)
    del small_export, normalization_export

    def append_predictions(seed, view, condition, cases, choices, losses):
        completed = []
        for case, choice, loss in zip(cases, choices, losses):
            completed.append({"seed": seed, "view": view, "condition": condition,
                "family": case["family"], "query_key": OBJECTS[public_head(case["query"])], "variant": case["variant"],
                "choice": int(choice), "gold": COLORS.index(case["member"]["answer"]), "restricted_ce": float(loss)})
        all_predictions.extend(completed)
        with (output / "predictions.jsonl").open("a") as stream:
            stream.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in completed))

    def evaluate_probe(seed, kind, arm, model, view, intervention="matched"):
        cases = cases_by_split[view]
        feature_cases = [cases[index] for index in shuffle_maps[view]] if intervention == "wrong_source" else cases
        features = np.stack([representation(seed, case, arm, intervention == "wrong_question") for case in feature_cases])
        stats = statistics_by_arm[(seed, "message_aware" if arm == "message_aware_shuffled_labels" else arm)]
        x = torch.from_numpy(preprocess(features, stats))
        heads = torch.tensor([public_head(case["query"]) for case in cases])
        gold = torch.from_numpy(true_targets(cases))
        choices, losses = [], []
        with torch.no_grad():
            for offset in range(0, len(cases), args.batch_size):
                logits = model(x[offset:offset + args.batch_size], heads[offset:offset + args.batch_size])
                choices.extend(logits.argmax(dim=-1).tolist())
                losses.extend(F.cross_entropy(logits, gold[offset:offset + args.batch_size], reduction="none").tolist())
        if not np.isfinite(losses).all():
            raise ValueError("non-finite fresh probe loss")
        name = kind + "_" + arm + ("_" + intervention if intervention != "matched" else "")
        append_predictions(seed, view, name, cases, choices, losses)

    def receiver_static(case, condition):
        text = case["query"]
        if condition in ("text", "symbolic"):
            field = "source" if condition == "text" else "symbolic"
            text = case["member"][field] + "\n" + text
        if text not in static_cache:
            tokens = receiver_tokenizer.encode(text, add_special_tokens=False)
            if not tokens or len(tokens) + config["slots"] > config["max_length"]:
                raise ValueError("receiver context overflow/empty query")
            with torch.no_grad():
                static_cache[text] = receiver.get_input_embeddings()(torch.tensor(tokens)).detach()
        return static_cache[text]

    def evaluate_receiver(seed, condition, view):
        cases = cases_by_split[view]
        choices, losses = [], []
        with torch.no_grad():
            for offset in range(0, len(cases), 16):
                batch = cases[offset:offset + 16]
                prefix = None
                if condition != "frozen_no_channel":
                    bridge_arm = condition if condition in TRAINED_ARMS else "query_aware"
                    if condition == "aware_zero":
                        messages = torch.zeros((len(batch), config["slots"], config["bottleneck"]))
                    else:
                        arm = {"query_aware": "message_aware", "query_blind": "message_blind", "private_constant": "message_private_constant", "text": "receiver_text", "symbolic": "receiver_symbolic"}[bridge_arm]
                        source_cases = [cases[shuffle_maps[view][index]] for index in range(offset, offset + len(batch))] if condition == "aware_shuffled" else batch
                        values = np.stack([representation(seed, case, arm, condition == "aware_wrong_question") for case in source_cases])
                        messages = torch.from_numpy(values).reshape(len(batch), config["slots"], config["bottleneck"])
                    prefix = bridges[seed][bridge_arm].decode(messages)
                sequences = [receiver_static(case, condition) if prefix is None else torch.cat((prefix[index], receiver_static(case, condition)), dim=0) for index, case in enumerate(batch)]
                embeddings = pad_sequence(sequences, batch_first=True)
                lengths = torch.tensor([len(sequence) for sequence in sequences])
                mask = (torch.arange(embeddings.shape[1])[None, :] < lengths[:, None]).long()
                logits = selected_logits(receiver, embeddings, mask, ids)
                choices.extend(logits.argmax(dim=-1).tolist())
                losses.extend(F.cross_entropy(logits, torch.from_numpy(true_targets(batch)), reduction="none").tolist())
        if not np.isfinite(losses).all():
            raise ValueError("non-finite fresh receiver loss")
        append_predictions(seed, view, "receiver_" + condition, cases, choices, losses)

    fresh_start = time.perf_counter()
    for seed in args.seeds:
        for (chosen_seed, kind, arm), state in selected_states.items():
            if chosen_seed != seed:
                continue
            dimension = statistics_by_arm[(seed, "message_aware" if arm == "message_aware_shuffled_labels" else arm)]["dimension"]
            model = make_probe(dimension, kind).eval()
            model.load_state_dict(state)
            for view in VIEWS:
                evaluate_probe(seed, kind, arm, model, view)
                evaluate_probe(seed, kind, arm, model, view, "wrong_source")
                if arm == "message_aware":
                    evaluate_probe(seed, kind, arm, model, view, "wrong_question")
            print(json.dumps({"phase": "fresh_probe_complete", "seed": seed, "reader": kind, "arm": arm}), flush=True)
        for view in VIEWS:
            for condition in EVALUATED_ARMS:
                evaluate_receiver(seed, condition, view)
        print(json.dumps({"phase": "fresh_receiver_complete", "seed": seed}), flush=True)
    fresh_seconds = time.perf_counter() - fresh_start
    summary = summarize(all_predictions, args.seeds)
    full_gate = {str(seed): next(row["validation"]["record_accuracy"] for row in fit_records if row["seed"] == seed and row["reader"] == "linear" and row["arm"] == "full_source") for seed in args.seeds}
    result = {"execution_status": "executed_frozen_message_readout_diagnostic", "reconstruction_gate": gate,
        "selection": selection, "config": {"seeds": args.seeds, "probe_seed_offset": 4242, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "gradient_clip_norm": 1.0,
            "max_source_positions": args.max_source_positions, "threads": args.threads, "linear_arms": list(LINEAR_ARMS), "mlp_arms_if_triggered": list(MESSAGE_ARMS)},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "model_files": actual_model_files, "data_manifest": json.loads((Path(args.data_dir) / "manifest.json").read_text()),
        "full_source_validation_competence": {"threshold": 0.8, "per_seed_accuracy": full_gate,
            "limited_seeds": [seed for seed in args.seeds if full_gate[str(seed)] < 0.8], "passed": all(value >= 0.8 for value in full_gate.values())},
        "summary": summary, "fits": fit_records,
        "accounting": {"message_float32_bytes": 64, "probe_question_head_parser": "Known public-text templates only; parser advantage disclosed",
            "source_unique_encodings": len(source_cache), "public_question_unique_encodings": len(question_cache),
            "frozen_source_cache_bytes": sum(value.numel() * value.element_size() for value in source_cache.values()),
            "full_source_feature_coordinates": 64 * config["sender_dim"], "pooled_feature_coordinates": config["slots"] * config["sender_dim"],
            "fresh_original_question_utf8_bytes_mean": float(np.mean([len(case["query"].encode()) for case in cases_by_split["iid"]])),
            "full_source_expanded_cache_published": False, "network_energy_speedup_measured": False},
        "timings_seconds": {"model_loading_and_checkpoint_exports": model_loading_seconds, "feature_preparation": preparation_seconds,
            "source_encoding": source_seconds, "question_encoding": question_seconds,
            "feature_preprocessing": sum(row["preprocessing_seconds"] for row in fit_records),
            "probe_training": sum(row["training_seconds"] for row in fit_records), "probe_validation": sum(row["validation_seconds"] for row in fit_records),
            "fresh_evaluation": fresh_seconds, "diagnostic_runner_excluding_reconstruction_and_downloads": time.perf_counter() - started},
        "retention": {"durable_bounded_exports": {path.name: file_metadata(path) for path in sorted(exports.iterdir())},
            "artifact_retention_days": 90, "larger_checkpoint_recipe": "Full selected .pt states and normalization.npz in workflow artifact; regenerate source/pool states from pinned model and exported reconstructed bridges using feature manifest hashes."},
        "limits": ["Behaviorally verified reconstruction does not prove historical final-message byte identity.",
            "Probes get extra supervised fitting and public-question parsing; an accessibility gap is not an equal-budget superior-agent claim.",
            "Larger raw/pool controls have unequal capacity and positional access; they cannot prove irreversible compression loss.",
            "A failed linear or prespecified small MLP probe does not prove relevant information is absent.",
            "Reuse of previous fitting data is explicit; only the 32 test families are entirely fresh after excluding 2,002 prior worlds.",
            "Intervals are descriptive and conditional on three bridge/probe seed pairs; views are not independent samples."]}
    if args.model_manifest:
        result["download_manifest"] = json.loads(Path(args.model_manifest).read_text())
    raw = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_predictions).encode()
    if (output / "predictions.jsonl").read_bytes() != raw:
        raise ValueError("incremental prediction log does not match completed in-memory records")
    (output / "predictions.jsonl.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    result["prediction_integrity"] = {"records": len(all_predictions), "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw),
        "compressed": file_metadata(output / "predictions.jsonl.gz")}
    (output / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2) + "\n")
    (output / "dataset_manifest.json").write_bytes((Path(args.data_dir) / "manifest.json").read_bytes())
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(make_report(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sender-model", "receiver-model", "original-data-dir", "reconstruction-dir", "reference-dir", "data-dir", "output", "source-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model-manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-positions", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Optional transformer dependencies unavailable: {error}\n")


if __name__ == "__main__":
    main()
