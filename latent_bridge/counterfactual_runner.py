"""Closed-choice private-fact experiment with matched adaptation and paired edits.

This module loads existing local model directories only. It never downloads a
model or uses a provider API. Choices are fixed before observing any labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .private_counterfactual import COLORS, OBJECTS, VARIANTS, VIEWS, load_partitions, parse_symbolic
from .transformers_runner import constant_observation, fingerprint_directory, make_bridge

TRAINED_ARMS = ("latent", "constant", "text", "symbolic")
EVALUATED_ARMS = (*TRAINED_ARMS, "frozen_no_channel", "latent_zero", "latent_shuffled")
METRICS = ("paired_both_correct", "record_accuracy", "answer_change_rate",
           "nuisance_both_correct", "nuisance_invariance")


def competence_gate(seeds, accuracies):
    if len(seeds) != len(accuracies) or not seeds or any(not 0 <= value <= 1 for value in accuracies):
        raise ValueError("invalid per-seed competence measurements")
    limited = [seed for seed, accuracy in zip(seeds, accuracies) if accuracy < 0.8]
    return {"threshold": 0.8,
            "metric": "Task-adapted full-text validation original/counterfactual record accuracy, required separately for every seed",
            "per_seed_accuracy": dict(zip(map(str, seeds), accuracies)),
            "per_seed_passed": {str(seed): seed not in limited for seed in seeds},
            "receiver_limited_seeds": limited, "mean_accuracy": float(np.mean(accuracies)), "passed": not limited}


def candidate_ids(tokenizer):
    """All eight options must be distinct single next tokens; no gold argument."""
    encoded = [tokenizer.encode(" " + color, add_special_tokens=False) for color in COLORS]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(COLORS):
        raise ValueError("each fixed leading-space color must encode to one distinct token; this study does not silently change its scoring rule")
    return [ids[0] for ids in encoded]


def selected_logits(receiver, embeddings, attention_mask, ids):
    """Exact selected LM-head logits for supported affine-head decoder models.

    Avoid materializing batch x sequence x vocabulary logits. The final query
    position is selected per row, so right padding cannot change the target.
    """
    import torch
    from torch.nn import functional as F

    if getattr(receiver.config, "model_type", None) not in ("gpt2", "gpt_neox"):
        raise ValueError("optimized scorer is audited only for GPT-2/GPT-NeoX affine LM heads")
    if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("embedding/mask shape mismatch")
    lengths = attention_mask.long().sum(dim=1)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
    expected_mask = positions < lengths[:, None]
    if bool((lengths <= 0).any()) or not torch.equal(attention_mask.bool(), expected_mask):
        raise ValueError("scoring requires a nonempty contiguous right-padded sequence")
    head = receiver.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("unsupported nonlinear output head")
    output = receiver.base_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                 use_cache=False, return_dict=True).last_hidden_state
    final = output[torch.arange(len(output), device=output.device), lengths - 1]
    weight = head.weight.index_select(0, ids)
    bias = head.bias.index_select(0, ids) if head.bias is not None else None
    return F.linear(final, weight, bias)


def flatten(rows, variants=VARIANTS):
    return [{"family": row["family_id"], "query_key": row["query_key"], "view": row["view"],
             "variant": variant, "query": row["query"], "member": row[variant]}
            for row in rows for variant in variants]


def source_derangement(cases):
    """Move every family's members coherently to a different family, label-free."""
    families = sorted({case["family"] for case in cases})
    if len(families) < 2:
        raise ValueError("derangement needs two distinct world families")
    successor = {family: families[(index + 1) % len(families)] for index, family in enumerate(families)}
    lookup = {(case["family"], case["query_key"], case["variant"]): index for index, case in enumerate(cases)}
    result = []
    for case in cases:
        other = (successor[case["family"]], case["query_key"], case["variant"])
        if other not in lookup:
            raise ValueError("derangement requires complete coherent families")
        result.append(lookup[other])
    return result


def oracle_accuracy(partitions):
    correct = count = 0
    for name, rows in partitions.items():
        if not name.startswith("test_"):
            continue
        for row in rows:
            for variant in VARIANTS:
                member = row[variant]
                predicted = parse_symbolic(member["symbolic"])[row["query_key"]]
                correct += predicted == member["answer"]
                count += 1
    return {"correct": correct, "records": count, "accuracy": correct / count,
            "role": "Deterministic task-solvability check with complete structured facts; not an LM or a communication efficiency result."}


def paired_family_scores(predictions):
    """Each family contributes four query pairs, regardless of rendering count."""
    grouped = {}
    for row in predictions:
        key = (row["seed"], row["view"], row["condition"], row["family"], row["query_key"])
        variants = grouped.setdefault(key, {})
        if row["variant"] in variants:
            raise ValueError("duplicate prediction for a paired member")
        variants[row["variant"]] = row
    family = {}
    for key, variants in grouped.items():
        if set(variants) != set(VARIANTS):
            raise ValueError("incomplete paired predictions")
        original, counter, nuisance = (variants[name] for name in VARIANTS)
        if original["gold"] == counter["gold"] or original["gold"] != nuisance["gold"]:
            raise ValueError("counterfactual labels violate paired task definition")
        correct_o = original["choice"] == original["gold"]
        correct_c = counter["choice"] == counter["gold"]
        correct_n = nuisance["choice"] == nuisance["gold"]
        values = {
            "paired_both_correct": float(correct_o and correct_c),
            "record_accuracy": (correct_o + correct_c) / 2,
            "answer_change_rate": float(original["choice"] != counter["choice"]),
            "nuisance_both_correct": float(correct_o and correct_n),
            "nuisance_invariance": float(original["choice"] == nuisance["choice"]),
        }
        family.setdefault(key[:4], []).append(values)
    result = {}
    for key, rows in family.items():
        if len(rows) != len(OBJECTS):
            raise ValueError("every scored family must contain four object queries")
        result[key] = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    return result


def summarize(predictions, seeds, bootstrap_count=2000, bootstrap_seed=66173):
    """World-cluster intervals conditional on the fixed trained seeds.

    The same sampled family indices are used for all arms/views and differences;
    repeated views and seeds never become extra independent world clusters.
    """
    scores = paired_family_scores(predictions)
    families = sorted({key[3] for key in scores})
    conditions = sorted({key[2] for key in scores})
    views = sorted({key[1] for key in scores})
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(families), (bootstrap_count, len(families)))
    arrays, summaries = {}, []
    for view in views:
        for condition in conditions:
            metrics = {}
            for metric in METRICS:
                values = np.array([[scores[(seed, view, condition, family)][metric] for family in families] for seed in seeds])
                per_family = values.mean(axis=0)
                arrays[(view, condition, metric)] = per_family
                bootstrap = per_family[draws].mean(axis=1)
                seed_means = values.mean(axis=1)
                metrics[metric] = {"mean": float(per_family.mean()),
                    "family_cluster_95_percentile_interval": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
                    "seed_means": {str(seed): float(mean) for seed, mean in zip(seeds, seed_means)},
                    "seed_range": [float(seed_means.min()), float(seed_means.max())]}
            summaries.append({"view": view, "condition": condition, "families": len(families), "metrics": metrics})
    comparisons = []
    for view in views:
        for control in ("constant", "text", "symbolic", "latent_shuffled"):
            delta = arrays[(view, "latent", "paired_both_correct")] - arrays[(view, control, "paired_both_correct")]
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            per_seed = {str(seed): float(np.mean([scores[(seed, view, "latent", family)]["paired_both_correct"] - scores[(seed, view, control, family)]["paired_both_correct"] for family in families])) for seed in seeds}
            comparisons.append({"view": view, "metric": "paired_both_correct", "contrast": "latent_minus_" + control.removeprefix("latent_"),
                                "mean_difference": float(delta.mean()),
                                "paired_family_cluster_95_percentile_interval": [float(value) for value in interval],
                                "seed_mean_differences": per_seed,
                                "seed_difference_range": [min(per_seed.values()), max(per_seed.values())]})
    shifts = []
    if "iid" in views:
        for view in views:
            if view == "iid":
                continue
            for condition in conditions:
                delta = arrays[(view, condition, "paired_both_correct")] - arrays[("iid", condition, "paired_both_correct")]
                interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
                shifts.append({"condition": condition, "metric": "paired_both_correct", "contrast": view + "_minus_iid",
                               "mean_difference": float(delta.mean()), "paired_family_cluster_95_percentile_interval": [float(value) for value in interval]})
    return {"conditions": summaries, "paired_comparisons": comparisons,
            "paired_view_differences": shifts,
            "primary_contrast": {"view": "iid", "metric": "paired_both_correct", "contrast": "latent_minus_shuffled"},
            "uncertainty": {"cluster_unit": "Original world family including all four queries, edits, and renderings",
                            "families": len(families), "bootstrap_resamples": bootstrap_count, "bootstrap_seed": bootstrap_seed,
                            "method": "Average each family's score over its four queries and three fixed seeds; resample whole families with identical draw indices for every arm/view/contrast.",
                            "scope": "Descriptive 95% intervals conditional on these fixed seeds. Three seeds do not support a population-of-training-seeds confidence claim. No multiplicity-adjusted significance claim."}}


def make_report(result):
    gate = result["validation_competence_gate"]
    lines = ["# Pretrained private-fact counterfactual study", "",
             f"Execution status: **{result['execution_status']}**. Interpretation: **{result['interpretation_status']}**.", "",
             f"The predeclared adapted-full-text validation gate is 80% in **each seed**. Seed accuracies: {gate['per_seed_accuracy']}. Receiver-limited seeds: {gate['receiver_limited_seeds']}. No seed is excluded from the reported results.", "",
             "These are eight-choice classifications, not open-ended generated-answer scores. They do not establish a universal language, real-world agent performance, or a speed/bandwidth advantage.", "",
             "| View | Condition | Both target answers correct | Target record accuracy | Nuisance both correct | Nuisance prediction stability |",
             "|---|---|---:|---:|---:|---:|"]
    for row in result["summary"]["conditions"]:
        metrics = row["metrics"]
        lines.append(f"| {row['view']} | {row['condition']} | {metrics['paired_both_correct']['mean']:.2%} | {metrics['record_accuracy']['mean']:.2%} | {metrics['nuisance_both_correct']['mean']:.2%} | {metrics['nuisance_invariance']['mean']:.2%} |")
    lines += ["", "## Reading the result", "",
              "The primary outcome requires both answers in a single-fact edit pair to be correct. An arbitrary answer flip is insufficient. Constant/no-channel conditions have a structural zero on this primary measure: the query and model input do not change while the two correct answers differ. Beating that zero alone is weak evidence; inspect absolute accuracy, the matched-versus-shuffled drop, and the task-adapted text/symbolic controls.", "",
              "A seed failing the full-text validation gate is labeled receiver-limited, regardless of the across-seed mean. A negative latent result in that setting is not evidence that useful latent communication is impossible. Passing the gate would still not itself establish superiority or novelty.", "",
              "The JSON reports seed means/ranges and paired family-cluster intervals. Four views reuse the same families; queries, edits, and trained seeds are averaged inside each cluster. Intervals are descriptive and conditional on these three seeds.", "",
              "The parser oracle is only a solvability check with complete facts. Four trained conditions have equal adaptation architecture, initial weights, training order, and update budget, but longer text/symbolic contexts consume more FLOPs. No runtime equality is implied.", "",
              "All negative or positive results must be retained. No held-out outcomes were used to choose this study's fixed settings."]
    return "\n".join(lines) + "\n"


def run(args):
    partitions = load_partitions(args.data_dir)
    oracle = oracle_accuracy(partitions)
    if oracle["accuracy"] != 1:
        raise ValueError("structured parser failed the task-solvability check")
    for path in (args.sender_model, args.receiver_model):
        if not Path(path).is_dir():
            raise ValueError("model paths must be existing local directories")
    import torch
    import transformers
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    options = {"local_files_only": True, "trust_remote_code": False}
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, **options)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, **options)
    ids = torch.tensor(candidate_ids(receiver_tokenizer), device=args.device)
    sender = AutoModel.from_pretrained(args.sender_model, **options).float().to(args.device).eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, **options).float().to(args.device).eval()
    for model in (sender, receiver):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in ("max_position_embeddings", "n_positions"):
            limit = getattr(model.config, name, None)
            if isinstance(limit, int) and args.max_length > limit:
                raise ValueError("configured maximum exceeds a model's context limit")
    loading_seconds = time.perf_counter() - start
    prep_start = time.perf_counter()
    hidden_cache, embedding_cache = {}, {}

    def prepare_case(case):
        member = case["member"]
        source = member["source"]
        if source not in hidden_cache:
            encoded = sender_tokenizer(source, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > args.max_length:
                raise ValueError("sender source exceeds maximum; truncation is forbidden")
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden_cache[source] = sender(**encoded).last_hidden_state.squeeze(0).detach()
        static = {}
        for name, prefix in (("query", ""), ("text", member["source"] + "\n"), ("symbolic", member["symbolic"] + "\n")):
            text = prefix + case["query"]
            if text not in embedding_cache:
                tokens = receiver_tokenizer.encode(text, add_special_tokens=False)
                if not tokens or len(tokens) + args.slots > args.max_length:
                    raise ValueError("receiver source/query exceeds maximum; truncation is forbidden")
                with torch.no_grad():
                    embedding_cache[text] = receiver.get_input_embeddings()(torch.tensor(tokens, device=args.device)).detach()
            static[name] = embedding_cache[text]
        return {**case, "hidden": hidden_cache[source], "static": static,
                "target": COLORS.index(member["answer"])}

    train_cases = [prepare_case(case) for case in flatten(partitions["train"], ("original",))]
    validation_cases = [prepare_case(case) for case in flatten(partitions["validation"], ("original", "counterfactual"))]
    test_cases = {view: [prepare_case(case) for case in flatten(partitions["test_" + view])] for view in VIEWS}
    shuffled_indices = {view: source_derangement(cases) for view, cases in test_cases.items()}
    for view, cases in test_cases.items():
        for index, case in enumerate(cases):
            case["shuffled_hidden"] = cases[shuffled_indices[view][index]]["hidden"]
    preparation_seconds = time.perf_counter() - prep_start
    sender_dim = train_cases[0]["hidden"].shape[-1]
    receiver_dim = receiver.get_input_embeddings().weight.shape[-1]
    all_predictions, runs = [], []

    def batch_logits(cases, condition, model, fixed_hidden, fixed_mask):
        size = len(cases)
        if condition == "frozen_no_channel":
            prefixes = None
        elif condition in ("constant", "text", "symbolic"):
            prefixes = model(fixed_hidden.expand(size, -1, -1), fixed_mask.expand(size, -1))
        elif condition == "latent_zero":
            messages = torch.zeros((size, args.slots, args.bottleneck), device=args.device)
            prefixes = model.decode(messages)
        else:
            states = [case["shuffled_hidden"] if condition == "latent_shuffled" else case["hidden"] for case in cases]
            padded = pad_sequence(states, batch_first=True)
            lengths = torch.tensor([len(value) for value in states], device=args.device)
            mask = torch.arange(padded.shape[1], device=args.device)[None, :] < lengths[:, None]
            prefixes = model(padded, mask)
        static_name = condition if condition in ("text", "symbolic") else "query"
        sequences = [case["static"][static_name] if prefixes is None else torch.cat((prefixes[index], case["static"][static_name]), dim=0)
                     for index, case in enumerate(cases)]
        embeddings = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=args.device)
        attention = (torch.arange(embeddings.shape[1], device=args.device)[None, :] < lengths[:, None]).long()
        return selected_logits(receiver, embeddings, attention, ids)

    def evaluate(cases, condition, model, fixed_hidden, fixed_mask, seed, view, collect=False):
        evaluation_start = time.perf_counter()
        losses, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for offset in range(0, len(cases), args.batch_size):
                batch = cases[offset:offset + args.batch_size]
                logits = batch_logits(batch, condition, model, fixed_hidden, fixed_mask)
                targets = torch.tensor([case["target"] for case in batch], device=args.device)
                per_case = F.cross_entropy(logits, targets, reduction="none")
                if not bool(torch.isfinite(per_case).all()):
                    raise ValueError("non-finite candidate loss")
                choices = logits.argmax(dim=-1)
                losses += float(per_case.sum())
                correct += int((choices == targets).sum())
                total += len(batch)
                if collect:
                    for index, case in enumerate(batch):
                        all_predictions.append({"seed": seed, "view": view, "condition": condition,
                            "family": case["family"], "query_key": case["query_key"], "variant": case["variant"],
                            "choice": int(choices[index]), "gold": case["target"], "restricted_ce": float(per_case[index])})
        return {"restricted_ce": losses / total, "record_accuracy": correct / total, "records": total,
                "evaluation_seconds": time.perf_counter() - evaluation_start}

    for seed in args.seeds:
        torch.manual_seed(seed)
        initial = make_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
        models = {}
        for arm in TRAINED_ARMS:
            model = make_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
            model.load_state_dict(copy.deepcopy(initial.state_dict()))
            models[arm] = model
        initial_hashes = {arm: hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for _, value in sorted(model.state_dict().items()))).hexdigest() for arm, model in models.items()}
        if len(set(initial_hashes.values())) != 1:
            raise ValueError("trained conditions must start from identical bridge parameters")
        fixed_hidden, fixed_mask = constant_observation(sender_dim, seed, args.device)
        optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=args.learning_rate) for arm, model in models.items()}
        best = {arm: float("inf") for arm in TRAINED_ARMS}
        checkpoints, selected_epochs, history = {}, {}, []
        arm_training_seconds = {arm: 0.0 for arm in TRAINED_ARMS}
        rng = random.Random(seed)
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(train_cases)))
            rng.shuffle(order)
            for arm in TRAINED_ARMS:
                arm_start = time.perf_counter()
                model, optimizer = models[arm], optimizers[arm]
                for offset in range(0, len(order), args.batch_size):
                    cases = [train_cases[index] for index in order[offset:offset + args.batch_size]]
                    optimizer.zero_grad()
                    logits = batch_logits(cases, arm, model, fixed_hidden, fixed_mask)
                    targets = torch.tensor([case["target"] for case in cases], device=args.device)
                    loss = F.cross_entropy(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                arm_training_seconds[arm] += time.perf_counter() - arm_start
                validation = evaluate(validation_cases, arm, model, fixed_hidden, fixed_mask, seed, "validation")
                record = {"seed": seed, "epoch": epoch, "arm": arm, "validation": validation}
                history.append(record)
                print(json.dumps(record), flush=True)
                if validation["restricted_ce"] < best[arm]:
                    best[arm] = validation["restricted_ce"]
                    selected_epochs[arm] = epoch
                    checkpoints[arm] = copy.deepcopy(model.state_dict())
        for arm, model in models.items():
            model.load_state_dict(checkpoints[arm])
        selected_validation = {arm: evaluate(validation_cases, arm, model, fixed_hidden, fixed_mask, seed, "validation") for arm, model in models.items()}
        metrics = {}
        for view in VIEWS:
            metrics[view] = {}
            for condition in EVALUATED_ARMS:
                model = models[condition] if condition in models else models["latent"]
                metric = evaluate(test_cases[view], condition, model, fixed_hidden, fixed_mask, seed, view, collect=True)
                metrics[view][condition] = metric
                print(json.dumps({"seed": seed, "view": view, "condition": condition, **metric}), flush=True)
        checkpoint_dir = Path(args.output) / "checkpoints" / str(seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for arm, model in models.items():
            torch.save(model.state_dict(), checkpoint_dir / (arm + ".pt"))
        runs.append({"seed": seed, "selected_epochs": selected_epochs, "selected_validation": selected_validation,
                     "history": history, "test_record_metrics": metrics, "arm_training_seconds": arm_training_seconds,
                     "optimizer_steps_per_arm": args.epochs * ((len(train_cases) + args.batch_size - 1) // args.batch_size),
                     "parameters_per_arm": sum(parameter.numel() for parameter in initial.parameters()),
                     "initial_bridge_state_sha256": initial_hashes,
                     "constant_input_sha256_float32": hashlib.sha256(fixed_hidden.cpu().numpy().tobytes()).hexdigest()})
    summary = summarize(all_predictions, args.seeds)
    full_text_validation = [run["selected_validation"]["text"]["record_accuracy"] for run in runs]
    gate = competence_gate(args.seeds, full_text_validation)
    limited_seeds = gate["receiver_limited_seeds"]
    for record in runs:
        record["validation_competence_gate_passed"] = record["seed"] not in limited_seeds
        record["communication_interpretation"] = "receiver_limited" if record["seed"] in limited_seeds else "competence_gate_passed_comparative_evidence_still_required"
    input_sizes = {}
    for name, cases in {"train": train_cases, "validation": validation_cases, **test_cases}.items():
        input_sizes[name] = {}
        sender_lengths = [len(case["hidden"]) for case in cases]
        input_sizes[name]["sender_observation"] = {"records": len(cases),
            "positions_min": min(sender_lengths), "positions_mean": float(np.mean(sender_lengths)),
            "positions_max": max(sender_lengths), "positions_sum_logical_records": sum(sender_lengths),
            "actual_sender_execution": "Cached once per unique source text across all splits/views/conditions"}
        for condition in EVALUATED_ARMS:
            static_name = condition if condition in ("text", "symbolic") else "query"
            lengths = [len(case["static"][static_name]) + (0 if condition == "frozen_no_channel" else args.slots) for case in cases]
            input_sizes[name][condition] = {"records": len(cases), "receiver_positions_min": min(lengths),
                                            "receiver_positions_mean": float(np.mean(lengths)),
                                            "receiver_positions_max": max(lengths), "receiver_positions_sum": sum(lengths)}
    result = {
        "execution_status": "executed_pretrained_counterfactual_study",
        "interpretation_status": "receiver_limited_in_one_or_more_seeds" if limited_seeds else "all_seed_competence_gates_passed_comparative_evidence_still_required",
        "config": {"seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size, "slots": args.slots,
                   "bottleneck": args.bottleneck, "learning_rate": args.learning_rate, "max_length": args.max_length,
                   "threads": args.threads, "device": args.device, "sender_dim": sender_dim, "receiver_dim": receiver_dim,
                   "training_arms": list(TRAINED_ARMS), "gradient_clip_norm": 1.0, "optimizer": "AdamW", "weight_decay": 0.01},
        "candidate_colors": list(COLORS), "candidate_token_ids": ids.cpu().tolist(),
        "candidate_scoring": "Restricted softmax over eight fixed single-token leading-space color logits at the final query position. No gold tokens/length enter model inputs.",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "model_files": {"sender": fingerprint_directory(args.sender_model), "receiver": fingerprint_directory(args.receiver_model)},
        "data_manifest": json.loads((Path(args.data_dir) / "manifest.json").read_text()),
        "data_file_sha256": {name: hashlib.sha256((Path(args.data_dir) / (name + ".jsonl")).read_bytes()).hexdigest() for name in partitions},
        "validation_competence_gate": gate,
        "structured_parser_oracle": oracle, "summary": summary, "runs": runs,
        "accounting": {"latent_payload_float32_bytes": args.slots * args.bottleneck * 4,
                       "bridge_parameter_float32_bytes_per_arm": runs[0]["parameters_per_arm"] * 4,
                       "sender_parameter_bytes": sum(p.numel() * p.element_size() for p in sender.parameters()),
                       "receiver_parameter_bytes": sum(p.numel() * p.element_size() for p in receiver.parameters()),
                       "unique_source_states": len(hidden_cache), "cached_sender_state_bytes": sum(value.numel() * value.element_size() for value in hidden_cache.values()),
                       "cached_receiver_embedding_bytes": sum(value.numel() * value.element_size() for value in embedding_cache.values()),
                       "iid_text_payload_utf8_bytes_mean": float(np.mean([len(case["member"]["source"].encode()) for case in test_cases["iid"]])),
                       "iid_symbolic_payload_utf8_bytes_mean": float(np.mean([len(case["member"]["symbolic"].encode()) for case in test_cases["iid"]])),
                       "input_sizes": input_sizes,
                       "wire_framing_network_energy_and_optimizer_activations_measured": False},
        "timings_seconds": {"model_loading": loading_seconds, "case_preparation": preparation_seconds,
                            "entire_runner_excluding_downloads": time.perf_counter() - start},
        "limits": ["This is a small synthetic lookup task, not real-world agent work, universal communication, or a new method claim.",
                   "Constant, zero, and frozen-no-channel inputs do not change within a targeted pair, so paired both-correct is structurally zero; beating this alone is weak evidence.",
                   "Different-family source shuffling is coherent across members and uses no labels; coincidentally matching colors remain possible.",
                   "Adaptation architecture/initialization/order/update budgets are equal; input lengths and therefore FLOPs are not equal.",
                   "Template views reuse the same world families; uncertainty resamples families jointly and is conditional on only three fixed training seeds.",
                   "All settings are fixed before held-out scoring. The validation gate describes interpretability, not a hyperparameter tuning loop.",
                   "Closed-choice classification eliminates generation-format errors but does not measure open-ended generation or inference speed.",
                   "The structured-parser oracle proves solvability with complete facts only; it is not evidence for an LM or a latent bandwidth advantage."],
    }
    if args.model_manifest:
        result["download_manifest"] = json.loads(Path(args.model_manifest).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in all_predictions))
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(make_report(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sender-model", "receiver-model", "data-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-manifest")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.slots, args.bottleneck, args.learning_rate, args.max_length, args.threads) <= 0:
        parser.error("sizes and learning rate must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Optional transformer dependencies unavailable: {error}\n")


if __name__ == "__main__":
    main()
