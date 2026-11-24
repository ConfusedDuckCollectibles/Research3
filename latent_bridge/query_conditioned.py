"""Fixed public-query-conditioning follow-up using existing local model weights.

The sender sees private text and the public question through separate frozen
encoder calls. Only a learned two-slot message reaches the frozen receiver.
No provider API, download, fact parser, or answer-derived feature is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .counterfactual_runner import (
    METRICS, candidate_ids, competence_gate, flatten, oracle_accuracy,
    paired_family_scores, selected_logits, source_derangement,
)
from .private_counterfactual import COLORS, OBJECTS, VIEWS, query_text
from .query_conditioned_data import load_partitions
from .transformers_runner import constant_observation, fingerprint_directory

TRAINED_ARMS = ("query_aware", "query_blind", "private_constant", "text", "symbolic")
EVALUATED_ARMS = (*TRAINED_ARMS, "frozen_no_channel", "aware_zero", "aware_shuffled", "aware_wrong_question")
QUERY_NORMALIZATION_EPSILON = 1e-5
FIXED_QUERY_SEED_OFFSET = 17291


def normalize_question(vector):
    """Nonparametric LayerNorm: population variance, epsilon inside the root."""
    import torch
    from torch.nn import functional as F

    if vector.ndim < 1 or vector.shape[-1] < 2 or not bool(torch.isfinite(vector).all()):
        raise ValueError("invalid public-question vector")
    return F.layer_norm(vector, (vector.shape[-1],), eps=QUERY_NORMALIZATION_EPSILON)


def constant_question(sender_dim, seed, device="cpu"):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed + FIXED_QUERY_SEED_OFFSET)
    return normalize_question(torch.randn(sender_dim, generator=generator, device=device))


def wrong_public_question(query):
    """Replace only the public question string; never inspect facts or labels."""
    mapping = {query_text(index, shifted): query_text((index + 1) % len(OBJECTS), shifted)
               for shifted in (False, True) for index in range(len(OBJECTS))}
    if query not in mapping:
        raise ValueError("question does not match a preregistered public template")
    return mapping[query]


def make_query_bridge(sender_dim, receiver_dim, bottleneck=8, slots=2):
    import torch
    from torch import nn

    if min(sender_dim, receiver_dim, bottleneck, slots) <= 0:
        raise ValueError("all bridge dimensions must be positive")

    class QueryConditionedBridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.sender_dim, self.bottleneck, self.slots = sender_dim, bottleneck, slots
            self.queries = nn.Parameter(torch.randn(slots, sender_dim) / math.sqrt(sender_dim))
            self.gates = nn.Parameter(torch.zeros(slots))
            self.compressor = nn.Sequential(nn.LayerNorm(sender_dim), nn.Linear(sender_dim, bottleneck), nn.Tanh())
            self.expander = nn.Linear(bottleneck, receiver_dim)

        def encode(self, hidden, mask, question):
            if hidden.ndim != 3 or hidden.shape[-1] != sender_dim or mask.shape != hidden.shape[:2] or not bool(mask.bool().any(dim=1).all()):
                raise ValueError("invalid private hidden sequence or all-masked observation")
            if question.shape != (hidden.shape[0], sender_dim):
                raise ValueError("public question vector has wrong shape")
            queries = self.queries[None, :, :] + self.gates[None, :, None] * question[:, None, :]
            logits = torch.einsum("bkd,bsd->bks", queries, hidden) / math.sqrt(sender_dim)
            logits = logits.masked_fill(~mask[:, None, :].bool(), torch.finfo(logits.dtype).min)
            pooled = torch.einsum("bks,bsd->bkd", logits.softmax(dim=-1), hidden)
            return self.compressor(pooled)

        def decode(self, message):
            if message.ndim != 3 or message.shape[1:] != (slots, bottleneck):
                raise ValueError("latent message has wrong shape")
            return self.expander(message)

        def forward(self, hidden, mask, question):
            return self.decode(self.encode(hidden, mask, question))

    return QueryConditionedBridge()


def condition_prefixes(cases, condition, model, fixed_hidden, fixed_mask, fixed_question):
    """Route only frozen state tensors, with the same rule in training/test.

    No target, world identity, facts dictionary, or query-key metadata is read.
    The private-constant control keeps the actual public-question path.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if condition not in EVALUATED_ARMS:
        raise ValueError("unknown query-conditioned arm")
    size = len(cases)
    if not size:
        raise ValueError("empty model batch")
    if condition == "frozen_no_channel":
        return None
    if condition == "aware_zero":
        return model.decode(fixed_hidden.new_zeros(size, model.slots, model.bottleneck))
    if condition in ("private_constant", "text", "symbolic"):
        hidden, mask = fixed_hidden.expand(size, -1, -1), fixed_mask.expand(size, -1)
    else:
        key = "shuffled_hidden" if condition == "aware_shuffled" else "hidden"
        states = [case[key] for case in cases]
        hidden = pad_sequence(states, batch_first=True)
        lengths = torch.tensor([len(state) for state in states], device=hidden.device)
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
    if condition == "query_blind":
        question = fixed_question[None, :].expand(size, -1)
    else:
        key = "wrong_question_vector" if condition == "aware_wrong_question" else "question_vector"
        question = torch.stack([case[key] for case in cases])
    return model(hidden, mask, question)


def summarize(predictions, seeds, bootstrap_count=2000, bootstrap_seed=66173):
    scores = paired_family_scores(predictions)
    families = sorted({key[3] for key in scores})
    conditions = sorted({key[2] for key in scores})
    views = sorted({key[1] for key in scores})
    if len(families) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("paired summary needs distinct seeds and at least two families")
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(families), (bootstrap_count, len(families)))
    arrays, summaries = {}, []
    for view in views:
        for condition in conditions:
            metrics = {}
            for metric in METRICS:
                values = np.array([[scores[(seed, view, condition, family)][metric] for family in families] for seed in seeds])
                per_family = values.mean(axis=0)
                arrays[(view, condition, metric)] = per_family
                interval = np.quantile(per_family[draws].mean(axis=1), [0.025, 0.975])
                seed_means = values.mean(axis=1)
                metrics[metric] = {"mean": float(per_family.mean()),
                    "family_cluster_95_percentile_interval": [float(value) for value in interval],
                    "seed_means": dict(zip(map(str, seeds), map(float, seed_means))),
                    "seed_range": [float(seed_means.min()), float(seed_means.max())]}
            summaries.append({"view": view, "condition": condition, "families": len(families), "metrics": metrics})
    comparisons = []
    controls = ("query_blind", "private_constant", "text", "symbolic", "aware_shuffled", "aware_wrong_question")
    for view in views:
        for control in controls:
            delta = arrays[(view, "query_aware", "paired_both_correct")] - arrays[(view, control, "paired_both_correct")]
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            per_seed = {str(seed): float(np.mean([scores[(seed, view, "query_aware", family)]["paired_both_correct"] - scores[(seed, view, control, family)]["paired_both_correct"] for family in families])) for seed in seeds}
            comparisons.append({"view": view, "metric": "paired_both_correct", "contrast": "query_aware_minus_" + control,
                                "mean_difference": float(delta.mean()),
                                "paired_family_cluster_95_percentile_interval": [float(value) for value in interval],
                                "seed_mean_differences": per_seed,
                                "seed_difference_range": [min(per_seed.values()), max(per_seed.values())]})
    shifts = []
    for view in views:
        if view == "iid":
            continue
        for condition in conditions:
            delta = arrays[(view, condition, "paired_both_correct")] - arrays[("iid", condition, "paired_both_correct")]
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            shifts.append({"condition": condition, "metric": "paired_both_correct", "contrast": view + "_minus_iid",
                           "mean_difference": float(delta.mean()),
                           "paired_family_cluster_95_percentile_interval": [float(value) for value in interval]})
    return {"conditions": summaries, "paired_comparisons": comparisons, "paired_view_differences": shifts,
            "primary_contrast": {"view": "iid", "metric": "paired_both_correct", "contrast": "query_aware_minus_query_blind"},
            "uncertainty": {"cluster_unit": "Original world family including four queries, edits and paired views",
                "families": len(families), "bootstrap_resamples": bootstrap_count, "bootstrap_seed": bootstrap_seed,
                "method": "Average each family's four query scores across the fixed seeds; resample whole families with identical indices for all arms, views and contrasts.",
                "scope": "Descriptive 95% intervals conditional on these fixed seeds, not population-of-training-seed confidence. Other comparisons are secondary; no multiplicity-adjusted significance claim."}}


def advancement_targets(summary, gate):
    iid = next(row for row in summary["conditions"] if row["view"] == "iid" and row["condition"] == "query_aware")
    pair_accuracy = iid["metrics"]["paired_both_correct"]["mean"]
    stability = iid["metrics"]["nuisance_invariance"]["mean"]
    return {"target_pair_accuracy_threshold": 0.8, "nuisance_stability_threshold": 0.9,
            "observed_target_pair_accuracy": pair_accuracy, "observed_nuisance_stability": stability,
            "each_seed_text_competence_passed": gate["passed"],
            "passed": bool(gate["passed"] and pair_accuracy >= 0.8 and stability >= 0.9),
            "meaning": "Predeclared engineering targets before considering efficiency studies; passing is not a novelty, universal-communication, or bandwidth claim."}


def make_report(result):
    gate, targets = result["validation_competence_gate"], result["advancement_targets"]
    primary = next(row for row in result["summary"]["paired_comparisons"]
                   if row["view"] == "iid" and row["contrast"] == "query_aware_minus_query_blind")
    lines = ["# Public-query-conditioned private-fact study", "",
        f"Execution: **{result['execution_status']}**. Interpretation: **{result['interpretation_status']}**.", "",
        f"The primary original-view query-aware minus query-blind paired-accuracy difference is {primary['mean_difference']:.2%}, with descriptive family-bootstrap interval {primary['paired_family_cluster_95_percentile_interval']}.", "",
        f"Adapted text must reach 80% validation accuracy in each seed: {gate['per_seed_accuracy']}. Receiver-limited seeds: {gate['receiver_limited_seeds']}. Every seed remains reported.", "",
        f"Predeclared advancement targets: paired accuracy at least 80%, nuisance stability at least 90%, and all text gates passing. Observed: {targets['observed_target_pair_accuracy']:.2%} paired accuracy and {targets['observed_nuisance_stability']:.2%} stability. Targets passed: **{targets['passed']}**.", "",
        "| View | Condition | Both target answers correct | Target record accuracy | Nuisance both correct | Nuisance stability |",
        "|---|---|---:|---:|---:|---:|"]
    for row in result["summary"]["conditions"]:
        metrics = row["metrics"]
        lines.append(f"| {row['view']} | {row['condition']} | {metrics['paired_both_correct']['mean']:.2%} | {metrics['record_accuracy']['mean']:.2%} | {metrics['nuisance_both_correct']['mean']:.2%} | {metrics['nuisance_invariance']['mean']:.2%} |")
    lines += ["", "## Interpretation and costs", "",
        "The primary comparison isolates sender access to the public question within the same trainable architecture, initialization and training budget. Inspect absolute accuracy, wrong-private-source and wrong-public-question interventions, nuisance invariance, and adapted full text before interpreting an improvement as useful fact transfer.", "",
        "The private-constant control receives the real public question. Its input stays identical within a target pair while the correct answer changes, so both-correct is structurally zero. Beating that zero alone is weak evidence. Arbitrary prediction flips also do not satisfy paired correctness.", "",
        "These are fixed eight-choice classifications, not open-ended generation. All four views reuse the same families. Descriptive intervals average queries and the three fixed seeds inside each family; they do not treat repeated views or seeds as independent worlds.", "",
        "The public-question encoding is computed from question tokens alone. Actual and fixed query vectors use nonparametric LayerNorm-style normalization with epsilon 1e-5. Scalar query gates start at zero; all trained arms start with identical bridge weights. Source and question caches are charged explicitly.", "",
        f"The latent payload is {result['accounting']['latent_payload_float32_bytes']} bytes. Original-view public questions average {result['accounting']['iid_public_question_utf8_bytes_mean']:.2f} UTF-8 bytes, charged separately on every request/reply. Question-conditioned messages cannot be reused for another question. Actual cached query encoder costs and logical uncached token work are reported separately; no hypothetical uncached wall time is presented as measured.", "",
        "The study excludes all 1,042 unique worlds from the two earlier pretrained experiments before assigning its new families. Earlier outputs remain unchanged. This is a targeted follow-up on a small synthetic task with existing receiver-conditioning prior art, not a new-method or universal-language claim. Equal adaptation budgets do not imply equal FLOPs. A two-byte packed representation can convey these four categorical facts, so accuracy alone cannot establish an efficiency advantage.", "",
        "Results below the practical targets are retained without relaxing thresholds or changing the architecture after seeing test outcomes."]
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
    preparation_start = time.perf_counter()
    hidden_cache, question_cache, embedding_cache = {}, {}, {}
    source_encoding_records, question_encoding_records = [], []

    def encode_sender_text(text):
        encoded = sender_tokenizer(text, return_tensors="pt", truncation=False)
        length = encoded["input_ids"].shape[1]
        if not 0 < length <= args.max_length:
            raise ValueError("sender source/question exceeds maximum or is empty; truncation is forbidden")
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.no_grad():
            return sender(**encoded).last_hidden_state.squeeze(0).detach(), length

    def question_vector(question):
        # Deliberately a separate call with only public-question text.
        if question not in question_cache:
            encoding_start = time.perf_counter()
            states, tokens = encode_sender_text(question)
            vector = normalize_question(states.mean(dim=0)).detach()
            question_cache[question] = vector
            question_encoding_records.append({
                "question": question, "utf8_bytes": len(question.encode("utf-8")), "sender_tokens": tokens,
                "encoding_and_normalization_seconds": time.perf_counter() - encoding_start,
                "vector_float32_bytes": vector.numel() * vector.element_size(),
                "vector_l2_norm": float(vector.norm()), "vector_mean": float(vector.mean()),
                "vector_population_variance": float(vector.var(unbiased=False)),
                "vector_sha256_float32": hashlib.sha256(vector.cpu().numpy().tobytes()).hexdigest(),
            })
        return question_cache[question]

    def prepare_case(case):
        member, source = case["member"], case["member"]["source"]
        if source not in hidden_cache:
            encoding_start = time.perf_counter()
            hidden_cache[source], tokens = encode_sender_text(source)
            source_encoding_records.append({"tokens": tokens, "seconds": time.perf_counter() - encoding_start})
        actual_question = question_vector(case["query"])
        wrong_question = question_vector(wrong_public_question(case["query"]))
        static = {}
        for name, prefix in (("query", ""), ("text", source + "\n"), ("symbolic", member["symbolic"] + "\n")):
            text = prefix + case["query"]
            if text not in embedding_cache:
                tokens = receiver_tokenizer.encode(text, add_special_tokens=False)
                if not tokens or len(tokens) + args.slots > args.max_length:
                    raise ValueError("receiver source/query exceeds maximum; truncation is forbidden")
                with torch.no_grad():
                    embedding_cache[text] = receiver.get_input_embeddings()(torch.tensor(tokens, device=args.device)).detach()
            static[name] = embedding_cache[text]
        # The target is used only by the loss/scorer, after every model input
        # above has been constructed exclusively from strings.
        return {**case, "hidden": hidden_cache[source], "question_vector": actual_question,
                "wrong_question_vector": wrong_question, "static": static,
                "target": COLORS.index(member["answer"])}

    train_cases = [prepare_case(case) for case in flatten(partitions["train"], ("original",))]
    validation_cases = [prepare_case(case) for case in flatten(partitions["validation"], ("original", "counterfactual"))]
    test_cases = {view: [prepare_case(case) for case in flatten(partitions["test_" + view])] for view in VIEWS}
    shuffled_indices = {view: source_derangement(cases) for view, cases in test_cases.items()}
    for view, cases in test_cases.items():
        for index, case in enumerate(cases):
            case["shuffled_hidden"] = cases[shuffled_indices[view][index]]["hidden"]
    preparation_seconds = time.perf_counter() - preparation_start
    sender_dim = train_cases[0]["hidden"].shape[-1]
    receiver_dim = receiver.get_input_embeddings().weight.shape[-1]
    all_predictions, runs = [], []

    def batch_logits(cases, condition, model, fixed_hidden, fixed_mask, fixed_question):
        prefixes = condition_prefixes(cases, condition, model, fixed_hidden, fixed_mask, fixed_question)
        static_name = condition if condition in ("text", "symbolic") else "query"
        sequences = [case["static"][static_name] if prefixes is None else torch.cat((prefixes[index], case["static"][static_name]), dim=0)
                     for index, case in enumerate(cases)]
        embeddings = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=args.device)
        attention = (torch.arange(embeddings.shape[1], device=args.device)[None, :] < lengths[:, None]).long()
        return selected_logits(receiver, embeddings, attention, ids)

    def evaluate(cases, condition, model, fixed_hidden, fixed_mask, fixed_question, seed, view, collect=False):
        evaluation_start = time.perf_counter()
        losses, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for offset in range(0, len(cases), args.batch_size):
                batch = cases[offset:offset + args.batch_size]
                logits = batch_logits(batch, condition, model, fixed_hidden, fixed_mask, fixed_question)
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
        initial = make_query_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
        models = {}
        for arm in TRAINED_ARMS:
            model = make_query_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
            model.load_state_dict(copy.deepcopy(initial.state_dict()))
            models[arm] = model
        initial_hashes = {arm: hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for _, value in sorted(model.state_dict().items()))).hexdigest() for arm, model in models.items()}
        if len(set(initial_hashes.values())) != 1:
            raise ValueError("trained conditions must start from identical bridge parameters")
        fixed_hidden, fixed_mask = constant_observation(sender_dim, seed, args.device)
        fixed_question = constant_question(sender_dim, seed, args.device)
        optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) for arm, model in models.items()}
        best = {arm: float("inf") for arm in TRAINED_ARMS}
        checkpoints, selected_epochs, history = {}, {}, []
        arm_training_seconds = {arm: 0.0 for arm in TRAINED_ARMS}
        order_digests = []
        rng = random.Random(seed)
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(train_cases)))
            rng.shuffle(order)
            order_digests.append(hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest())
            for arm in TRAINED_ARMS:
                arm_start = time.perf_counter()
                model, optimizer = models[arm], optimizers[arm]
                for offset in range(0, len(order), args.batch_size):
                    cases = [train_cases[index] for index in order[offset:offset + args.batch_size]]
                    optimizer.zero_grad()
                    logits = batch_logits(cases, arm, model, fixed_hidden, fixed_mask, fixed_question)
                    targets = torch.tensor([case["target"] for case in cases], device=args.device)
                    loss = F.cross_entropy(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                arm_training_seconds[arm] += time.perf_counter() - arm_start
                validation = evaluate(validation_cases, arm, model, fixed_hidden, fixed_mask, fixed_question, seed, "validation")
                record = {"seed": seed, "epoch": epoch, "arm": arm, "validation": validation}
                history.append(record)
                print(json.dumps(record), flush=True)
                if validation["restricted_ce"] < best[arm]:
                    best[arm] = validation["restricted_ce"]
                    selected_epochs[arm] = epoch
                    checkpoints[arm] = copy.deepcopy(model.state_dict())
        for arm, model in models.items():
            model.load_state_dict(checkpoints[arm])
        selected_validation = {arm: evaluate(validation_cases, arm, model, fixed_hidden, fixed_mask, fixed_question, seed, "validation") for arm, model in models.items()}
        metrics = {}
        for view in VIEWS:
            metrics[view] = {}
            for condition in EVALUATED_ARMS:
                model = models[condition] if condition in models else models["query_aware"]
                metric = evaluate(test_cases[view], condition, model, fixed_hidden, fixed_mask, fixed_question, seed, view, collect=True)
                metrics[view][condition] = metric
                print(json.dumps({"seed": seed, "view": view, "condition": condition, **metric}), flush=True)
        checkpoint_dir = Path(args.output) / "checkpoints" / str(seed)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for arm, model in models.items():
            torch.save(model.state_dict(), checkpoint_dir / (arm + ".pt"))
        runs.append({"seed": seed, "selected_epochs": selected_epochs, "selected_validation": selected_validation,
            "history": history, "test_record_metrics": metrics, "arm_training_seconds": arm_training_seconds,
            "validation_seconds": sum(row["validation"]["evaluation_seconds"] for row in history) + sum(row["evaluation_seconds"] for row in selected_validation.values()),
            "heldout_evaluation_seconds": sum(row["evaluation_seconds"] for view in metrics.values() for row in view.values()),
            "optimizer_steps_per_arm": args.epochs * math.ceil(len(train_cases) / args.batch_size),
            "parameters_per_arm": sum(parameter.numel() for parameter in initial.parameters()),
            "initial_bridge_state_sha256": initial_hashes, "shared_epoch_training_order_sha256": order_digests,
            "selected_query_gates": {arm: model.gates.detach().cpu().tolist() for arm, model in models.items()},
            "constant_private_input_seed": seed + 9841,
            "constant_private_input_sha256_float32": hashlib.sha256(fixed_hidden.cpu().numpy().tobytes()).hexdigest(),
            "constant_question_seed": seed + FIXED_QUERY_SEED_OFFSET,
            "constant_question_sha256_float32": hashlib.sha256(fixed_question.cpu().numpy().tobytes()).hexdigest(),
            "constant_question_l2_norm": float(fixed_question.norm()),
            "constant_question_mean": float(fixed_question.mean()),
            "constant_question_population_variance": float(fixed_question.var(unbiased=False))})

    summary = summarize(all_predictions, args.seeds)
    gate = competence_gate(args.seeds, [run["selected_validation"]["text"]["record_accuracy"] for run in runs])
    targets = advancement_targets(summary, gate)
    for record in runs:
        record["validation_competence_gate_passed"] = record["seed"] not in gate["receiver_limited_seeds"]
    input_sizes = {}
    question_token_counts = {row["question"]: row["sender_tokens"] for row in question_encoding_records}
    for name, cases in {"train": train_cases, "validation": validation_cases, **test_cases}.items():
        input_sizes[name] = {}
        for label, lengths in (
            ("sender_private_observation", [len(case["hidden"]) for case in cases]),
            ("sender_public_question", [question_token_counts[case["query"]] for case in cases]),
            ("sender_wrong_public_question", [question_token_counts[wrong_public_question(case["query"])] for case in cases]),
        ):
            input_sizes[name][label] = {"records": len(cases), "positions_min": min(lengths),
                "positions_mean": float(np.mean(lengths)), "positions_max": max(lengths),
                "positions_sum_logical_records_without_cache": sum(lengths)}
        for condition in EVALUATED_ARMS:
            static_name = condition if condition in ("text", "symbolic") else "query"
            lengths = [len(case["static"][static_name]) + (0 if condition == "frozen_no_channel" else args.slots) for case in cases]
            input_sizes[name][condition] = {"records": len(cases), "receiver_positions_min": min(lengths),
                "receiver_positions_mean": float(np.mean(lengths)), "receiver_positions_max": max(lengths),
                "receiver_positions_sum": sum(lengths)}
        for field, extractor in (
            ("public_question_utf8_bytes", lambda case: case["query"]),
            ("private_text_utf8_bytes", lambda case: case["member"]["source"]),
            ("private_symbolic_utf8_bytes", lambda case: case["member"]["symbolic"]),
        ):
            lengths = [len(extractor(case).encode("utf-8")) for case in cases]
            input_sizes[name][field] = {"records": len(cases), "min": min(lengths), "mean": float(np.mean(lengths)), "max": max(lengths), "sum": sum(lengths)}
    query_feedback = input_sizes["iid"]["public_question_utf8_bytes"]["mean"]
    result = {
        "execution_status": "executed_pretrained_query_conditioned_study",
        "interpretation_status": ("receiver_limited_in_one_or_more_seeds" if not gate["passed"] else
                                  "engineering_advancement_targets_met_comparative_evidence_still_required" if targets["passed"] else
                                  "competence_gates_passed_engineering_advancement_targets_not_met"),
        "config": {"seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size,
            "slots": args.slots, "bottleneck": args.bottleneck, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "max_length": args.max_length, "threads": args.threads,
            "device": args.device, "sender_dim": sender_dim, "receiver_dim": receiver_dim,
            "training_arms": list(TRAINED_ARMS), "optimizer": "AdamW", "gradient_clip_norm": 1.0,
            "query_normalization": "(mean-token-state - feature mean) / sqrt(population feature variance + epsilon), no learned affine parameters",
            "query_normalization_epsilon": QUERY_NORMALIZATION_EPSILON,
            "attention_query_rule": "base_query[k] + scalar_gate[k] * normalized_public_query_vector", "initial_query_gates": [0.0] * args.slots},
        "candidate_colors": list(COLORS), "candidate_token_ids": ids.cpu().tolist(),
        "candidate_scoring": "Restricted eight-color softmax at the final actual query position; fixed single-token candidates, no gold token or gold length in inputs.",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "model_files": {"sender": fingerprint_directory(args.sender_model), "receiver": fingerprint_directory(args.receiver_model)},
        "data_manifest": json.loads((Path(args.data_dir) / "manifest.json").read_text()),
        "data_file_sha256": {name: hashlib.sha256((Path(args.data_dir) / (name + ".jsonl")).read_bytes()).hexdigest() for name in partitions},
        "validation_competence_gate": gate, "advancement_targets": targets,
        "structured_parser_oracle": oracle, "summary": summary, "runs": runs,
        "public_question_encodings": question_encoding_records,
        "interventions": {"private_source": "Sorted-family successor derangement; coherent query and variant identity; identical across views",
            "public_question": "Cyclic next object question within the same template, replacing only the sender-side question vector",
            "zero": "Zero compressed slots before the trained receiver-side expansion"},
        "accounting": {
            "latent_payload_float32_bytes": args.slots * args.bottleneck * 4,
            "iid_public_question_utf8_bytes_mean": query_feedback,
            "iid_query_aware_request_reply_bytes_mean_excluding_framing": args.slots * args.bottleneck * 4 + query_feedback,
            "iid_text_payload_utf8_bytes_mean": input_sizes["iid"]["private_text_utf8_bytes"]["mean"],
            "iid_symbolic_payload_utf8_bytes_mean": input_sizes["iid"]["private_symbolic_utf8_bytes"]["mean"],
            "query_blind_world_message_reuse_possible": True, "query_aware_world_message_reuse_across_questions_possible": False,
            "question_feedback_rule": "Conservatively charge the full public question UTF8 bytes on each query-specific request; latent reply and framing are separate.",
            "bridge_parameter_float32_bytes_per_arm": runs[0]["parameters_per_arm"] * 4,
            "sender_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in sender.parameters()),
            "receiver_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in receiver.parameters()),
            "actual_unique_source_encoder_calls": len(source_encoding_records),
            "actual_source_encoder_tokens": sum(row["tokens"] for row in source_encoding_records),
            "cached_sender_state_bytes": sum(value.numel() * value.element_size() for value in hidden_cache.values()),
            "actual_unique_public_question_encoder_calls": len(question_encoding_records),
            "actual_public_question_encoder_tokens": sum(row["sender_tokens"] for row in question_encoding_records),
            "cached_public_question_vector_bytes": sum(value.numel() * value.element_size() for value in question_cache.values()),
            "cached_receiver_embedding_bytes": sum(value.numel() * value.element_size() for value in embedding_cache.values()),
            "uncached_public_query_calls_per_seed_per_query_using_arm": {"training": args.epochs * len(train_cases),
                "validation": (args.epochs + 1) * len(validation_cases), "test_per_view": len(test_cases["iid"])},
            "cache_scope": "Each distinct source and public question encoded once across all splits/views/seeds/arms; only frozen model states cached, never trained bridge outputs.",
            "uncached_cost_scope": "Logical encoder call/token counts are reported; uncached wall time, network, framing, energy and full optimizer/activation memory were not measured.",
            "input_sizes": input_sizes,
        },
        "timings_seconds": {"model_loading": loading_seconds, "case_preparation": preparation_seconds,
            "actual_cached_source_encoding": sum(row["seconds"] for row in source_encoding_records),
            "actual_cached_public_question_encoding_and_normalization": sum(row["encoding_and_normalization_seconds"] for row in question_encoding_records),
            "all_training": sum(sum(run["arm_training_seconds"].values()) for run in runs),
            "all_validation": sum(run["validation_seconds"] for run in runs),
            "all_heldout_evaluation": sum(run["heldout_evaluation_seconds"] for run in runs),
            "entire_runner_excluding_downloads": time.perf_counter() - start},
        "limits": [
            "Small synthetic task and two fixed pretrained models; no new-method, universal-language or real-world agent-performance claim.",
            "Private-constant retains the true public question but has structural zero paired correctness; beating it alone does not establish private fact transfer.",
            "Question-dependent messages require question feedback and cannot reuse a world message across different questions.",
            "Equal parameter/init/order/update budgets do not imply equal input lengths, FLOPs or wall time.",
            "Family intervals are conditional on three fixed seeds; repeated queries, variants and views do not increase the independent family count.",
            "Source and query interventions are label-free; wrong sources can coincidentally share the answer and wrong questions can leave predictions unchanged.",
            "Frozen states are cached for actual execution. Logical uncached work is counted separately and is not a measured uncached latency.",
            "All settings and advancement targets were fixed before held-out scoring; no posthoc tuning is justified by this run.",
            "Eight-choice scoring does not measure open-ended generation. A two-byte symbolic representation makes a bandwidth-breakthrough claim inappropriate.",
        ],
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
    parser.add_argument("--sender-model", "--sender", dest="sender_model", required=True)
    parser.add_argument("--receiver-model", "--receiver", dest="receiver_model", required=True)
    parser.add_argument("--data-dir", "--data", dest="data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--bottleneck", type=int, default=8)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-manifest")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.slots, args.bottleneck, args.learning_rate, args.max_length, args.threads) <= 0 or args.weight_decay < 0:
        parser.error("sizes and learning rate must be positive; weight decay must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Optional transformer dependencies unavailable: {error}\n")


if __name__ == "__main__":
    main()
