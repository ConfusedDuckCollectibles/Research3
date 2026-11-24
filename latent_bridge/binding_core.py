"""Fixed loss, routing, token preflight and reporting for the binding study."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import numpy as np

from .binding_data import COLORS, MEMBERS, OBJECTS, parse_question, public_question
from .query_conditioned import make_query_bridge, normalize_question, constant_question
from .transformers_runner import constant_observation
from .counterfactual_runner import selected_logits

SEEDS = (101, 202, 303)
ARMS = ("aware_ce", "aware_aux", "blind_ce", "private_constant", "text", "symbolic")
MODEL_CONDITIONS = ARMS + tuple(
    arm + "_" + intervention for arm in ("aware_ce", "aware_aux")
    for intervention in ("wrong_source", "wrong_question", "zero")
) + ("frozen_no_channel",)
CONDITIONS = MODEL_CONDITIONS + ("rank_codec_oracle",)
RECEIVER_SEPARATOR = "\n"
EPOCHS, QUARTETS_PER_BATCH, SLOTS, BOTTLENECK, MAX_LENGTH = 8, 4, 2, 8, 192
LEARNING_RATE, WEIGHT_DECAY, CLIP_NORM, MARGIN, AUX_WEIGHT = .001, .01, 1., .25, 1.
DESIGN_COMMIT = "dfce06b5dfb2860414af166ef4441eefd9548afc"
HISTORICAL_HELPER_COMMIT = "9eb4478fae1ad373234eb2a725345b66c8c63c57"
PINNED_MODELS = {
    "sender": {"id": "distilbert/distilgpt2", "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa"},
    "receiver": {"id": "EleutherAI/pythia-14m", "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac"},
}
PINNED_PACKAGES = {
    "torch": "2.6.0+cpu", "transformers": "4.48.2", "numpy": "2.4.6",
    "tokenizers": "0.21.4", "safetensors": "0.8.0", "huggingface-hub": "0.36.2",
}
CONFIG = {
    "seeds": list(SEEDS), "epochs": EPOCHS, "quartets_per_batch": QUARTETS_PER_BATCH,
    "records_per_batch": 16, "slots": SLOTS, "bottleneck": BOTTLENECK,
    "max_length": MAX_LENGTH, "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": CLIP_NORM,
    "auxiliary_weight": AUX_WEIGHT, "margin": MARGIN, "message_coordinates": 16,
    "receiver_separator": RECEIVER_SEPARATOR, "query_normalization_epsilon": 1e-5,
    "private_control_seed_offset": 9841, "fixed_question_seed_offset": 17291,
    "bootstrap_seed": 66173, "bootstrap_resamples": 2000, "threads": 2,
}


def flattened_cases(quartets):
    return [{"family": q["family"], "query_index": q["query_index"],
             "query": q["query"], "variant": m["variant"], "member": m}
            for q in quartets for m in q["members"]]


def target_indices(cases):
    return np.asarray([COLORS.index(case["member"]["answer"]) for case in cases], dtype=np.int64)


def receiver_text(case, arm):
    if arm in ("text", "symbolic"):
        field = "source" if arm == "text" else "symbolic"
        return case["member"][field] + RECEIVER_SEPARATOR + case["query"]
    return case["query"]


def wrong_public_question(question):
    return public_question((parse_question(question) + 1) % len(OBJECTS))


def route_inputs(arm, hidden, mask, question, fixed_hidden, fixed_mask, fixed_question):
    if arm not in ARMS:
        raise ValueError("unknown training arm")
    if arm in ("private_constant", "text", "symbolic"):
        hidden = fixed_hidden.expand(question.shape[0], -1, -1)
        mask = fixed_mask.expand(question.shape[0], -1)
    if arm == "blind_ce":
        question = fixed_question[None, :].expand(question.shape[0], -1)
    return hidden, mask, question


def quartet_loss(logits, targets, messages, auxiliary=False):
    import torch
    from torch.nn import functional as F
    if logits.ndim != 2 or logits.shape[1] != 8 or len(logits) % 4:
        raise ValueError("loss requires whole O,T,N,R quartets and eight logits")
    if messages.shape != (len(logits), SLOTS, BOTTLENECK) or targets.shape != (len(logits),):
        raise ValueError("invalid message or target shape")
    if not bool(torch.isfinite(messages).all()) or bool((messages.abs() > 1.000001).any()):
        raise ValueError("loss must use the actual bounded Tanh message")
    z = messages.reshape(-1, 4, 16)
    distances = {
        "nuisance_distance": (z[:, 0] - z[:, 2]).square().mean(dim=1),
        "reorder_distance": (z[:, 0] - z[:, 3]).square().mean(dim=1),
        "target_distance": (z[:, 0] - z[:, 1]).square().mean(dim=1),
    }
    hinge = (MARGIN - distances["target_distance"]).clamp_min(0.)
    aux = .5 * (distances["nuisance_distance"] + distances["reorder_distance"]) + hinge
    ce = F.cross_entropy(logits, targets, reduction="none").reshape(-1, 4).mean(dim=1)
    loss = (ce + AUX_WEIGHT * aux if auxiliary else ce).mean()
    terms = {name: value.mean() for name, value in distances.items()}
    terms.update(restricted_ce=ce.mean(), hinge=hinge.mean(), auxiliary=aux.mean(), objective=loss)
    return loss, terms


def epoch_orders(seed, quartet_count=512):
    if seed not in SEEDS or quartet_count != 512:
        raise ValueError("epoch ordering must use the frozen cohort and seeds")
    result = []
    for epoch in range(1, EPOCHS + 1):
        order = list(range(quartet_count))
        random.Random(seed + 1009 * epoch).shuffle(order)
        result.append(order)
    return result


def order_hashes(orders):
    return [hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
            for order in orders]


def select_epoch(history):
    if not history or any(not np.isfinite(row["validation_ce"]) for row in history):
        raise ValueError("checkpoint selection requires finite validation CE")
    return min(history, key=lambda row: (row["validation_ce"], row["epoch"]))["epoch"]


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        array = np.ascontiguousarray(array)
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in sorted(state.items())}
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    value = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(value).hexdigest(),
            "bytes": len(value), "state_sha256": state_hash(state)}


def load_state(path):
    import torch
    with np.load(path, allow_pickle=False) as archive:
        return {name: torch.from_numpy(archive[name].copy()) for name in archive.files}


def stats(values):
    values = list(values)
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {"count": len(values), "min": min(values), "mean": float(np.mean(values)),
            "max": max(values), "sum": sum(values)}


def tokenizer_preflight(partitions, tokenizers):
    """Data-only checks include test text, but never evaluate a model."""
    candidates = [tokenizers["receiver"].encode(" " + color, add_special_tokens=False) for color in COLORS]
    if any(len(ids) != 1 for ids in candidates) or len({ids[0] for ids in candidates}) != 8:
        raise ValueError("colors must have distinct single leading-space candidate tokens")
    references, checked, lengths = {}, {}, {}
    for role, tokenizer in tokenizers.items():
        token_cache = {}
        def tokens(text):
            if text not in token_cache:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    raise ValueError("empty token sequence")
                token_cache[text] = ids
            return token_cache[text]
        for split, quartets in partitions.items():
            counts = {"private_text": [], "private_symbolic": [], "query": [],
                      "latent_receiver": [], "text_receiver": [], "symbolic_receiver": [],
                      "frozen_no_channel": []}
            for quartet in quartets:
                query = quartet["query"]
                for member in quartet["members"]:
                    for field in ("source", "symbolic"):
                        ids = tokens(member[field])
                        bag = tuple(sorted(Counter(ids).items()))
                        key = (role, field)
                        if key in references and references[key] != bag:
                            raise ValueError("private tokenizer multiset changed: " + role + "/" + field)
                        references[key] = bag
                        counts["private_text" if field == "source" else "private_symbolic"].append(len(ids))
                        if len(ids) > MAX_LENGTH:
                            raise ValueError("private sequence exceeds maximum length")
                    query_length = len(tokens(query))
                    counts["query"].append(query_length)
                    if query_length + SLOTS > MAX_LENGTH:
                        raise ValueError("query exceeds maximum length")
                    counts["latent_receiver"].append(query_length + SLOTS)
                    counts["frozen_no_channel"].append(query_length)
                    for field, counter in (("source", "text_receiver"), ("symbolic", "symbolic_receiver")):
                        ids = tokens(member[field] + RECEIVER_SEPARATOR + query)
                        key = (role, "complete_" + field, query)
                        bag = tuple(sorted(Counter(ids).items()))
                        if key in references and references[key] != bag:
                            raise ValueError("complete-prompt multiset changed conditional on question")
                        references[key] = bag
                        counts[counter].append(len(ids) + SLOTS)
                        if len(ids) + SLOTS > MAX_LENGTH:
                            raise ValueError("complete receiver sequence exceeds maximum length")
            lengths[role + "/" + split] = {name: stats(value) for name, value in counts.items()}
        checked[role] = len(token_cache)
    return {
        "passed": True, "candidate_colors": list(COLORS),
        "candidate_token_ids": [ids[0] for ids in candidates],
        "private_and_complete_conditional_token_multisets_equal": True,
        "tokenizer_unique_strings": checked, "token_lengths": lengths,
        "bag_sha256": hashlib.sha256(json.dumps(
            [(list(key), list(bag)) for key, bag in sorted(references.items())],
            separators=(",", ":")).encode()).hexdigest(),
        "receiver_separator": RECEIVER_SEPARATOR,
        "test_access": "data-only tokenization and consistency; no model features or predictions",
    }


def grouped_quartets(rows):
    grouped = {}
    for row in rows:
        key = (row["family"], row["query_index"])
        member = grouped.setdefault(key, {})
        if row["variant"] in member:
            raise ValueError("duplicate prediction member")
        member[row["variant"]] = row
    if any(set(value) != set(MEMBERS) for value in grouped.values()):
        raise ValueError("incomplete quartet predictions")
    return grouped


def quartet_metrics(members):
    original, target, nuisance, reorder = [members[name] for name in MEMBERS]
    correct = {name: members[name]["choice"] == members[name]["gold"] for name in MEMBERS}
    metrics = {
        "record_accuracy": sum(correct.values()) / 4,
        "target_paired_both_correct": float(correct["O"] and correct["T"]),
        "target_answer_change": float(original["choice"] != target["choice"]),
        "nuisance_stability": float(original["choice"] == nuisance["choice"]),
        "nuisance_both_correct": float(correct["O"] and correct["N"]),
        "reorder_stability": float(original["choice"] == reorder["choice"]),
        "reorder_both_correct": float(correct["O"] and correct["R"]),
    }
    if all(members[name].get("restricted_ce") is not None for name in MEMBERS):
        metrics["restricted_ce"] = float(np.mean([members[name]["restricted_ce"] for name in MEMBERS]))
    if all("source_gold" in members[name] for name in MEMBERS):
        supplied = {name: members[name]["choice"] == members[name]["source_gold"] for name in MEMBERS}
        metrics["supplied_source_record_accuracy"] = sum(supplied.values()) / 4
        metrics["supplied_source_target_paired_both_correct"] = float(supplied["O"] and supplied["T"])
        metrics["supplied_source_answer_unchanged_fraction"] = np.mean(
            [members[name]["source_gold"] == members[name]["gold"] for name in MEMBERS]).item()
    return metrics


def validation_metrics(rows):
    groups = grouped_quartets(rows)
    if len(groups) != 256 or len({key[0] for key in groups}) != 32:
        raise ValueError("validation must cover all 32 families and eight queries")
    families = {key[0] for key in groups}
    if any({query for family, query in groups if family == selected} != set(range(8)) for selected in families):
        raise ValueError("validation must contain all eight public queries in every family")
    metrics = [quartet_metrics(value) for value in groups.values()]
    return {
        "quartets": len(groups), "records": len(rows),
        "metrics": {key: float(np.mean([value[key] for value in metrics])) for key in metrics[0]},
        "counts": {
            "target_paired_both_correct": sum(int(value["target_paired_both_correct"]) for value in metrics),
            "nuisance_stable": sum(int(value["nuisance_stability"]) for value in metrics),
            "reorder_stable": sum(int(value["reorder_stability"]) for value in metrics),
            "record_correct": sum(row["choice"] == row["gold"] for row in rows),
        },
    }


def competence_gate(text_fits):
    if len(text_fits) != 3 or {row["seed"] for row in text_fits} != set(SEEDS) or any(
        row["arm"] != "text" or not row.get("complete") for row in text_fits
    ):
        raise ValueError("gate requires all three completed selected text fits")
    rows = []
    for fit in sorted(text_fits, key=lambda row: row["seed"]):
        validation = fit["validation"]
        metrics = validation["metrics"]
        passed = (metrics["target_paired_both_correct"] >= .8 and
                  metrics["nuisance_stability"] >= .9 and metrics["reorder_stability"] >= .9)
        rows.append({"seed": fit["seed"], "passed": passed, "validation": validation,
                     "selected_epoch": fit["selected_epoch"], "state_sha256": fit["state_sha256"]})
    return {"passed": all(row["passed"] for row in rows), "per_seed": rows,
            "failed_seeds": [row["seed"] for row in rows if not row["passed"]],
            "thresholds": {"target_paired_both_correct": .8, "nuisance_stability": .9, "reorder_stability": .9},
            "fresh_test_scoring_started": False, "other_arms_started": False}


def authorize_fit(arm, completed, gate):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm != "text" and (gate is None or not gate.get("passed") or
                         {row["seed"] for row in completed if row["arm"] == "text" and row.get("complete")} != set(SEEDS)):
        raise ValueError("non-text fitting is forbidden before every text control passes")


def freeze_selection(fits, gate):
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if not gate.get("passed") or len(fits) != 18 or {
        (fit["seed"], fit["arm"]) for fit in fits if fit.get("complete")
    } != expected:
        raise ValueError("all 18 fits must complete before fresh test model access")
    for seed in SEEDS:
        selected = [fit for fit in fits if fit["seed"] == seed]
        if len({fit["initial_state_sha256"] for fit in selected}) != 1 or any(
            fit["updates"] != 1024 or fit["epoch_order_sha256"] != order_hashes(epoch_orders(seed))
            for fit in selected
        ):
            raise ValueError("adaptation initialization/order/update budget mismatch")
    return {"all_required_fits_complete": True, "fresh_test_model_cases_loaded": False,
            "config": CONFIG, "competence_gate": gate,
            "checkpoints": [{key: fit[key] for key in (
                "seed", "arm", "selected_epoch", "state_sha256", "checkpoint", "query_gates",
                "initial_state_sha256", "updates", "epoch_order_sha256")}
                for fit in sorted(fits, key=lambda row: (row["seed"], row["arm"]))]}


def require_test_authorized(selection_path, fits, gate):
    path = Path(selection_path)
    expected = freeze_selection(fits, gate)
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("fresh model access requires the complete persisted selection manifest")
    return expected


def source_successor_map(quartets):
    families = sorted({quartet["family"] for quartet in quartets})
    if len(families) < 2:
        raise ValueError("source intervention requires distinct families")
    successor = dict(zip(families, families[1:] + families[:1]))
    index = {(q["family"], q["query_index"]): number for number, q in enumerate(quartets)}
    if len(index) != len(quartets):
        raise ValueError("duplicate source quartet")
    return [index[(successor[q["family"]], q["query_index"])] for q in quartets]


def summarize_predictions(predictions):
    grouped = {}
    for row in predictions:
        grouped.setdefault((row["condition"], row["seed"]), []).append(row)
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(grouped) != expected:
        raise ValueError("primary complete-study report requires every evaluation condition and seed")
    family_ids = None
    values = {}
    for key, rows in grouped.items():
        quartets = grouped_quartets(rows)
        families = sorted({family for family, _ in quartets})
        if len(families) != 64 or len(quartets) != 512 or len(rows) != 2048:
            raise ValueError("incomplete fresh prediction grid")
        if family_ids is None:
            family_ids = families
        if families != family_ids or any(
            {query for family, query in quartets if family == selected} != set(range(8))
            for selected in families
        ):
            raise ValueError("fresh family/query grids differ")
        values[key] = [
            {metric: float(np.mean([quartet_metrics(quartets[(family, qi)])[metric] for qi in range(8)]))
             for metric in quartet_metrics(quartets[(family, 0)])}
            for family in families
        ]
    samples = np.random.default_rng(66173).integers(0, 64, size=(2000, 64))
    family_values, conditions = {}, []
    for condition in CONDITIONS:
        metrics = {}
        for metric in values[(condition, SEEDS[0])][0]:
            matrix = np.asarray([[row[metric] for row in values[(condition, seed)]] for seed in SEEDS])
            means = matrix.mean(axis=0)
            interval = np.percentile(means[samples].mean(axis=1), [2.5, 97.5])
            metrics[metric] = {"mean": float(means.mean()), "family_cluster_95_interval": interval.tolist(),
                              "seed_means": {str(seed): float(matrix[i].mean()) for i, seed in enumerate(SEEDS)},
                              "seed_range": [float(matrix.mean(axis=1).min()), float(matrix.mean(axis=1).max())]}
            family_values[(condition, metric)] = means
        conditions.append({"condition": condition, "metrics": metrics})
    contrasts = [("primary_aux_minus_ce", "aware_aux", "aware_ce"),
                 ("aux_minus_blind", "aware_aux", "blind_ce")]
    contrasts += [(arm + "_minus_" + intervention, arm, arm + "_" + intervention)
                  for arm in ("aware_ce", "aware_aux")
                  for intervention in ("wrong_source", "wrong_question", "zero")]
    comparisons = []
    metric = "target_paired_both_correct"
    for name, a, b in contrasts:
        difference = family_values[(a, metric)] - family_values[(b, metric)]
        interval = np.percentile(difference[samples].mean(axis=1), [2.5, 97.5])
        seed_differences = {
            str(seed): float(np.mean([row[metric] for row in values[(a, seed)]]) -
                             np.mean([row[metric] for row in values[(b, seed)]]))
            for seed in SEEDS}
        comparisons.append({"contrast": name, "a": a, "b": b,
                            "mean_difference": float(difference.mean()),
                            "family_cluster_95_interval": interval.tolist(),
                            "seed_differences": seed_differences,
                            "seed_difference_range": [min(seed_differences.values()), max(seed_differences.values())]})
    return {"conditions": conditions, "paired_comparisons": comparisons,
            "family_count": 64, "bootstrap_resamples": 2000, "bootstrap_seed": 66173,
            "interval_scope": "descriptive family intervals conditional on three fixed seed pairs"}
