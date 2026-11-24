"""Small probe/readout helpers with explicit public-input and selection rules."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .counterfactual_runner import METRICS, paired_family_scores
from .private_counterfactual import OBJECTS, query_text

MESSAGE_ARMS = ("message_aware", "message_blind", "message_private_constant", "query_only", "message_aware_shuffled_labels")
LARGE_ARMS = ("pooled_aware", "pooled_blind", "full_source")
LINEAR_ARMS = MESSAGE_ARMS + LARGE_ARMS


def public_head(question):
    mapping = {query_text(index, shifted): index for shifted in (False, True) for index in range(4)}
    if question not in mapping:
        raise ValueError("public question does not match the declared parser; metadata fallback is forbidden")
    return mapping[question]


def fit_preprocessing(training_features):
    values = np.asarray(training_features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid training feature matrix")
    mean, deviation = values.mean(axis=0), values.std(axis=0)
    scale = np.where(deviation >= 1e-4, deviation, 1.0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "dimension": values.shape[1]}


def preprocess(features, statistics):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != statistics["dimension"]:
        raise ValueError("feature width does not match training-only statistics")
    if not np.isfinite(values).all() or np.any(statistics["scale"] <= 0):
        raise ValueError("non-finite features or invalid scale")
    return ((values - statistics["mean"]) / statistics["scale"] / math.sqrt(statistics["dimension"])).astype(np.float32)


def flatten_source(states, max_positions=64):
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or not 0 < len(values) <= max_positions:
        raise ValueError("full-source overflow/empty states: truncation and answer-position selection are forbidden")
    result = np.zeros((max_positions, values.shape[1]), dtype=np.float32)
    result[:len(values)] = values
    return result.reshape(-1)


def pool_before_compressor(bridge, hidden, mask, question):
    """Exactly the frozen attention pool, before compressor LayerNorm."""
    import torch

    queries = bridge.queries[None] + bridge.gates[None, :, None] * question[:, None, :]
    logits = torch.einsum("bkd,bsd->bks", queries, hidden) / math.sqrt(bridge.sender_dim)
    logits = logits.masked_fill(~mask[:, None, :].bool(), torch.finfo(logits.dtype).min)
    return torch.einsum("bks,bsd->bkd", logits.softmax(dim=-1), hidden)


def make_probe(dimension, kind="linear"):
    import torch
    from torch import nn

    if dimension <= 0 or kind not in ("linear", "mlp") or (kind == "mlp" and dimension != 16):
        raise ValueError("unsupported prespecified probe")

    class PublicQuestionProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleList([nn.Linear(dimension, 8) if kind == "linear" else
                nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)) for _ in OBJECTS])

        def forward(self, features, heads):
            if features.ndim != 2 or features.shape[1] != dimension or heads.shape != (len(features),):
                raise ValueError("feature/head shape mismatch")
            if bool(((heads < 0) | (heads >= len(OBJECTS))).any()):
                raise ValueError("unknown public head")
            # Compute only the requested head for each row, retaining gradients.
            output = features.new_zeros((len(features), 8))
            for index, layer in enumerate(self.heads):
                positions = torch.nonzero(heads == index, as_tuple=True)[0]
                if len(positions):
                    output = output.index_copy(0, positions, layer(features.index_select(0, positions)))
            return output

    return PublicQuestionProbe()


def label_derangement(cases):
    families = sorted({case["family"] for case in cases})
    if len(families) < 2 or any(case["variant"] != "original" for case in cases):
        raise ValueError("label control requires original training worlds only")
    next_family = {family: families[(index + 1) % len(families)] for index, family in enumerate(families)}
    lookup = {(case["family"], public_head(case["query"])): index for index, case in enumerate(cases)}
    if len(lookup) != len(cases):
        raise ValueError("duplicate training world/question")
    return [lookup[(next_family[case["family"]], public_head(case["query"]))] for case in cases]


def secondary_trigger(validation_pair_accuracy, seeds):
    if set(validation_pair_accuracy) != set(seeds) or any(not 0 <= value <= 1 for value in validation_pair_accuracy.values()):
        raise ValueError("secondary decision requires every seed's selected linear validation score")
    return {"triggered": any(value < 0.8 for value in validation_pair_accuracy.values()),
            "threshold": 0.8, "criterion": "any selected linear-aware validation paired accuracy below 80%",
            "per_seed_validation_pair_accuracy": {str(seed): validation_pair_accuracy[seed] for seed in seeds},
            "decision_uses_fresh_test": False}


def freeze_selection(selected, seeds, trigger):
    expected = {(seed, "linear", arm) for seed in seeds for arm in LINEAR_ARMS}
    if trigger["triggered"]:
        expected |= {(seed, "mlp", arm) for seed in seeds for arm in MESSAGE_ARMS}
    if set(selected) != expected:
        raise ValueError("fresh scoring forbidden until every declared probe checkpoint is fixed")
    if any(not value.get("state_sha256") or not value.get("selected_epoch") for value in selected.values()):
        raise ValueError("missing selected-state identity")
    return {"all_probe_checkpoints_fixed": True, "trigger": trigger,
            "checkpoints": [{"seed": seed, "reader": kind, "arm": arm, **selected[(seed, kind, arm)]}
                            for seed, kind, arm in sorted(selected)],
            "fresh_test_scoring_started": False}


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize(predictions, seeds, resamples=2000, bootstrap_seed=66173):
    scores = paired_family_scores(predictions)
    families = sorted({key[3] for key in scores})
    conditions = sorted({key[2] for key in scores})
    views = sorted({key[1] for key in scores})
    draws = np.random.default_rng(bootstrap_seed).integers(0, len(families), (resamples, len(families)))
    arrays, reports = {}, []
    for view in views:
        for condition in conditions:
            metrics = {}
            for metric in METRICS:
                values = np.array([[scores[(seed, view, condition, family)][metric] for family in families] for seed in seeds])
                average = values.mean(axis=0)
                arrays[(view, condition, metric)] = average
                interval = np.quantile(average[draws].mean(axis=1), [.025, .975])
                metrics[metric] = {"mean": float(average.mean()), "family_cluster_95_percentile_interval": interval.tolist(),
                    "seed_means": dict(zip(map(str, seeds), map(float, values.mean(axis=1)))),
                    "seed_range": [float(values.mean(axis=1).min()), float(values.mean(axis=1).max())]}
            reports.append({"view": view, "condition": condition, "families": len(families), "metrics": metrics})
    contrasts = [("linear_message_aware", "receiver_query_aware", "primary_accessibility_gap"),
                 ("linear_message_aware", "linear_message_blind", "linear_aware_minus_blind"),
                 ("linear_message_aware", "linear_message_aware_wrong_source", "linear_aware_source_effect"),
                 ("linear_pooled_aware", "linear_message_aware", "larger_pool_minus_message"),
                 ("linear_full_source", "linear_message_aware", "larger_source_minus_message")]
    if "mlp_message_aware" in conditions:
        contrasts += [("mlp_message_aware", "linear_message_aware", "secondary_mlp_minus_linear"),
                      ("mlp_message_aware", "mlp_message_aware_wrong_source", "secondary_mlp_source_effect")]
    comparisons = []
    for view in views:
        for left, right, name in contrasts:
            delta = arrays[(view, left, "paired_both_correct")] - arrays[(view, right, "paired_both_correct")]
            per_seed = {str(seed): float(np.mean([scores[(seed, view, left, family)]["paired_both_correct"] - scores[(seed, view, right, family)]["paired_both_correct"] for family in families])) for seed in seeds}
            comparisons.append({"view": view, "contrast": name, "left": left, "right": right,
                "mean_difference": float(delta.mean()), "paired_family_cluster_95_percentile_interval": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                "seed_mean_differences": per_seed})
    return {"conditions": reports, "paired_comparisons": comparisons,
        "primary_contrast": {"view": "iid", "contrast": "primary_accessibility_gap", "metric": "paired_both_correct"},
        "uncertainty": {"families": len(families), "resamples": resamples, "seed": bootstrap_seed,
            "method": "Average four queries and three paired bridge/probe seeds within each family; resample the same whole-family indices for all contrasts/views.",
            "scope": "Descriptive intervals conditional on three paired seeds, not population-level uncertainty; larger controls and MLP are secondary."}}
