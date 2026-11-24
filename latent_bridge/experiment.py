"""Reproducible CPU experiment; all reported measurements come from execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from .core import Bridge, FrozenAgents, HEADER, SEMANTIC_DIM, VALUES, decode_latent, decode_packed, decode_symbols, encode_english, encode_latent, encode_packed, encode_symbols, make_splits, one_hot, score, train


def experiment(seeds=(101, 202, 303), widths=(4, 8, 16), epochs=180):
    splits = make_splits()
    agents = FrozenAgents.create()
    states = {name: agents.observe(data) for name, data in splits.items()}
    labels = splits["test"]
    # Deliberately strong synthetic control: train-only least-squares decoding.
    x_train = np.column_stack((states["train"], np.ones(len(splits["train"]))))
    linear, *_ = np.linalg.lstsq(x_train, one_hot(splits["train"]), rcond=None)
    linear_scores = np.column_stack((states["test"], np.ones(len(labels)))) @ linear
    majority = np.array([np.bincount(splits["train"][:, i], minlength=VALUES).argmax() for i in range(4)])
    prior_logits = np.array([np.log(np.bincount(splits["train"][:, i], minlength=VALUES) + 1) for i in range(4)])
    baselines = {
        "no_message_train_prior": score(np.broadcast_to(prior_logits, (len(labels), 4, VALUES)), labels),
        "least_squares_full_sender_state": score(linear_scores.reshape(-1, 4, VALUES), labels),
        "uint8_symbols": {"attribute_accuracy": float(np.mean(np.array([decode_symbols(encode_symbols(row)) for row in labels]) == labels)), "payload_bytes": 4, "framed_bytes": len(encode_symbols(labels[0]))},
        "packed_symbols": {"attribute_accuracy": float(np.mean(np.array([decode_packed(encode_packed(row)) for row in labels]) == labels)), "payload_bytes": 2, "framed_bytes": len(encode_packed(labels[0]))},
        "fixed_grammar_english": {"payload_bytes_mean": float(np.mean([len(encode_english(row)) - HEADER.size for row in labels])), "framed_bytes_mean": float(np.mean([len(encode_english(row)) for row in labels])), "task_accuracy_evaluated": False},
    }
    runs = []
    for width in widths:
        for seed in seeds:
            bridge = Bridge(agents, latent_dim=width, seed=seed)
            began = time.perf_counter()
            training = train(bridge, states["train"], splits["train"], states["validation"], splits["validation"], epochs=epochs, seed=seed)
            seconds = time.perf_counter() - began
            # Deployable f32 parameters, represented as doubles for CPU algebra.
            bridge.params = {k: v.astype(np.float32).astype(np.float64) for k, v in bridge.params.items()}
            z = bridge.encode(states["test"])
            delivered = np.array([decode_latent(encode_latent(row), width) for row in z])
            # A nonzero circular shift is a derangement: never its own message.
            shift = int(np.random.default_rng(seed + 9000).integers(1, len(labels)))
            permuted = np.roll(delivered, shift, axis=0)
            params = sum(v.size for v in bridge.params.values())
            run = {
                "seed": seed, "latent_dim": width, "train_seconds": seconds,
                "validation_selection": training,
                "matched": score(bridge.decode(delivered), labels),
                "zero_message": score(bridge.decode(np.zeros_like(delivered)), labels),
                "permuted_message": score(bridge.decode(permuted), labels),
                "permutation_shift": shift,
                "latent_payload_bytes": width * 4,
                "latent_framed_bytes": len(encode_latent(z[0])),
                "bridge_parameter_count": params,
                "bridge_checkpoint_float32_bytes": params * 4,
                "training_parameters_and_adam_moments_float64_bytes": params * 8 * 3,
            }
            runs.append(run)
            print(f"latent_dim={width} seed={seed}: matched={run['matched']['attribute_accuracy']:.4f}, zero={run['zero_message']['attribute_accuracy']:.4f}, shuffled={run['permuted_message']['attribute_accuracy']:.4f}", flush=True)
    summary = []
    for width in widths:
        selected = [row for row in runs if row["latent_dim"] == width]
        summary.append({"latent_dim": width, **{f"{condition}_accuracy_mean": statistics.mean(row[condition]["attribute_accuracy"] for row in selected) for condition in ("matched", "zero_message", "permuted_message")}, "matched_accuracy_std_across_initializations": statistics.stdev([row["matched"]["attribute_accuracy"] for row in selected]) if len(selected) > 1 else None, "payload_bytes": width * 4, "framed_bytes": width * 4 + HEADER.size})
    return {
        "experiment": "synthetic_private_attribute_latent_bridge_v1",
        "status": "executed_synthetic_experiment_not_pretrained_llm_evaluation",
        "config": {"seeds": list(seeds), "latent_widths": list(widths), "epochs": epochs, "split_seed": 23, "frozen_agents_seed": 11, "fields": 4, "values_per_field": 8, "sender_dim": 48, "receiver_dim": 40, "bridge_hidden_dim": 64, "learning_rate": 0.003, "batch_size": 128},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "splits": {name: {"worlds": len(data), "attribute_queries": len(data) * 4, "sha256_uint8_rows": hashlib.sha256(data.tobytes()).hexdigest()} for name, data in splits.items()},
        "baselines": baselines,
        "setup_accounting": {"frozen_agent_weights_float32_bytes": (agents.sender_weight.size + agents.receiver_readout.size) * 4, "raw_training_worlds_uint8_bytes": splits["train"].nbytes, "raw_validation_worlds_uint8_bytes": splits["validation"].nbytes, "raw_test_worlds_uint8_bytes": splits["test"].nbytes, "cached_train_sender_states_float64_bytes": states["train"].nbytes, "common_frame_header_bytes": HEADER.size, "shared_schema": "four attributes a,b,c,d, each an integer from 0 through 7; receiver knows query index", "network_transfer_measured": False, "training_storage_is_not_total_training_io_or_energy": True},
        "summary": summary, "runs": runs,
        "interpretation": ["Held-out worlds are unseen combinations of familiar values; this is not a domain-shift test.", "Frozen coordinates are deliberately related to known attribute semantics. This is a favorable bridge smoke test, not a discovery of a universal representation.", "The symbolic baselines are exact codecs, not language models; fixed-grammar English bytes are counted but its task accuracy is not evaluated.", "Float latents use more payload bytes than the lossless packed-symbol baseline. No bandwidth, token, or end-to-end speed advantage is claimed.", "Multiple seeds vary bridge initialization and minibatch order, not the fixed train/test split. Standard deviations are not confidence intervals over independent tasks.", "Zero and permuted messages intervene on the receiver channel. Test labels and sender states are not otherwise accessible to the receiver.", "No pretrained model has been trained or evaluated in this experiment."],
    }


def report(result):
    rows = ["# Executed synthetic experiment", "", "This is a controlled learned-channel experiment, **not an LLM benchmark or a scientific breakthrough**.", "", "| Latent width | Matched query accuracy | Zero message | Permuted message | Payload bytes | Framed bytes |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        rows.append(f"| {row['latent_dim']} | {row['matched_accuracy_mean']:.2%} | {row['zero_message_accuracy_mean']:.2%} | {row['permuted_message_accuracy_mean']:.2%} | {row['payload_bytes']} | {row['framed_bytes']} |")
    rows += ["", "Accuracies are means over the recorded bridge seeds. Each seed uses the same held-out world split. The receiver chooses among eight possible answers; uniform chance is 12.5%.", "", f"The train-only least-squares decoder using the full sender state scores {result['baselines']['least_squares_full_sender_state']['attribute_accuracy']:.2%}. This strong control exposes how favorable the synthetic representation is.", "", "The lossless packed-symbol baseline scores 100% and uses **2 payload bytes / 10 framed bytes**. Four uint8 symbols also score 100% with 4 payload bytes / 12 framed bytes. These codecs have explicit access to the synthetic attributes, as does the sender before its frozen observation transform.", "", "**Finding:** this pipeline can be tested for causal use of a learned continuous channel. These results do not establish better bandwidth than symbols, better inference than English, interoperability between pretrained models, or novelty.", "", "## Reproduction and accounting", "", "The adjacent JSON contains every run, configuration, split hashes, validation checkpoint selection, training times, bridge checkpoint size, frozen-agent setup size, raw training data size, and feature-cache size. Timing is environment-specific. Optimizer storage is an estimate for the arrays named; it excludes transient activations, process overhead, total I/O, and energy.", "", "All interpretation limits:", ""]
    rows.extend("- " + note for note in result["interpretation"])
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic.json"))
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.widths)) != len(args.widths):
        parser.error("seeds and widths must be unique")
    result = experiment(args.seeds, args.widths, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(report(result))


if __name__ == "__main__":
    main()
