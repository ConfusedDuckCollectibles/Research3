"""Local-model latent-bridge experiment with causal and adaptation controls.

No model download is allowed. Base models stay frozen; gradients flow through
the receiver into a small learned continuous prefix. Metrics include NLL and
bounded greedy generated-answer exact match. No production speed claim follows.
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


def read_partitions(paths):
    """Validate data before any model is loaded; IDs and worlds cannot cross splits."""
    parts, ids, observations, worlds = {}, {}, {}, {}
    for name, path in paths.items():
        rows = []
        for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "sender", "query", "answer")):
                raise ValueError(f"{name}:{line_no}: require nonempty string id/sender/query/answer")
            if row["id"] in ids:
                raise ValueError(f"duplicate id across records: {row['id']}")
            observation = hashlib.sha256(row["sender"].encode()).hexdigest()
            if observation in observations and observations[observation] != name:
                raise ValueError("identical sender observations cannot cross data partitions")
            if "world_id" in row:
                if not isinstance(row["world_id"], str) or not row["world_id"]:
                    raise ValueError("world_id must be a nonempty string")
                if row["world_id"] in worlds and worlds[row["world_id"]] != name:
                    raise ValueError("world identities cannot cross data partitions, including paraphrases")
                worlds[row["world_id"]] = name
            if "symbolic" in row and (not isinstance(row["symbolic"], str) or not row["symbolic"].strip()):
                raise ValueError("symbolic must be a nonempty string when provided")
            ids[row["id"]], observations[observation] = name, name
            rows.append(row)
        if not rows:
            raise ValueError(f"{name}: empty partition")
        parts[name] = rows
    return parts


def different_observation_indices(rows):
    """Cyclic derangement of observations, not merely row IDs.

    Reject a test ordering without such a derangement instead of silently using
    an unchanged private observation in the causal control.
    """
    observations = [row.get("world_id", row["sender"]) for row in rows]
    for shift in range(1, len(rows)):
        indices = [(index + shift) % len(rows) for index in range(len(rows))]
        if all(observations[index] != observations[source] for index, source in enumerate(indices)):
            return indices
    raise ValueError("test observations require a cyclic derangement with different sender facts; deduplicate or reorder test worlds")


def make_bridge(sender_dim, receiver_dim, bottleneck=16, slots=4):
    """Returns a torch module lazily so the NumPy package needs no torch import.

    encode() lives with the sender. Only its slots x bottleneck floats cross
    the channel. decode() lives with the receiver and expands to soft tokens.
    """
    import torch
    from torch import nn

    if min(sender_dim, receiver_dim, bottleneck, slots) <= 0:
        raise ValueError("all dimensions must be positive")

    class SoftPrefixBridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = nn.Parameter(torch.randn(slots, sender_dim) / math.sqrt(sender_dim))
            self.compressor = nn.Sequential(nn.LayerNorm(sender_dim), nn.Linear(sender_dim, bottleneck), nn.Tanh())
            self.expander = nn.Linear(bottleneck, receiver_dim)

        def encode(self, hidden, mask):
            if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not bool(mask.any(dim=1).all()):
                raise ValueError("invalid hidden sequence or all-masked observation")
            logits = torch.einsum("kd,bsd->bks", self.queries, hidden) / math.sqrt(sender_dim)
            logits = logits.masked_fill(~mask[:, None, :].bool(), torch.finfo(logits.dtype).min)
            pooled = torch.einsum("bks,bsd->bkd", logits.softmax(dim=-1), hidden)
            return self.compressor(pooled)

        def decode(self, message):
            if message.ndim != 3 or message.shape[1:] != (slots, bottleneck):
                raise ValueError("latent message has wrong shape")
            return self.expander(message)

        def forward(self, hidden, mask):
            return self.decode(self.encode(hidden, mask))

    return SoftPrefixBridge()


def answer_batch(tokenizer, query, answer, max_length, device):
    """Tokenize prompt and answer separately to make the label boundary exact."""
    import torch

    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not prompt_ids or not answer_ids:
        raise ValueError("query and answer must each encode to at least one token")
    # No silent truncation: it could remove the target or private facts.
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError("query+answer exceeds max length; curate data or raise limit")
    tokens = torch.tensor([prompt_ids + answer_ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + answer_ids], device=device)
    return tokens, labels, len(answer_ids)


def prefix_embeddings(receiver, bridge, hidden, mask, condition, text_prefix, reference):
    """Build the same condition-specific prefix for training and generation."""
    import torch
    if condition in ("text", "symbolic"):
        if text_prefix is None:
            raise ValueError("text/symbolic condition requires the complete source prefix")
        return receiver.get_input_embeddings()(text_prefix)
    elif condition == "no_channel":
        return reference[:, :0]
    elif condition in ("matched", "zero", "permuted", "trained_constant"):
        message = bridge.encode(hidden, mask)
        if condition == "zero":
            message = torch.zeros_like(message)
        return bridge.decode(message).to(reference.dtype)
    raise ValueError(f"unknown evaluation condition: {condition}")


def answer_loss(receiver, bridge, hidden, mask, tokens, labels, condition="matched", text_prefix=None):
    """The causal model shifts labels internally; prefix and prompt stay masked."""
    import torch

    embeddings = receiver.get_input_embeddings()(tokens)
    prefix = prefix_embeddings(receiver, bridge, hidden, mask, condition, text_prefix, embeddings)
    inputs = torch.cat((prefix.to(embeddings.dtype), embeddings), dim=1)
    ignored = torch.full((1, prefix.shape[1]), -100, dtype=torch.long, device=labels.device)
    combined_labels = torch.cat((ignored, labels), dim=1)
    attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    # receiver.eval() disables dropout, but do NOT use no_grad while training.
    return receiver(inputs_embeds=inputs, attention_mask=attention_mask, labels=combined_labels, use_cache=False).loss


def normalize_answer(answer):
    """A prespecified normalized exact match; punctuation remains significant."""
    return " ".join(answer.strip().casefold().split())


def greedy_answer(receiver, bridge, hidden, mask, prompt_tokens, tokenizer,
                  condition="matched", text_prefix=None, max_new_tokens=4):
    """Generate from query only. Neither target IDs nor target length are read."""
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    embeddings = receiver.get_input_embeddings()(prompt_tokens)
    prefix = prefix_embeddings(receiver, bridge, hidden, mask, condition, text_prefix, embeddings)
    inputs = torch.cat((prefix, embeddings), dim=1)
    generated = []
    # Intentionally simple bounded uncached decoding; not a speed benchmark.
    with torch.no_grad():
        for _ in range(max_new_tokens):
            attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
            logits = receiver(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits[:, -1, :]
            next_id = int(logits.argmax(dim=-1).item())
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in decoded:
                return decoded.split("\n", 1)[0], len(generated)
            next_embedding = receiver.get_input_embeddings()(torch.tensor([[next_id]], device=inputs.device))
            inputs = torch.cat((inputs, next_embedding), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def fingerprint_directory(path):
    """Hash model/tokenizer files, including weights, without loading them in RAM."""
    files = {}
    for candidate in sorted(Path(path).iterdir()):
        if candidate.is_file() and candidate.suffix in (".json", ".safetensors", ".bin", ".txt", ".model"):
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[candidate.name] = {"sha256": digest.hexdigest(), "bytes": candidate.stat().st_size}
    return files


def run(args):
    paths = {"train": args.train, "validation": args.validation, "test": args.test}
    if args.generalization_test:
        paths["generalization"] = args.generalization_test
    partitions = read_partitions(paths)
    evaluation_names = [name for name in paths if name not in ("train", "validation")]
    permutations = {name: different_observation_indices(partitions[name]) for name in evaluation_names}
    for path in (args.sender_model, args.receiver_model):
        if not Path(path).is_dir():
            raise ValueError("model paths must be existing local directories; downloads are disabled")
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    began = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    sender_tokenizer = AutoTokenizer.from_pretrained(args.sender_model, **kwargs)
    receiver_tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, **kwargs)
    sender = AutoModel.from_pretrained(args.sender_model, **kwargs).to(args.device).float().eval()
    receiver = AutoModelForCausalLM.from_pretrained(args.receiver_model, **kwargs).to(args.device).float().eval()
    for model in (sender, receiver):
        for param in model.parameters():
            param.requires_grad_(False)
        for attribute in ("max_position_embeddings", "n_positions"):
            limit = getattr(model.config, attribute, None)
            if isinstance(limit, int) and args.max_length > limit:
                raise ValueError(f"max_length exceeds {attribute}={limit} for a loaded model")
    load_seconds = time.perf_counter() - began
    preparation_started = time.perf_counter()
    hidden_cache = {}

    def prepare(row):
        observation = row["sender"]
        if observation not in hidden_cache:
            encoded = sender_tokenizer(observation, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > args.max_length:
                raise ValueError("sender observation exceeds max length")
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = sender(**encoded).last_hidden_state.detach()
            mask = encoded.get("attention_mask", torch.ones(hidden.shape[:2], device=args.device))
            hidden_cache[observation] = (hidden, mask)
        hidden, mask = hidden_cache[observation]
        tokens, labels, count = answer_batch(receiver_tokenizer, row["query"], row["answer"], args.max_length, args.device)
        prompt_tokens = tokens[:, :-count]
        maximum_receiver_suffix = max(tokens.shape[1], prompt_tokens.shape[1] + args.max_new_tokens)
        if maximum_receiver_suffix + args.slots > args.max_length:
            raise ValueError("latent prefix plus query/answer or generation budget exceeds max length")
        prefixes = {}
        for name, text in (("text", row["sender"]), ("symbolic", row.get("symbolic"))):
            if text is None:
                continue
            ids = receiver_tokenizer.encode(text + "\n", add_special_tokens=False)
            if len(ids) + maximum_receiver_suffix > args.max_length:
                raise ValueError(f"{name} baseline plus suffix exceeds max length; no facts are truncated")
            prefixes[name] = torch.tensor([ids], device=args.device)
        return {"hidden": hidden, "mask": mask, "tokens": tokens, "prompt_tokens": prompt_tokens,
                "labels": labels, "answer_tokens": count, "prefixes": prefixes}

    cache = {name: [prepare(row) for row in rows] for name, rows in partitions.items()}
    preparation_seconds = time.perf_counter() - preparation_started
    sender_dim = cache["train"][0]["hidden"].shape[-1]
    receiver_dim = receiver.get_input_embeddings().weight.shape[-1]
    bridge = make_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
    constant_bridge = make_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
    constant_bridge.load_state_dict(copy.deepcopy(bridge.state_dict()))
    # Fixed length and values: neither private facts nor source lengths leak.
    constant_hidden = torch.zeros((1, 8, sender_dim), device=args.device)
    constant_mask = torch.ones((1, 8), device=args.device)
    models = {"matched": bridge, "trained_constant": constant_bridge}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate) for name, model in models.items()}
    predictions = []

    def evaluate(name, condition, generate=False):
        active = constant_bridge if condition == "trained_constant" else bridge
        total_nll, answer_tokens, correct, generated_tokens = 0.0, 0, 0, 0
        with torch.no_grad():
            for index, row in enumerate(cache[name]):
                source_index = permutations[name][index] if condition == "permuted" else index
                source = cache[name][source_index]
                if condition == "trained_constant":
                    hidden, mask = constant_hidden, constant_mask
                else:
                    hidden, mask = source["hidden"], source["mask"]
                prefix = row["prefixes"].get(condition)
                loss = answer_loss(receiver, active, hidden, mask, row["tokens"], row["labels"], condition, prefix)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"non-finite {name}/{condition} evaluation loss")
                total_nll += float(loss) * row["answer_tokens"]
                answer_tokens += row["answer_tokens"]
                if generate:
                    answer, count = greedy_answer(receiver, active, hidden, mask, row["prompt_tokens"], receiver_tokenizer,
                                                  condition, prefix, args.max_new_tokens)
                    gold = partitions[name][index]
                    matched = normalize_answer(answer) == normalize_answer(gold["answer"])
                    correct += int(matched)
                    generated_tokens += count
                    predictions.append({"split": name, "id": gold["id"], "world_id": gold.get("world_id"),
                                        "condition": condition, "expected": gold["answer"], "generated": answer,
                                        "correct": matched, "generated_tokens": count,
                                        "source_id": partitions[name][source_index]["id"] if condition == "permuted" else None})
        result = {"answer_token_nll": total_nll / answer_tokens, "answer_tokens": answer_tokens,
                  "examples": len(cache[name]), "distinct_worlds": len({row.get("world_id", row["sender"]) for row in partitions[name]})}
        if generate:
            result.update(generated_exact_match=correct / len(cache[name]), correct=correct,
                          generated_tokens=generated_tokens)
        return result

    best = {name: float("inf") for name in models}
    checkpoints, best_epochs, history = {}, {}, []
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(cache["train"])))
        rng.shuffle(order)
        # Equal architecture, initial weights, order, labels, updates and rate.
        for index in order:
            row = cache["train"][index]
            for name, model in models.items():
                hidden, mask = (row["hidden"], row["mask"]) if name == "matched" else (constant_hidden, constant_mask)
                optimizer = optimizers[name]
                optimizer.zero_grad()
                loss = answer_loss(receiver, model, hidden, mask, row["tokens"], row["labels"], name)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite training loss; inspect model compatibility and data")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        record = {"epoch": epoch}
        for name, model in models.items():
            validation = evaluate("validation", name)["answer_token_nll"]
            record[name + "_validation_answer_token_nll"] = validation
            if validation < best[name]:
                best[name], best_epochs[name] = validation, epoch
                checkpoints[name] = copy.deepcopy(model.state_dict())
        history.append(record)
        print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - training_started
    for name, model in models.items():
        model.load_state_dict(checkpoints[name])
    evaluation_started = time.perf_counter()
    evaluation = {}
    for name in evaluation_names:
        conditions = ["matched", "trained_constant", "zero", "permuted", "no_channel", "text"]
        if all("symbolic" in row["prefixes"] for row in cache[name]):
            conditions.append("symbolic")
        evaluation[name] = {condition: evaluate(name, condition, generate=True) for condition in conditions}
        for condition, metric in evaluation[name].items():
            print(json.dumps({"split": name, "condition": condition, **metric}), flush=True)
    evaluation_seconds = time.perf_counter() - evaluation_started
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), output / "bridge.pt")
    torch.save(constant_bridge.state_dict(), output / "constant_bridge.pt")
    (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    payload_bytes = args.slots * args.bottleneck * next(bridge.parameters()).element_size()
    result = {
        "status": "executed_local_pretrained_model_experiment",
        "seed": args.seed,
        "sender_model": str(Path(args.sender_model).resolve()),
        "receiver_model": str(Path(args.receiver_model).resolve()),
        "model_files": {"sender": fingerprint_directory(args.sender_model), "receiver": fingerprint_directory(args.receiver_model)},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "config": {"slots": args.slots, "bottleneck": args.bottleneck, "epochs": args.epochs,
                   "learning_rate": args.learning_rate, "max_length": args.max_length, "device": args.device,
                   "max_new_tokens": args.max_new_tokens, "threads": args.threads,
                   "constant_hidden_shape": [1, 8, sender_dim], "sender_dim": sender_dim, "receiver_dim": receiver_dim},
        "split_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()},
        "history": history, "selected_epochs": best_epochs,
        "optimizer_steps_per_trained_condition": args.epochs * len(cache["train"]),
        **evaluation,
        "latent_payload_bytes": payload_bytes,
        "bridge_parameter_bytes": sum(p.numel() * p.element_size() for p in bridge.parameters()),
        "constant_bridge_parameter_bytes": sum(p.numel() * p.element_size() for p in constant_bridge.parameters()),
        "cached_sender_hidden_bytes": sum(hidden.numel() * hidden.element_size() for hidden, _ in hidden_cache.values()),
        "timings_seconds": {"model_load": load_seconds, "source_preparation_and_cache": preparation_seconds,
                            "both_bridges_training_and_validation": training_seconds,
                            "all_test_conditions_nll_and_uncached_generation": evaluation_seconds,
                            "entire_runner_excluding_downloads": time.perf_counter() - began},
        "generated_answer_accuracy_measured": True,
        "generation_stopping": "Greedy global max_new_tokens or first newline/EOS; target tokens and target length are not inputs to generation.",
        "generated_exact_match_normalization": "Strip/collapse whitespace and case-fold only; punctuation remains significant.",
        "production_latency_measured": False,
        "limitations": [
            "The private-lookup task remains synthetic; results are not real-world coding or general agent task performance.",
            "One seed is exploratory. Multiple questions per world are correlated; example counts are not independent world counts.",
            "The latent and constant bridges receive equal training budgets; text and symbolic baselines use the frozen receiver without separate task adaptation.",
            "Zero/permuted controls alone do not rule out receiver adaptation; compare against the separately trained constant-information bridge.",
            "The bridge is specific to this model pair; the generalization partition changes task phrasing/order, not model identity.",
            "Teacher-forced NLL and greedy generated exact match are distinct outcomes.",
            "Exact-string/world-id checks cannot detect every form of leakage in user-provided data.",
            "Timing includes uncached reference decoding and is not an optimized production latency comparison.",
            "No real network traffic, API bill, energy use, or amortized training break-even is measured.",
        ],
    }
    if args.model_manifest:
        result["download_manifest"] = json.loads(Path(args.model_manifest).read_text())
    result["setup_accounting"] = {
        "sender_parameter_bytes": sum(p.numel() * p.element_size() for p in sender.parameters()),
        "receiver_parameter_bytes": sum(p.numel() * p.element_size() for p in receiver.parameters()),
        "raw_jsonl_bytes": {name: Path(path).stat().st_size for name, path in paths.items()},
        "bridge_checkpoint_file_bytes": (output / "bridge.pt").stat().st_size,
        "constant_bridge_checkpoint_file_bytes": (output / "constant_bridge.pt").stat().st_size,
        "text_test_payload_utf8_bytes_mean": sum(len(row["sender"].encode("utf-8")) for row in partitions["test"]) / len(partitions["test"]),
        "symbolic_test_payload_utf8_bytes_mean": sum(len(row.get("symbolic", "").encode("utf-8")) for row in partitions["test"]) / len(partitions["test"]) if all("symbolic" in row for row in partitions["test"]) else None,
        "text_test_receiver_prefix_tokens_mean": sum(row["prefixes"]["text"].shape[1] for row in cache["test"]) / len(cache["test"]),
        "latent_receiver_prefix_positions": args.slots,
        "excludes_transient_activations_and_optimizer_storage": True,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sender-model", "receiver-model", "train", "validation", "test"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--generalization-test")
    parser.add_argument("--model-manifest")
    parser.add_argument("--output", default="checkpoints/local-bridge")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.slots, args.bottleneck, args.epochs, args.max_length, args.learning_rate, args.max_new_tokens, args.threads) <= 0:
        parser.error("training sizes and learning rate must be positive")
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Optional dependencies missing: {error}. Install this project's transformers extra.\n")


if __name__ == "__main__":
    main()
