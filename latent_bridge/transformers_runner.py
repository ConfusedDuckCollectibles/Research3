"""Optional local-model experiment. Implemented, but not run in the initial release.

No model download is allowed. Base models stay frozen; gradients flow through
the receiver into a small learned continuous prefix. Metrics are answer-token
negative log likelihood, not generated-answer accuracy or inference speed.
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
    parts, ids, observations = {}, {}, {}
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
    observations = [row["sender"] for row in rows]
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


def answer_loss(receiver, bridge, hidden, mask, tokens, labels, condition="matched", text_prefix=None):
    """The causal model shifts labels internally; prefix and prompt stay masked."""
    import torch

    embeddings = receiver.get_input_embeddings()(tokens)
    if condition == "text":
        prefix = receiver.get_input_embeddings()(text_prefix)
    elif condition == "no_channel":
        prefix = embeddings[:, :0]
    else:
        message = bridge.encode(hidden, mask)
        if condition == "zero":
            message = torch.zeros_like(message)
        prefix = bridge.decode(message)
    inputs = torch.cat((prefix.to(embeddings.dtype), embeddings), dim=1)
    ignored = torch.full((1, prefix.shape[1]), -100, dtype=torch.long, device=labels.device)
    combined_labels = torch.cat((ignored, labels), dim=1)
    attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    # receiver.eval() disables dropout, but do NOT use no_grad while training.
    return receiver(inputs_embeds=inputs, attention_mask=attention_mask, labels=combined_labels, use_cache=False).loss


def run(args):
    partitions = read_partitions({"train": args.train, "validation": args.validation, "test": args.test})
    permuted_indices = different_observation_indices(partitions["test"])
    for path in (args.sender_model, args.receiver_model):
        if not Path(path).is_dir():
            raise ValueError("model paths must be existing local directories; downloads are disabled")
    import torch
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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

    def prepare(row):
        encoded = sender_tokenizer(row["sender"], return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > args.max_length:
            raise ValueError("sender observation exceeds max length")
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = sender(**encoded).last_hidden_state.detach()
        tokens, labels, count = answer_batch(receiver_tokenizer, row["query"], row["answer"], args.max_length, args.device)
        if tokens.shape[1] + args.slots > args.max_length:
            raise ValueError("latent prefix + query + answer exceeds max length")
        text_ids = receiver_tokenizer.encode(row["sender"] + "\n", add_special_tokens=False)
        if len(text_ids) + tokens.shape[1] > args.max_length:
            raise ValueError("text baseline exceeds max length; use the same complete facts for all conditions")
        text_prefix = torch.tensor([text_ids], device=args.device)
        return {"hidden": hidden, "mask": encoded.get("attention_mask", torch.ones(hidden.shape[:2], device=args.device)), "tokens": tokens, "labels": labels, "answer_tokens": count, "text_prefix": text_prefix}

    # This cache is charged explicitly; it trades memory for avoiding repeated
    # sender passes during bridge training. It is not a production memory claim.
    cache = {name: [prepare(row) for row in rows] for name, rows in partitions.items()}
    sender_dim = cache["train"][0]["hidden"].shape[-1]
    receiver_dim = receiver.get_input_embeddings().weight.shape[-1]
    bridge = make_bridge(sender_dim, receiver_dim, args.bottleneck, args.slots).to(args.device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.learning_rate)

    def evaluate(name, condition):
        total, tokens = 0.0, 0
        with torch.no_grad():
            for index, row in enumerate(cache[name]):
                source = cache[name][permuted_indices[index]] if condition == "permuted" else row
                loss = answer_loss(receiver, bridge, source["hidden"], source["mask"], row["tokens"], row["labels"], condition, row["text_prefix"])
                total += float(loss) * row["answer_tokens"]
                tokens += row["answer_tokens"]
        return {"answer_token_nll": total / tokens, "answer_tokens": tokens, "examples": len(cache[name])}

    if len(cache["test"]) < 2:
        raise ValueError("test set must contain at least two observations for permuted-message control")
    best, best_state, history = float("inf"), None, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(cache["train"])))
        rng.shuffle(order)
        for index in order:
            row = cache["train"][index]
            optimizer.zero_grad()
            loss = answer_loss(receiver, bridge, row["hidden"], row["mask"], row["tokens"], row["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss; inspect model compatibility and data")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            optimizer.step()
        validation = evaluate("validation", "matched")["answer_token_nll"]
        history.append({"epoch": epoch, "validation_answer_token_nll": validation})
        if validation < best:
            best = validation
            best_state = copy.deepcopy(bridge.state_dict())
        print(json.dumps(history[-1]), flush=True)
    bridge.load_state_dict(best_state)
    metrics = {condition: evaluate("test", condition) for condition in ("matched", "zero", "permuted", "no_channel", "text")}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), output / "bridge.pt")
    payload_bytes = args.slots * args.bottleneck * next(bridge.parameters()).element_size()
    result = {"status": "executed_local_pretrained_model_experiment", "seed": args.seed, "sender_model": str(Path(args.sender_model).resolve()), "receiver_model": str(Path(args.receiver_model).resolve()), "versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "config": {"slots": args.slots, "bottleneck": args.bottleneck, "epochs": args.epochs, "learning_rate": args.learning_rate, "max_length": args.max_length, "device": args.device}, "split_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in {"train": args.train, "validation": args.validation, "test": args.test}.items()}, "history": history, "test": metrics, "latent_payload_bytes": payload_bytes, "bridge_parameter_bytes": sum(p.numel() * p.element_size() for p in bridge.parameters()), "cached_sender_hidden_bytes": sum(row["hidden"].numel() * row["hidden"].element_size() for rows in cache.values() for row in rows), "training_and_evaluation_seconds_excluding_model_load_and_cache": time.perf_counter() - start, "generated_answer_accuracy_measured": False, "production_latency_measured": False, "limitations": ["Teacher-forced answer-token NLL does not establish generated-answer accuracy.", "The text baseline supplies complete sender observations to the same frozen receiver; it has no separately trained text adapter.", "Base models are frozen. This bridge is specific to the selected model pair and task distribution.", "Data checks cannot detect paraphrase leakage or answer leakage in query fields; curate and independently review inputs.", "No protocol network framing or actual network transfer is measured.", "Model weights and tokenizer files must be retained externally for reproducibility; data split hashes alone do not identify local model revisions."]}
    result["setup_accounting"] = {
        "sender_parameter_bytes": sum(p.numel() * p.element_size() for p in sender.parameters()),
        "receiver_parameter_bytes": sum(p.numel() * p.element_size() for p in receiver.parameters()),
        "raw_jsonl_bytes": {name: Path(path).stat().st_size for name, path in {"train": args.train, "validation": args.validation, "test": args.test}.items()},
        "checkpoint_file_bytes": (output / "bridge.pt").stat().st_size,
        "text_test_payload_utf8_bytes_mean": sum(len(row["sender"].encode("utf-8")) for row in partitions["test"]) / len(partitions["test"]),
        "text_test_receiver_prefix_tokens_mean": sum(row["text_prefix"].shape[1] for row in cache["test"]) / len(cache["test"]),
        "latent_receiver_prefix_positions": args.slots,
        "excludes_transient_activations_and_optimizer_storage": True,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sender-model", "receiver-model", "train", "validation", "test"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--output", default="checkpoints/local-bridge")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if min(args.slots, args.bottleneck, args.epochs, args.max_length, args.learning_rate) <= 0:
        parser.error("training sizes and learning rate must be positive")
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Optional dependencies missing: {error}. Install this project's transformers extra.\n")


if __name__ == "__main__":
    main()
