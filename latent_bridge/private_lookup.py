"""Deterministic private-fact tasks for a small pretrained-model experiment.

This is synthetic evaluation data, not a real-world agent benchmark. Complete
worlds, rather than individual questions, are assigned to disjoint partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

OBJECTS = ("apple", "book", "cup", "lamp")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "purple", "orange")


def world_identity(world):
    canonical = json.dumps(dict(zip(OBJECTS, world)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_tasks(seed=20261123, train_worlds=64, validation_worlds=16, test_worlds=16, shift_worlds=16):
    counts = {"train": train_worlds, "validation": validation_worlds, "test": test_worlds, "generalization": shift_worlds}
    if any(not isinstance(count, int) or count < 2 for count in counts.values()):
        raise ValueError("each partition must contain at least two worlds")
    worlds = list(itertools.product(COLORS, repeat=len(OBJECTS)))
    if sum(counts.values()) > len(worlds):
        raise ValueError("requested more distinct worlds than the task supports")
    random.Random(seed).shuffle(worlds)
    partitions, cursor = {}, 0
    for name, count in counts.items():
        rows = []
        for index, world in enumerate(worlds[cursor:cursor + count]):
            facts = list(zip(OBJECTS, world))
            if name == "generalization":
                sender = "In this record, " + "; ".join(f"the color listed for the {obj} is {color}" for obj, color in reversed(facts)) + "."
            else:
                sender = " ".join(f"The {obj} is {color}." for obj, color in facts)
            symbolic = ";".join(f"{obj}={color}" for obj, color in facts)
            for obj, color in facts:
                query = f"According to the record, the {obj} has color" if name == "generalization" else f"The color of the {obj} is"
                rows.append({"id": f"{name}-{index:03d}-{obj}", "world_id": world_identity(world), "sender": sender, "symbolic": symbolic, "query": query, "answer": " " + color + "\n"})
        partitions[name] = rows
        cursor += count
    return partitions


def write_tasks(output, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    partitions = make_tasks(**kwargs)
    metadata = {}
    for name, rows in partitions.items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        (output / f"{name}.jsonl").write_text(payload)
        counts = {color: sum(row["answer"].strip() == color for row in rows) for color in COLORS}
        metadata[name] = {"examples": len(rows), "distinct_worlds": len({row["world_id"] for row in rows}), "answer_counts": counts, "sha256": hashlib.sha256(payload.encode()).hexdigest(), "utf8_bytes": len(payload.encode())}
    manifest = {
        "task": "private_object_color_lookup_v1",
        "status": "generated_synthetic_task_data_not_model_results",
        "seed": kwargs.get("seed", 20261123), "objects": list(OBJECTS), "colors": list(COLORS),
        "partitions": metadata,
        "generation": "Enumerate all 8^4 worlds, shuffle once with Python random.Random(seed), assign complete worlds to partitions in train/validation/test/generalization order, ask all four object queries.",
        "generalization_shift": "Unseen worlds, reversed source fact order, new source phrasing, and a changed query template; colors and objects remain familiar.",
        "source_visibility": "Only sender sees object-color assignments; receiver sees object query and the condition-specific message.",
        "stopping_rule": "Answers end with newline; generate at most a global fixed token budget or stop on newline/EOS, never use gold answer length.",
        "accuracy_definition": "Normalize by stripping/collapsing whitespace and case-folding, then require entire generated answer equal entire gold answer. Punctuation is not removed.",
        "limits": ["A deliberately small lookup task, not coding, open-ended reasoning, or universal transfer.", "Four queries in one world are statistically dependent; report world count as well as example count.", "The symbolic baseline has access to the same complete facts, serialized compactly.", "No-info baselines can exploit label priors; report actual answer distributions."]
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local_data/private_lookup"))
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--train-worlds", type=int, default=64)
    parser.add_argument("--validation-worlds", type=int, default=16)
    parser.add_argument("--test-worlds", type=int, default=16)
    parser.add_argument("--shift-worlds", type=int, default=16)
    args = parser.parse_args()
    manifest = write_tasks(args.output, seed=args.seed, train_worlds=args.train_worlds, validation_worlds=args.validation_worlds, test_worlds=args.test_worlds, shift_worlds=args.shift_worlds)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
