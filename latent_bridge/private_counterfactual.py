"""Private-fact counterfactual families for a preregistered closed-choice study."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from .private_lookup import COLORS, OBJECTS, world_identity

VIEWS = ("iid", "sender", "query", "joint")
VARIANTS = ("original", "counterfactual", "nuisance")


def changed(world, key):
    result = list(world)
    result[key] = COLORS[(COLORS.index(world[key]) + 1) % len(COLORS)]
    return tuple(result)


def family_worlds(world):
    return (tuple(world), *(changed(world, key) for key in range(len(OBJECTS))))


def source_text(world, shifted=False):
    facts = list(zip(OBJECTS, world))
    if shifted:
        return "In this record, " + "; ".join(f"the color listed for the {obj} is {color}" for obj, color in reversed(facts)) + "."
    return " ".join(f"The {obj} is {color}." for obj, color in facts)


def query_text(key, shifted=False):
    return f"According to the record, the {OBJECTS[key]} has color" if shifted else f"The color of the {OBJECTS[key]} is"


def symbolic_text(world):
    return ";".join(f"{obj}={color}" for obj, color in zip(OBJECTS, world))


def parse_symbolic(text):
    parts = text.split(";")
    result = {}
    for part in parts:
        if part.count("=") != 1:
            raise ValueError("invalid symbolic record")
        key, value = part.split("=")
        if key in result or key not in OBJECTS or value not in COLORS:
            raise ValueError("invalid symbolic key/value")
        result[key] = value
    if set(result) != set(OBJECTS):
        raise ValueError("incomplete symbolic record")
    return result


def pair_records(world, view="iid"):
    if view not in VIEWS:
        raise ValueError("unknown rendering view")
    family_id = "family-" + world_identity(world)
    rows = []
    for key, obj in enumerate(OBJECTS):
        row = {"pair_id": family_id + ":" + obj, "family_id": family_id,
               "view": view, "query_key": obj, "query": query_text(key, view in ("query", "joint"))}
        for name, member in (("original", world), ("counterfactual", changed(world, key)),
                             ("nuisance", changed(world, (key + 1) % len(OBJECTS)))):
            row[name] = {"world_id": world_identity(member), "facts": dict(zip(OBJECTS, member)),
                         "source": source_text(member, view in ("sender", "joint")),
                         "symbolic": symbolic_text(member), "answer": member[key]}
        rows.append(row)
    return rows


def generate(seed=20261124, train_families=128, validation_families=32, test_families=32):
    counts = {"train": train_families, "validation": validation_families, "test": test_families}
    if any(type(count) is not int or count < 2 for count in counts.values()):
        raise ValueError("each partition must contain at least two families")
    worlds = list(itertools.product(COLORS, repeat=len(OBJECTS)))
    random.Random(seed).shuffle(worlds)
    selected, reserved = [], set()
    for world in worlds:
        members = set(family_worlds(world))
        if members & reserved:
            continue
        selected.append(world)
        reserved |= members
        if len(selected) == sum(counts.values()):
            break
    if len(selected) != sum(counts.values()):
        raise ValueError("not enough disjoint counterfactual families for requested counts")
    partitions, cursor = {}, 0
    for name, count in counts.items():
        partition_worlds = selected[cursor:cursor + count]
        cursor += count
        if name == "test":
            for view in VIEWS:
                partitions["test_" + view] = [row for world in partition_worlds for row in pair_records(world, view)]
        else:
            partitions[name] = [row for world in partition_worlds for row in pair_records(world)]
    return partitions


def validate_record(row):
    if not isinstance(row, dict) or row.get("view") not in VIEWS or row.get("query_key") not in OBJECTS:
        raise ValueError("invalid pair identity or view")
    key = OBJECTS.index(row["query_key"])
    view = row["view"]
    if row.get("query") != query_text(key, view in ("query", "joint")):
        raise ValueError("query does not match its declared view/key")
    members = {}
    for name in VARIANTS:
        member = row.get(name, {})
        facts = member.get("facts", {})
        if set(facts) != set(OBJECTS) or any(color not in COLORS for color in facts.values()):
            raise ValueError("invalid private world")
        world = tuple(facts[obj] for obj in OBJECTS)
        members[name] = world
        if member.get("world_id") != world_identity(world):
            raise ValueError("world hash does not match facts")
        if member.get("source") != source_text(world, view in ("sender", "joint")):
            raise ValueError("source does not match facts/view")
        if member.get("symbolic") != symbolic_text(world) or parse_symbolic(member["symbolic"]) != facts:
            raise ValueError("symbolic record does not match facts")
        if member.get("answer") != world[key]:
            raise ValueError("answer does not match queried fact")
    original = members["original"]
    if row.get("family_id") != "family-" + world_identity(original):
        raise ValueError("family identity does not match original world")
    if row.get("pair_id") != row["family_id"] + ":" + row["query_key"]:
        raise ValueError("pair identity does not match family/query")
    if members["counterfactual"] != changed(original, key):
        raise ValueError("target counterfactual must change only the queried fact")
    if members["nuisance"] != changed(original, (key + 1) % len(OBJECTS)):
        raise ValueError("nuisance edit must change only the next unqueried fact")


def load_partitions(directory):
    directory = Path(directory)
    names = ("train", "validation", *("test_" + view for view in VIEWS))
    parts = {}
    for name in names:
        rows = [json.loads(line) for line in (directory / (name + ".jsonl")).read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError("empty partition")
        seen_pairs = set()
        for row in rows:
            validate_record(row)
            if row["pair_id"] in seen_pairs:
                raise ValueError("duplicate pair in a rendering")
            seen_pairs.add(row["pair_id"])
        families = {row["family_id"] for row in rows}
        if any({r["query_key"] for r in rows if r["family_id"] == family} != set(OBJECTS) for family in families):
            raise ValueError("every family must have exactly four query pairs")
        expected_view = name.removeprefix("test_") if name.startswith("test_") else "iid"
        if any(row["view"] != expected_view for row in rows):
            raise ValueError("unexpected rendering in partition")
        parts[name] = rows
    owners = {}
    for name in ("train", "validation", "test_iid"):
        for row in parts[name]:
            for member in family_worlds(tuple(row["original"]["facts"][obj] for obj in OBJECTS)):
                identity = world_identity(member)
                owner = (name, row["family_id"])
                if identity in owners and owners[identity] != owner:
                    raise ValueError("world reused across distinct families or partitions")
                owners[identity] = owner
    base = {row["pair_id"]: row for row in parts["test_iid"]}
    for view in VIEWS[1:]:
        rows = parts["test_" + view]
        if {row["pair_id"] for row in rows} != set(base):
            raise ValueError("test renderings must share the same paired families")
        for row in rows:
            if any(row[name]["world_id"] != base[row["pair_id"]][name]["world_id"] for name in VARIANTS):
                raise ValueError("test renderings changed facts instead of only presentation")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in names:
        expected = manifest.get("partitions", {}).get(name, {}).get("sha256")
        actual = hashlib.sha256((directory / (name + ".jsonl")).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("dataset file does not match its manifest hash")
    return parts


def write_dataset(directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    partitions = generate(**kwargs)
    metadata = {}
    for name, rows in partitions.items():
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        (directory / (name + ".jsonl")).write_text(content)
        metadata[name] = {"families": len({row["family_id"] for row in rows}), "query_pairs": len(rows),
                          "original_training_records_if_train": len(rows) if name == "train" else None,
                          "original_plus_counterfactual_records": len(rows) * 2,
                          "nuisance_records": len(rows), "sha256": hashlib.sha256(content.encode()).hexdigest(),
                          "answer_counts_by_member": {variant: {color: sum(row[variant]["answer"] == color for row in rows) for color in COLORS} for variant in VARIANTS}}
    manifest = {"task": "private_single_fact_counterfactual_v1", "generation_seed": kwargs.get("seed", 20261124),
                "objects": list(OBJECTS), "candidate_colors": list(COLORS), "partitions": metadata,
                "family_rule": "Reserve original world and its four one-key next-color edits; reject overlap with every previously chosen family, including across partitions.",
                "training": "Use original member only, all four object queries per training family.",
                "validation": "Select each trained arm checkpoint by mean restricted-eight-choice CE over original and targeted-counterfactual members.",
                "test_views": "IID, source-only, query-only, joint. Same world families and answers across views; these are paired presentations, not independent datasets.",
                "counterfactual": "Only the queried object's color advances one step in the fixed color cycle; the query and all other facts stay fixed.",
                "nuisance": "Only the next unqueried object's color advances one step; queried answer stays fixed.",
                "oracle": "Parse the complete symbolic facts and select query_key; task-solvability check only, not an LM baseline or performance claim."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_partitions(directory)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local_data/counterfactual"))
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--train-families", type=int, default=128)
    parser.add_argument("--validation-families", type=int, default=32)
    parser.add_argument("--test-families", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output, seed=args.seed, train_families=args.train_families,
                                  validation_families=args.validation_families, test_families=args.test_families), indent=2))


if __name__ == "__main__":
    main()
