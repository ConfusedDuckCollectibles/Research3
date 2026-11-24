"""Fixed eight-object bijective binding data and a two-byte exact codec."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re

OBJECTS = ("apple", "book", "cup", "lamp", "key", "plate", "shoe", "ball")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "purple", "orange")
MEMBERS = ("O", "T", "N", "R")
SCHEMA = "eight-object-bijective-binding-v1"
DATA_SEED, ORDER_SEED, EDIT_SEED = 20261127, 20261128, 20261129
SPLIT_COUNTS = {"train": 64, "validation": 32, "test": 64}


def checked_world(world):
    world = tuple(world)
    if len(world) != 8 or any(type(value) is not int for value in world) or sorted(world) != list(range(8)):
        raise ValueError("world must be a permutation of eight color indices")
    return world


def swap_world(world, i, j):
    world = list(checked_world(world))
    if type(i) is not int or type(j) is not int or not 0 <= i < 8 or not 0 <= j < 8 or i == j:
        raise ValueError("swap requires distinct valid object indices")
    world[i], world[j] = world[j], world[i]
    return tuple(world)


def reserved_worlds(world):
    world = checked_world(world)
    return frozenset([world] + [swap_world(world, i, j) for i, j in itertools.combinations(range(8), 2)])


def canonical_id(world):
    value = {"schema": SCHEMA, "objects": OBJECTS, "colors": COLORS,
             "world": checked_world(world)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_order(order):
    order = tuple(order)
    if len(order) != 8 or any(type(i) is not int for i in order) or sorted(order) != list(range(8)):
        raise ValueError("sentence order must contain each object once")
    return order


def render_source(world, order, symbolic=False):
    world, order = checked_world(world), checked_order(order)
    if symbolic:
        return "".join(f"{OBJECTS[i]}={COLORS[world[i]]};\n" for i in order)
    return "".join(f"The {OBJECTS[i]} is {COLORS[world[i]]}.\n" for i in order)


def parse_source(text, symbolic=False):
    if not isinstance(text, str) or not text.endswith("\n"):
        raise ValueError("source requires a final newline")
    lines = text.split("\n")[:-1]
    if len(lines) != 8:
        raise ValueError("source requires exactly eight facts")
    pattern = r"([a-z]+)=([a-z]+);" if symbolic else r"The ([a-z]+) is ([a-z]+)\."
    values = {}
    for line in lines:
        match = re.fullmatch(pattern, line)
        if match is None or match[1] not in OBJECTS or match[2] not in COLORS or match[1] in values:
            raise ValueError("invalid or duplicate source fact")
        values[match[1]] = COLORS.index(match[2])
    if set(values) != set(OBJECTS):
        raise ValueError("source omits an object")
    return checked_world(tuple(values[object_name] for object_name in OBJECTS))


def public_question(query_index):
    if type(query_index) is not int or not 0 <= query_index < 8:
        raise ValueError("invalid public object index")
    return f"The color of the {OBJECTS[query_index]} is"


def parse_question(text):
    for i in range(8):
        if text == public_question(i):
            return i
    raise ValueError("unsupported public question grammar")


def encode_rank(world):
    world = checked_world(world)
    remaining, rank = list(range(8)), 0
    for i, color in enumerate(world):
        rank += remaining.index(color) * math.factorial(7 - i)
        remaining.remove(color)
    return rank.to_bytes(2, "big")


def decode_rank(packet):
    if not isinstance(packet, bytes) or len(packet) != 2:
        raise ValueError("permutation code requires exactly two bytes")
    rank = int.from_bytes(packet, "big")
    if rank >= math.factorial(8):
        raise ValueError("permutation rank is out of range")
    remaining, world = list(range(8)), []
    for i in range(8):
        index, rank = divmod(rank, math.factorial(7 - i))
        world.append(remaining.pop(index))
    return tuple(world)


def file_metadata(path):
    value = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def generate_dataset():
    bases = list(itertools.permutations(range(8)))
    random.Random(DATA_SEED).shuffle(bases)
    accepted, occupied = [], set()
    for base in bases:
        reservation = reserved_worlds(base)
        if occupied.isdisjoint(reservation):
            accepted.append(base)
            occupied.update(reservation)
            if len(accepted) == 160:
                break
    if len(accepted) != 160 or len(occupied) != 4640:
        raise ValueError("fixed greedy construction cannot provide the frozen cohort")
    order_rng, edit_rng = random.Random(ORDER_SEED), random.Random(EDIT_SEED)
    partitions = {name: [] for name in SPLIT_COUNTS}
    families = []
    for number, base in enumerate(accepted):
        split = "train" if number < 64 else "validation" if number < 96 else "test"
        family = f"binding-family-{number:03d}"
        order = list(range(8))
        order_rng.shuffle(order)
        while True:
            reorder = list(range(8))
            order_rng.shuffle(reorder)
            if reorder != order:
                break
        while True:
            donors = list(range(8))
            edit_rng.shuffle(donors)
            if all(i != donor for i, donor in enumerate(donors)):
                break
        nuisance_pairs = [
            sorted(edit_rng.sample([j for j in range(8) if j != i], 2))
            for i in range(8)
        ]
        reservations = sorted(reserved_worlds(base))
        families.append({
            "family": family, "partition": split, "base_world": list(base),
            "base_world_id": canonical_id(base), "order": order, "reorder": reorder,
            "donors": donors, "nuisance_pairs": nuisance_pairs,
            "reserved_worlds": [list(world) for world in reservations],
            "reserved_world_ids": [canonical_id(world) for world in reservations],
        })
        for query_index in range(8):
            worlds = [base, swap_world(base, query_index, donors[query_index]),
                      swap_world(base, *nuisance_pairs[query_index]), base]
            members = []
            for variant, world in zip(MEMBERS, worlds):
                member_order = reorder if variant == "R" else order
                members.append({
                    "variant": variant, "world": list(world),
                    "world_id": canonical_id(world), "order": list(member_order),
                    "source": render_source(world, member_order),
                    "symbolic": render_source(world, member_order, symbolic=True),
                    "answer": COLORS[world[query_index]],
                })
            partitions[split].append({
                "family": family, "query_index": query_index,
                "query": public_question(query_index),
                "partner": donors[query_index],
                "nuisance_pair": nuisance_pairs[query_index], "members": members,
            })
    audit = validate_partitions(partitions)
    manifest = {
        "schema": SCHEMA, "data_seed": DATA_SEED, "order_seed": ORDER_SEED,
        "edit_seed": EDIT_SEED, "objects": list(OBJECTS), "colors": list(COLORS),
        "members": list(MEMBERS), "split_family_counts": SPLIT_COUNTS,
        "reserved_world_count": len(occupied),
        "reserved_id_set_sha256": hashlib.sha256(
            json.dumps(sorted(canonical_id(world) for world in occupied),
                       separators=(",", ":")).encode()).hexdigest(),
        "families": families, "audit": audit,
        "identity_serialization": "SHA256 of compact sorted-key JSON {schema,objects,colors,world}; order ignored",
        "prior_study_relationship": "new eight-object bijection domain; no trained bridge reuse",
    }
    return partitions, manifest


def validate_partitions(partitions):
    if set(partitions) != set(SPLIT_COUNTS):
        raise ValueError("all three partitions are required for complete validation")
    global_reservations, global_families = set(), set()
    frequencies, source_ids = {}, {}
    for split, expected_families in SPLIT_COUNTS.items():
        quartets = partitions[split]
        if len(quartets) != expected_families * 8:
            raise ValueError("incorrect fixed quartet count")
        grouped = {}
        frequency = {str(q): {variant: [0] * 8 for variant in MEMBERS} for q in range(8)}
        for quartet in quartets:
            family, qi = quartet["family"], quartet["query_index"]
            if quartet["query"] != public_question(qi):
                raise ValueError("public question differs from its declared index")
            if [member["variant"] for member in quartet["members"]] != list(MEMBERS):
                raise ValueError("quartet member ordering differs from O,T,N,R")
            grouped.setdefault(family, []).append(quartet)
            original, target, nuisance, reordered = quartet["members"]
            base = checked_world(original["world"])
            pair = quartet["nuisance_pair"]
            if len(pair) != 2 or pair != sorted(pair) or qi in pair:
                raise ValueError("invalid nuisance pair")
            expected_worlds = [base, swap_world(base, qi, quartet["partner"]),
                               swap_world(base, *pair), base]
            standard = checked_order(original["order"])
            if checked_order(reordered["order"]) == standard:
                raise ValueError("reorder is unchanged")
            for member, world in zip(quartet["members"], expected_worlds):
                if checked_world(member["world"]) != world or member["world_id"] != canonical_id(world):
                    raise ValueError("member world/edit identity mismatch")
                if member["variant"] != "R" and checked_order(member["order"]) != standard:
                    raise ValueError("fact edit also changed order")
                for field, symbolic in (("source", False), ("symbolic", True)):
                    rendered = render_source(world, member["order"], symbolic)
                    if member[field] != rendered or parse_source(rendered, symbolic) != world:
                        raise ValueError("source rendering or round trip mismatch")
                if decode_rank(encode_rank(world)) != world:
                    raise ValueError("rank-code round trip failed")
                if member["answer"] != COLORS[world[qi]]:
                    raise ValueError("answer does not follow private world")
                frequency[str(qi)][member["variant"]][COLORS.index(member["answer"])] += 1
                key = canonical_id(world) + ":" + ",".join(map(str, member["order"]))
                digest = hashlib.sha256(member["source"].encode()).hexdigest()
                if key in source_ids and source_ids[key] != digest:
                    raise ValueError("same world/order has query-dependent source text")
                source_ids[key] = digest
            if original["answer"] == target["answer"] or any(
                member["answer"] != original["answer"] for member in (nuisance, reordered)
            ):
                raise ValueError("quartet target/invariance relationship failed")
        if len(grouped) != expected_families:
            raise ValueError("incorrect family count")
        for family, rows in grouped.items():
            if family in global_families or sorted(row["query_index"] for row in rows) != list(range(8)):
                raise ValueError("duplicate family or query")
            global_families.add(family)
            rows = sorted(rows, key=lambda row: row["query_index"])
            base = checked_world(rows[0]["members"][0]["world"])
            if any(checked_world(row["members"][0]["world"]) != base for row in rows):
                raise ValueError("family has different base worlds")
            for role_index in range(4):
                if any(row["members"][role_index]["order"] != rows[0]["members"][role_index]["order"] for row in rows):
                    raise ValueError("rendering order leaks query")
                if Counter(row["members"][role_index]["answer"] for row in rows) != Counter(COLORS):
                    raise ValueError("member answers are not balanced within family")
            if sorted(row["partner"] for row in rows) != list(range(8)):
                raise ValueError("target donors are not a permutation")
            reservation = reserved_worlds(base)
            if len(reservation) != 29 or not global_reservations.isdisjoint(reservation):
                raise ValueError("family reservation overlap")
            global_reservations.update(reservation)
        frequencies[split] = frequency
    if len(global_reservations) != 4640:
        raise ValueError("incorrect complete reservation count")
    return {"family_counts": dict(SPLIT_COUNTS),
            "quartet_counts": {key: len(value) for key, value in partitions.items()},
            "record_counts": {key: 4 * len(value) for key, value in partitions.items()},
            "reserved_world_count": len(global_reservations),
            "answer_frequencies_by_query_member": frequencies,
            "unique_source_order_identities": len(source_ids),
            "source_identity_sha256": hashlib.sha256(
                json.dumps(source_ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def write_dataset(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")):
        raise ValueError("refusing to overwrite an existing binding dataset")
    partitions, manifest = generate_dataset()
    manifest["files"] = {}
    for split, rows in partitions.items():
        path = output / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        manifest["files"][path.name] = file_metadata(path)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_partitions(data_dir, include_test=False):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or (
        manifest.get("data_seed"), manifest.get("order_seed"), manifest.get("edit_seed")
    ) != (DATA_SEED, ORDER_SEED, EDIT_SEED):
        raise ValueError("data schema or seeds differ from the frozen design")
    result = {}
    for split in ("train", "validation", "test") if include_test else ("train", "validation"):
        path = data_dir / (split + ".jsonl")
        if file_metadata(path) != manifest["files"][path.name]:
            raise ValueError("dataset byte hash mismatch")
        result[split] = [json.loads(line) for line in path.read_text().splitlines()]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = write_dataset(args.output)
    print(json.dumps({"status": "generated", "reserved_worlds": manifest["reserved_world_count"],
                      "audit": manifest["audit"]["quartet_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
