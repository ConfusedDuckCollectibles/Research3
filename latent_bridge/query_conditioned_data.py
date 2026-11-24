"""Fresh paired worlds for the fixed public-query-conditioning follow-up.

Earlier pretrained-study worlds are excluded before partition assignment.
This module only generates synthetic task records; it does not run models.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from .private_counterfactual import (
    COLORS, OBJECTS, VARIANTS, VIEWS, family_worlds, pair_records,
    generate as previous_counterfactual, load_partitions as load_paired_partitions,
)
from .private_lookup import make_tasks, world_identity

TASK = "private_query_conditioned_counterfactual_v1"


def identity_hash(identities):
    """Hash sorted identities as compact UTF-8 JSON, without a final newline."""
    return hashlib.sha256(json.dumps(sorted(set(identities)), separators=(",", ":")).encode("utf-8")).hexdigest()


def prior_world_exclusions():
    previous = previous_counterfactual(seed=20261124, train_families=128,
                                       validation_families=32, test_families=32)
    counterfactual_ids = set()
    for split in ("train", "validation", "test_iid"):
        for row in previous[split]:
            base = tuple(row["original"]["facts"][obj] for obj in OBJECTS)
            counterfactual_ids.update(world_identity(world) for world in family_worlds(base))
    pilot = make_tasks(seed=20261123, train_worlds=64, validation_worlds=16,
                       test_worlds=16, shift_worlds=16)
    pilot_ids = {row["world_id"] for rows in pilot.values() for row in rows}
    identities = counterfactual_ids | pilot_ids
    return {
        "prior_counterfactual_worlds": len(counterfactual_ids),
        "prior_pilot_worlds": len(pilot_ids),
        "prior_study_overlap_worlds": len(counterfactual_ids & pilot_ids),
        "excluded_worlds": len(identities),
        "excluded_world_identities": sorted(identities),
        "excluded_world_identities_sha256": identity_hash(identities),
        "identity_hash_serialization": "UTF-8 JSON of sorted unique world SHA256 strings, separators=(comma,colon), no trailing newline",
        "prior_generators": {
            "counterfactual": {"seed": 20261124, "train_families": 128, "validation_families": 32, "test_families": 32},
            "pilot": {"seed": 20261123, "train_worlds": 64, "validation_worlds": 16, "test_worlds": 16, "shift_worlds": 16},
        },
    }


def generate(seed=20261125, train_families=128, validation_families=32, test_families=32):
    counts = {"train": train_families, "validation": validation_families, "test": test_families}
    if any(type(count) is not int or count < 2 for count in counts.values()):
        raise ValueError("each partition must contain at least two families")
    worlds = list(itertools.product(COLORS, repeat=len(OBJECTS)))
    random.Random(seed).shuffle(worlds)
    reserved = set(prior_world_exclusions()["excluded_world_identities"])
    selected = []
    for world in worlds:
        members = {world_identity(member) for member in family_worlds(world)}
        if members & reserved:
            continue
        selected.append(world)
        reserved.update(members)
        if len(selected) == sum(counts.values()):
            break
    if len(selected) != sum(counts.values()):
        raise ValueError("not enough fresh disjoint families after reserving both prior pretrained studies")
    partitions, cursor = {}, 0
    for name, count in counts.items():
        selected_partition = selected[cursor:cursor + count]
        cursor += count
        if name == "test":
            for view in VIEWS:
                partitions["test_" + view] = [row for world in selected_partition for row in pair_records(world, view)]
        else:
            partitions[name] = [row for world in selected_partition for row in pair_records(world)]
    return partitions


def reserved_identities(partitions):
    result = {}
    for name in ("train", "validation", "test_iid"):
        identities = set()
        for row in partitions[name]:
            base = tuple(row["original"]["facts"][obj] for obj in OBJECTS)
            identities.update(world_identity(world) for world in family_worlds(base))
        result[name] = sorted(identities)
    return result


def load_partitions(directory):
    directory = Path(directory)
    parts = load_paired_partitions(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("task") != TASK:
        raise ValueError("not a fresh query-conditioned dataset")
    exclusions = prior_world_exclusions()
    if manifest.get("prior_world_exclusions") != exclusions:
        raise ValueError("prior-world exclusion manifest does not match the fixed studies")
    reserved = reserved_identities(parts)
    blocked = set(exclusions["excluded_world_identities"])
    if any(blocked.intersection(identities) for identities in reserved.values()):
        raise ValueError("new family intersects a prior pretrained-study world")
    for name, identities in reserved.items():
        metadata = manifest.get("reserved_worlds_by_partition", {}).get(name, {})
        if metadata.get("identities") != identities or metadata.get("sha256") != identity_hash(identities):
            raise ValueError("reserved new-world identities do not match the manifest")
        if metadata.get("worlds") != len(identities):
            raise ValueError("reserved new-world count does not match the manifest")
    return parts


def write_dataset(directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parts = generate(**kwargs)
    metadata = {}
    for name, rows in parts.items():
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        (directory / (name + ".jsonl")).write_text(content)
        metadata[name] = {
            "families": len({row["family_id"] for row in rows}), "query_pairs": len(rows),
            "original_training_records_if_train": len(rows) if name == "train" else None,
            "original_plus_counterfactual_records": len(rows) * 2, "nuisance_records": len(rows),
            "utf8_bytes": len(content.encode("utf-8")), "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "answer_counts_by_member": {variant: {color: sum(row[variant]["answer"] == color for row in rows) for color in COLORS} for variant in VARIANTS},
        }
    reserved = reserved_identities(parts)
    all_reserved = {identity for identities in reserved.values() for identity in identities}
    manifest = {
        "task": TASK, "generation_seed": kwargs.get("seed", 20261125),
        "objects": list(OBJECTS), "candidate_colors": list(COLORS), "partitions": metadata,
        "prior_world_exclusions": prior_world_exclusions(),
        "reserved_worlds_by_partition": {name: {"worlds": len(identities), "identities": identities, "sha256": identity_hash(identities)} for name, identities in reserved.items()},
        "new_reserved_worlds": len(all_reserved), "new_reserved_world_identities_sha256": identity_hash(all_reserved),
        "generation": "Enumerate all 8^4 worlds, shuffle once with Python random.Random(seed); reject any base-plus-four-edits family intersecting either prior pretrained study or any accepted family; assign accepted families in train/validation/test order.",
        "training": "Original member only, all four public object queries per training family.",
        "validation": "Each arm selects its lowest mean restricted-eight-choice CE on original and targeted-counterfactual members.",
        "test_views": "Four paired presentations of the same test families: original, source-only shift, question-only shift, joint shift.",
        "counterfactual": "Only queried color advances one step in the fixed color cycle; other facts and the public query are unchanged.",
        "nuisance": "Only the next unqueried object's color advances; queried answer remains unchanged.",
        "model_input_boundary": "Only source strings, public query strings, and condition-specific complete text/symbolic strings are encoder inputs. Fact dictionaries, IDs, labels and query_key metadata are not model features.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_partitions(directory)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local_data/query_conditioned"))
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--train-families", type=int, default=128)
    parser.add_argument("--validation-families", type=int, default=32)
    parser.add_argument("--test-families", type=int, default=32)
    args = parser.parse_args()
    manifest = write_dataset(args.output, seed=args.seed, train_families=args.train_families,
                             validation_families=args.validation_families, test_families=args.test_families)
    print(json.dumps({"task": manifest["task"], "partitions": manifest["partitions"],
                      "prior_excluded_worlds": manifest["prior_world_exclusions"]["excluded_worlds"],
                      "new_reserved_worlds": manifest["new_reserved_worlds"]}, indent=2))


if __name__ == "__main__":
    main()
