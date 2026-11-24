"""Reuse the declared fitting data and reserve new readout-test families."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from .private_counterfactual import COLORS, OBJECTS, VARIANTS, VIEWS, family_worlds, pair_records, load_partitions as load_paired
from .private_lookup import world_identity
from .query_conditioned_data import generate as query_data, identity_hash, prior_world_exclusions, reserved_identities, load_partitions as load_original

TASK = "frozen_message_readout_diagnostic_v1"
EXCLUSION_HASH = "a34f2c6f1b1bf4602e9daea8c4feaacdd83272ebb98372b032a90cefa52499c1"


def exclusions():
    values = set(prior_world_exclusions()["excluded_world_identities"])
    for identities in reserved_identities(query_data()).values():
        values.update(identities)
    if len(values) != 2002 or identity_hash(values) != EXCLUSION_HASH:
        raise ValueError("historical world generators no longer match the frozen exclusion set")
    return sorted(values)


def fresh_families(seed=20261126, count=32):
    if type(count) is not int or count < 2:
        raise ValueError("at least two fresh families are required")
    worlds = list(itertools.product(COLORS, repeat=len(OBJECTS)))
    random.Random(seed).shuffle(worlds)
    blocked, chosen = set(exclusions()), []
    for world in worlds:
        members = {world_identity(value) for value in family_worlds(world)}
        if members & blocked:
            continue
        chosen.append(world)
        blocked.update(members)
        if len(chosen) == count:
            return chosen
    raise ValueError("fixed greedy generator cannot supply the requested fresh family count")


def write_dataset(original_data_dir, output, seed=20261126, test_families=32):
    original = load_original(original_data_dir)
    reference = query_data()
    for name in ("train", "validation"):
        if original[name] != reference[name]:
            raise ValueError("probe fitting data must match the exact previous study")
    chosen = fresh_families(seed, test_families)
    parts = {name: original[name] for name in ("train", "validation")}
    parts.update({"test_" + view: [row for world in chosen for row in pair_records(world, view)] for view in VIEWS})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, rows in parts.items():
        blob = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode()
        (output / (name + ".jsonl")).write_bytes(blob)
        metadata[name] = {"families": len({row["family_id"] for row in rows}), "query_pairs": len(rows),
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            "answer_counts_by_member": {variant: {color: sum(row[variant]["answer"] == color for row in rows) for color in COLORS} for variant in VARIANTS}}
    fresh_ids = sorted({world_identity(member) for world in chosen for member in family_worlds(world)})
    manifest = {"task": TASK, "generation_seed": seed, "partitions": metadata,
        "excluded_worlds": 2002, "excluded_world_identities": exclusions(), "excluded_world_identities_sha256": EXCLUSION_HASH,
        "fresh_reserved_worlds": len(fresh_ids), "fresh_reserved_identities": fresh_ids,
        "fresh_reserved_identities_sha256": identity_hash(fresh_ids),
        "fitting_data": "Reuse only the previous query-conditioned study's 128 training and 32 validation families; no previous test family is fitted.",
        "generation": "Fixed greedy generator at seed 20261126 accepts 53 families after exclusions; first 32 used. This is not a combinatorial packing maximum.",
        "views": "Same fresh families, paired original/source/query/joint renderings, no independent-view sample claim."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_partitions(output)
    return manifest


def load_partitions(directory):
    directory = Path(directory)
    parts = load_paired(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("task") != TASK:
        raise ValueError("wrong readout dataset task")
    if manifest.get("excluded_world_identities") != exclusions() or manifest.get("excluded_world_identities_sha256") != EXCLUSION_HASH:
        raise ValueError("readout exclusion provenance mismatch")
    if manifest.get("excluded_worlds") != 2002:
        raise ValueError("readout exclusion count mismatch")
    original = query_data()
    if any(parts[name] != original[name] for name in ("train", "validation")):
        raise ValueError("readout fitting data changed from the declared previous study")
    ids = reserved_identities(parts)["test_iid"]
    if set(ids) & set(exclusions()):
        raise ValueError("fresh readout test family overlaps a previous pretrained-study world")
    if ids != manifest.get("fresh_reserved_identities") or identity_hash(ids) != manifest.get("fresh_reserved_identities_sha256"):
        raise ValueError("fresh reserved identities mismatch")
    if len(ids) != manifest.get("fresh_reserved_worlds"):
        raise ValueError("fresh reserved count mismatch")
    expected = [row for world in fresh_families(manifest["generation_seed"], len(parts["test_iid"]) // 4) for row in pair_records(world)]
    if parts["test_iid"] != expected:
        raise ValueError("fresh families do not follow the fixed greedy generation rule")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--test-families", type=int, default=32)
    args = parser.parse_args()
    result = write_dataset(args.original_data_dir, args.output, args.seed, args.test_families)
    print(json.dumps({key: result[key] for key in ("task", "excluded_worlds", "fresh_reserved_worlds", "partitions")}, indent=2))


if __name__ == "__main__":
    main()
