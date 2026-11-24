"""Independent stdlib-only tests for the frozen eight-object binding data.

These tests do not fit or load models. Actual pinned-tokenizer ID-multiset
checks belong to the separate tokenizer preflight, not these grammar tests.

Golden cohort/edit fixtures were independently constructed in JavaScript from
the frozen lexicographic population and integer seeds, using a separate
CPython-compatible MT19937, shuffle, and sample implementation. They were not
obtained by calling binding_data, inspecting fitted predictions, or fitting
models. The test's definition-level Python oracle checks the fixed fixtures.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

from latent_bridge import binding_data as bd

OBJECTS = ("apple", "book", "cup", "lamp", "key", "plate", "shoe", "ball")
COLORS = ("red", "blue", "green", "yellow", "black", "white", "purple", "orange")
MEMBERS = ("O", "T", "N", "R")
SCHEMA = "eight-object-bijective-binding-v1"
SPLITS = {"train": 64, "validation": 32, "test": 64}
QUESTIONS = (
    "The color of the apple is", "The color of the book is",
    "The color of the cup is", "The color of the lamp is",
    "The color of the key is", "The color of the plate is",
    "The color of the shoe is", "The color of the ball is",
)

# Independently obtained from the declared lexicographic population using a
# separately written MT19937/shuffle implementation, not binding_data helpers.
# The same construction examined 340 candidates before accepting family 160.
# These fixed data fixtures are not model outcomes.
GOLDEN_BASES = tuple(tuple(int(c) for c in code) for code in """
45103726 36157204 17230645 62734501 01652347 05374162 36724015 05637142
17460235 73460125 67120345 14065723 35614207 06457132 12075346 56372041
20431657 27345061 74126350 35476120 14203765 51427360 50174623 70312654
37041265 75036214 30162547 43507612 20547316 73125640 40716352 64517023
10356724 25460731 74160253 01763254 14257063 74315062 76042153 52603417
41560237 60534127 41635720 23415076 27561403 47206351 40216735 15026734
63270415 42301576 32150467 52710643 52746130 21057643 76314502 10265437
52436710 45210673 31264750 34027615 31602457 75613402 13764502 51607324
26014357 02145637 12467350 62341705 14756032 24057136 64730152 51276340
04765312 71643250 12604753 76205143 01274635 60142357 45763021 15647230
06541372 20135746 07142365 57264310 45702613 07524613 03174526 46520173
40562713 36025471 73061452 61532740 13472560 21463507 15370426 50317642
23501764 63154027 67143052 52367014 64712035 47213605 04312567 57026431
34752106 45172063 37214056 64235170 46237015 56012437 62453107 25601347
20576134 20156473 72106534 51260743 14625073 04576321 65237104 72564031
26304751 56304217 06173452 45136072 01542763 70465213 01375642 35720164
43027156 03751426 12476035 32506741 37640521 32657041 34516270 30247651
01432576 02743516 64307215 61245307 76051234 67425130 20673145 53607241
75341620 20657431 40123657 07346512 41273560 26354170 74526103 42630157
43150762 10725634 20714536 71246035 16502374 40152376 20765341 52147063
""".split())
GOLDEN_METADATA = {
  "0": {
    "order": [
      3,
      1,
      2,
      4,
      6,
      5,
      0,
      7
    ],
    "reorder": [
      5,
      3,
      7,
      1,
      0,
      4,
      2,
      6
    ],
    "donors": [
      2,
      7,
      3,
      0,
      6,
      1,
      5,
      4
    ],
    "nuisance": [
      [
        1,
        2
      ],
      [
        2,
        4
      ],
      [
        5,
        7
      ],
      [
        0,
        6
      ],
      [
        3,
        5
      ],
      [
        0,
        4
      ],
      [
        4,
        7
      ],
      [
        0,
        6
      ]
    ]
  },
  "63": {
    "order": [
      5,
      2,
      1,
      7,
      4,
      6,
      3,
      0
    ],
    "reorder": [
      7,
      0,
      3,
      4,
      1,
      2,
      5,
      6
    ],
    "donors": [
      3,
      5,
      4,
      7,
      2,
      6,
      1,
      0
    ],
    "nuisance": [
      [
        1,
        6
      ],
      [
        2,
        5
      ],
      [
        1,
        6
      ],
      [
        4,
        6
      ],
      [
        3,
        7
      ],
      [
        1,
        2
      ],
      [
        0,
        7
      ],
      [
        1,
        3
      ]
    ]
  },
  "64": {
    "order": [
      7,
      5,
      3,
      2,
      6,
      4,
      0,
      1
    ],
    "reorder": [
      4,
      6,
      3,
      1,
      5,
      0,
      7,
      2
    ],
    "donors": [
      6,
      4,
      7,
      0,
      3,
      2,
      1,
      5
    ],
    "nuisance": [
      [
        3,
        7
      ],
      [
        3,
        4
      ],
      [
        3,
        6
      ],
      [
        5,
        6
      ],
      [
        2,
        6
      ],
      [
        0,
        3
      ],
      [
        2,
        7
      ],
      [
        2,
        4
      ]
    ]
  },
  "95": {
    "order": [
      2,
      4,
      3,
      0,
      1,
      7,
      5,
      6
    ],
    "reorder": [
      4,
      1,
      3,
      2,
      6,
      7,
      0,
      5
    ],
    "donors": [
      5,
      7,
      4,
      0,
      6,
      1,
      3,
      2
    ],
    "nuisance": [
      [
        1,
        6
      ],
      [
        5,
        6
      ],
      [
        3,
        7
      ],
      [
        0,
        4
      ],
      [
        5,
        6
      ],
      [
        1,
        4
      ],
      [
        1,
        4
      ],
      [
        1,
        4
      ]
    ]
  },
  "96": {
    "order": [
      6,
      4,
      7,
      3,
      5,
      0,
      2,
      1
    ],
    "reorder": [
      6,
      3,
      7,
      2,
      4,
      1,
      5,
      0
    ],
    "donors": [
      7,
      6,
      1,
      4,
      3,
      0,
      2,
      5
    ],
    "nuisance": [
      [
        4,
        7
      ],
      [
        2,
        7
      ],
      [
        1,
        4
      ],
      [
        4,
        5
      ],
      [
        2,
        5
      ],
      [
        6,
        7
      ],
      [
        1,
        7
      ],
      [
        1,
        3
      ]
    ]
  },
  "159": {
    "order": [
      0,
      7,
      3,
      1,
      6,
      2,
      4,
      5
    ],
    "reorder": [
      4,
      1,
      7,
      5,
      2,
      6,
      3,
      0
    ],
    "donors": [
      6,
      3,
      4,
      1,
      0,
      2,
      7,
      5
    ],
    "nuisance": [
      [
        3,
        7
      ],
      [
        0,
        7
      ],
      [
        0,
        3
      ],
      [
        2,
        7
      ],
      [
        1,
        6
      ],
      [
        4,
        6
      ],
      [
        2,
        7
      ],
      [
        1,
        2
      ]
    ]
  }
}


def neighbors(world):
    """Definition-level oracle, independent of reserved_worlds/swap_world."""
    result = {tuple(world)}
    for left in range(8):
        for right in range(left + 1, 8):
            candidate = list(world)
            candidate[left], candidate[right] = candidate[right], candidate[left]
            result.add(tuple(candidate))
    return result


def exchanged(world, left, right):
    return tuple(world[right] if index == left else world[left] if index == right else value
                 for index, value in enumerate(world))


def source_text(world, order, symbolic=False):
    if symbolic:
        return "".join(OBJECTS[i] + "=" + COLORS[world[i]] + ";\n" for i in order)
    return "".join("The " + OBJECTS[i] + " is " + COLORS[world[i]] + ".\n" for i in order)


def expected_id(world):
    literal = {"schema": SCHEMA, "objects": list(OBJECTS), "colors": list(COLORS), "world": list(world)}
    return hashlib.sha256(json.dumps(literal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replace_member_world(member, world, query):
    member.update(world=list(world), world_id=expected_id(world),
                  source=source_text(world, member["order"]),
                  symbolic=source_text(world, member["order"], True),
                  answer=COLORS[world[query]])


class ExactGrammarAndCodecTests(unittest.TestCase):
    def test_public_orders_seeds_schema_and_questions_are_fixed(self):
        self.assertEqual(tuple(bd.OBJECTS), OBJECTS)
        self.assertEqual(tuple(bd.COLORS), COLORS)
        self.assertEqual(tuple(bd.MEMBERS), MEMBERS)
        self.assertEqual(bd.SCHEMA, SCHEMA)
        self.assertEqual((bd.DATA_SEED, bd.ORDER_SEED, bd.EDIT_SEED), (20261127, 20261128, 20261129))
        self.assertEqual(bd.SPLIT_COUNTS, SPLITS)
        for query, question in enumerate(QUESTIONS):
            self.assertEqual(bd.public_question(query), question)
            self.assertEqual(bd.parse_question(question), query)
        for invalid in (-1, 8, True, 1.0, "apple", None):
            with self.assertRaises((ValueError, TypeError)):
                bd.public_question(invalid)
        for invalid in ("The color of the pear is", QUESTIONS[0] + " red",
                        QUESTIONS[0] + "\n", QUESTIONS[0] + " ", QUESTIONS[0].lower(), None):
            with self.assertRaises((ValueError, TypeError)):
                bd.parse_question(invalid)

    def test_literal_text_and_symbolic_formats_have_exact_lf_terminators(self):
        world, order = tuple(range(8)), tuple(range(8))
        english = ("The apple is red.\nThe book is blue.\nThe cup is green.\nThe lamp is yellow.\n"
                   "The key is black.\nThe plate is white.\nThe shoe is purple.\nThe ball is orange.\n")
        symbolic = ("apple=red;\nbook=blue;\ncup=green;\nlamp=yellow;\n"
                    "key=black;\nplate=white;\nshoe=purple;\nball=orange;\n")
        self.assertEqual(bd.render_source(world, order), english)
        self.assertEqual(bd.render_source(world, order, symbolic=True), symbolic)
        self.assertEqual(bd.parse_source(english), world)
        self.assertEqual(bd.parse_source(symbolic, symbolic=True), world)
        for use_symbolic in (False, True):
            reversed_text = source_text(world, tuple(reversed(order)), use_symbolic)
            self.assertEqual(bd.render_source(world, tuple(reversed(order)), use_symbolic), reversed_text)
            self.assertEqual(bd.parse_source(reversed_text, use_symbolic), world)

    def test_parser_rejects_wrong_facts_noncanonical_separators_and_extra_text(self):
        for symbolic in (False, True):
            text = source_text(tuple(range(8)), tuple(range(8)), symbolic)
            lines = text.split("\n")[:-1]
            invalid = [
                text[:-1], text + "\n", " " + text, text + QUESTIONS[0],
                text.replace("\n", "\r\n"), text.replace("\n", "\u2028", 1),
                text.replace("\n", "\x0b", 1), text.replace("\n", "\x85", 1),
                text.replace("apple", "pear", 1), text.replace("red", "cyan", 1),
                text.replace("blue", "red", 1), "\n".join([lines[0], lines[0], *lines[2:]]) + "\n",
            ]
            for malformed in invalid:
                with self.subTest(symbolic=symbolic, malformed=repr(malformed)):
                    with self.assertRaises(ValueError):
                        bd.parse_source(malformed, symbolic=symbolic)
            with self.assertRaises(ValueError):
                bd.parse_source(text, symbolic=not symbolic)
            with self.assertRaises(ValueError):
                bd.parse_source(text.encode(), symbolic=symbolic)

    def test_world_and_order_validation_rejects_nonbijective_or_noninteger_inputs(self):
        identity = tuple(range(8))
        invalid = [(0,) * 8, tuple(range(7)), tuple(range(9)), (-1, 1, 2, 3, 4, 5, 6, 7),
                   (8, 1, 2, 3, 4, 5, 6, 7), (False, 1, 2, 3, 4, 5, 6, 7),
                   (0.0, 1, 2, 3, 4, 5, 6, 7), "01234567", None]
        for world in invalid:
            with self.assertRaises((ValueError, TypeError)):
                bd.encode_rank(world)
            with self.assertRaises((ValueError, TypeError)):
                bd.render_source(world, identity)
            with self.assertRaises((ValueError, TypeError)):
                bd.render_source(identity, world)
        for indices in ((0, 0), (-1, 2), (0, 8), (True, 2), (0.0, 2)):
            with self.assertRaises(ValueError):
                bd.swap_world(identity, *indices)

    def test_reservation_is_exactly_one_swap_ball_not_transitive_closure(self):
        identity = tuple(range(8))
        # A different mathematical characterization avoids asking the helper
        # to construct both the actual and expected reservations.
        expected = {world for world in itertools.permutations(range(8))
                    if sum(a != b for a, b in zip(world, identity)) in (0, 2)}
        self.assertEqual(len(expected), 29)
        self.assertEqual(set(bd.reserved_worlds(identity)), expected)
        self.assertNotIn((1, 0, 3, 2, 4, 5, 6, 7), expected)  # Two distinct swaps.
        for left, right in itertools.combinations(range(8), 2):
            self.assertEqual(bd.swap_world(identity, left, right), exchanged(identity, left, right))
            self.assertEqual(bd.swap_world(bd.swap_world(identity, left, right), left, right), identity)

    def test_canonical_identity_contains_schema_and_complete_binding_ignoring_order(self):
        identity = tuple(range(8))
        swapped = exchanged(identity, 0, 1)
        self.assertEqual(bd.canonical_id(identity), expected_id(identity))
        self.assertEqual(bd.canonical_id(swapped), expected_id(swapped))
        self.assertNotEqual(bd.canonical_id(identity), bd.canonical_id(swapped))
        for order in (tuple(range(8)), tuple(reversed(range(8))), (2, 4, 6, 0, 1, 3, 5, 7)):
            decoded = bd.parse_source(source_text(identity, order))
            self.assertEqual(bd.canonical_id(decoded), expected_id(identity))

    def test_every_world_has_its_exact_lexicographic_two_byte_code(self):
        # itertools enumeration is an independent rank oracle; it does not use
        # the implementation's factorial/Lehmer arithmetic.
        count = 0
        for rank, world in enumerate(itertools.permutations(range(8))):
            expected = rank.to_bytes(2, "big")
            encoded = bd.encode_rank(world)
            if encoded != expected or bd.decode_rank(expected) != world:
                self.fail(f"Incorrect lexicographic code at rank {rank}: {world}")
            self.assertIsInstance(encoded, bytes)
            self.assertEqual(len(encoded), 2)
            count += 1
        self.assertEqual(count, 40320)
        self.assertEqual(bd.encode_rank(tuple(range(8))), b"\x00\x00")
        self.assertEqual(bd.encode_rank(tuple(reversed(range(8)))), b"\x9d\x7f")
        self.assertEqual(bd.encode_rank((1, 0, 2, 3, 4, 5, 6, 7)), b"\x13\xb0")

    def test_all_unused_sixteen_bit_codes_and_wrong_packet_shapes_are_rejected(self):
        for rank in range(40320, 65536):
            try:
                bd.decode_rank(rank.to_bytes(2, "big"))
            except ValueError:
                continue
            self.fail(f"Out-of-range rank {rank} was accepted")
        for packet in (b"", b"\x00", b"\x00\x00\x00", b"\x00\x00\x00\x00",
                       "\x00\x00", bytearray(2), memoryview(b"\x00\x00"), 0, None):
            with self.assertRaises(ValueError):
                bd.decode_rank(packet)

    def test_one_complete_world_code_answers_all_eight_public_questions(self):
        world = (7, 0, 5, 1, 6, 2, 4, 3)
        for symbolic in (False, True):
            text = source_text(world, (4, 6, 1, 5, 3, 0, 7, 2), symbolic)
            packet = bd.encode_rank(bd.parse_source(text, symbolic))
            decoded = bd.decode_rank(packet)
            for query, question in enumerate(QUESTIONS):
                self.assertEqual(COLORS[decoded[bd.parse_question(question)]], COLORS[world[query]])
            self.assertEqual(len(packet), 2)


class FrozenDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partitions, cls.manifest = bd.generate_dataset()
        cls.by_family = {}
        for split, rows in cls.partitions.items():
            for row in rows:
                cls.by_family.setdefault(row["family"], []).append(row)
        cls.manifest_families = cls.manifest["families"]

    def test_all_160_golden_bases_and_340_candidate_feasibility(self):
        self.assertEqual(len(GOLDEN_BASES), 160)
        self.assertEqual(tuple(tuple(row["base_world"]) for row in self.manifest_families), GOLDEN_BASES)
        # Reproduce only the prescribed shuffled candidate order independently,
        # then use the definition-level neighbor oracle and hard-coded cohort.
        population = list(itertools.permutations(range(8)))
        random.Random(20261127).shuffle(population)
        self.assertEqual(population.index(GOLDEN_BASES[-1]) + 1, 340)
        occupied, accepted = set(), []
        for base in population[:340]:
            reservation = neighbors(base)
            if occupied.isdisjoint(reservation):
                accepted.append(base)
                occupied.update(reservation)
        self.assertEqual(tuple(accepted), GOLDEN_BASES)
        self.assertEqual(len(occupied), 4640)

    def test_manifest_constants_counts_and_exact_split_boundaries(self):
        self.assertEqual(tuple(self.partitions), ("train", "validation", "test"))
        self.assertEqual((self.manifest["schema"], self.manifest["data_seed"],
                          self.manifest["order_seed"], self.manifest["edit_seed"]),
                         (SCHEMA, 20261127, 20261128, 20261129))
        self.assertEqual(self.manifest["objects"], list(OBJECTS))
        self.assertEqual(self.manifest["colors"], list(COLORS))
        self.assertEqual(self.manifest["members"], list(MEMBERS))
        self.assertEqual(self.manifest["split_family_counts"], SPLITS)
        self.assertEqual(self.manifest["reserved_world_count"], 4640)
        expected_counts = {"train": (64, 512, 2048), "validation": (32, 256, 1024), "test": (64, 512, 2048)}
        for split, (families, quartets, records) in expected_counts.items():
            rows = self.partitions[split]
            self.assertEqual(len({row["family"] for row in rows}), families)
            self.assertEqual(len(rows), quartets)
            self.assertEqual(sum(len(row["members"]) for row in rows), records)
        for number, family in enumerate(self.manifest_families):
            split = "train" if number < 64 else "validation" if number < 96 else "test"
            self.assertEqual(family["partition"], split)
            self.assertEqual(tuple(family["base_world"]), GOLDEN_BASES[number])
            rows = self.by_family[family["family"]]
            self.assertEqual([row["query_index"] for row in rows], list(range(8)))
            self.assertTrue(all(row in self.partitions[split] for row in rows))

    def test_all_reservations_are_pairwise_disjoint_including_unscored_neighbors(self):
        occupied, identity_set, partitions = set(), set(), defaultdict(set)
        total_unscored = 0
        for number, family in enumerate(self.manifest_families):
            expected = neighbors(GOLDEN_BASES[number])
            actual = {tuple(world) for world in family["reserved_worlds"]}
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 29)
            self.assertTrue(occupied.isdisjoint(actual))
            occupied.update(actual)
            partitions[family["partition"]].update(actual)
            expected_ids = [expected_id(world) for world in sorted(expected)]
            self.assertEqual(family["reserved_world_ids"], expected_ids)
            self.assertEqual(family["base_world_id"], expected_id(GOLDEN_BASES[number]))
            identity_set.update(expected_ids)
            scored = {tuple(member["world"]) for row in self.by_family[family["family"]] for member in row["members"]}
            self.assertTrue(scored <= expected)
            total_unscored += len(expected - scored)
        self.assertEqual(len(occupied), 4640)
        self.assertEqual(len(identity_set), 4640)
        for a, b in itertools.combinations(SPLITS, 2):
            self.assertTrue(partitions[a].isdisjoint(partitions[b]))
        self.assertEqual({name: len(values) for name, values in partitions.items()},
                         {"train": 1856, "validation": 928, "test": 1856})
        self.assertEqual(total_unscored, 2460)  # Independently counted from the frozen edit draws.
        expected_hash = hashlib.sha256(json.dumps(sorted(identity_set), separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(self.manifest["reserved_id_set_sha256"], expected_hash)

    def test_independent_rendering_and_edit_rng_boundary_anchors(self):
        for number, expected in GOLDEN_METADATA.items():
            family = self.manifest_families[int(number)]
            self.assertEqual(family["order"], expected["order"])
            self.assertEqual(family["reorder"], expected["reorder"])
            self.assertEqual(family["donors"], expected["donors"])
            self.assertEqual(family["nuisance_pairs"], expected["nuisance"])

    def test_every_quartet_has_exact_binding_edits_answers_orders_and_derangement(self):
        donor_patterns, nuisance_patterns = set(), set()
        for family in self.manifest_families:
            base = tuple(family["base_world"])
            rows = self.by_family[family["family"]]
            donors = [row["partner"] for row in rows]
            self.assertEqual(sorted(donors), list(range(8)))
            self.assertTrue(all(query != donor for query, donor in enumerate(donors)))
            self.assertEqual(donors, family["donors"])
            donor_patterns.add(tuple(donors))
            self.assertNotEqual(family["order"], family["reorder"])
            self.assertEqual(sorted(family["order"]), list(range(8)))
            self.assertEqual(sorted(family["reorder"]), list(range(8)))
            for query, row in enumerate(rows):
                self.assertEqual(row["query"], QUESTIONS[query])
                self.assertEqual(tuple(member["variant"] for member in row["members"]), MEMBERS)
                left, right = row["nuisance_pair"]
                self.assertTrue(0 <= left < right < 8)
                self.assertNotIn(query, (left, right))
                nuisance_patterns.add((query, left, right))
                expected_worlds = (base, exchanged(base, query, row["partner"]),
                                   exchanged(base, left, right), base)
                for index, (member, world) in enumerate(zip(row["members"], expected_worlds)):
                    self.assertEqual(tuple(member["world"]), world)
                    self.assertEqual(sorted(world), list(range(8)))
                    self.assertEqual(member["world_id"], expected_id(world))
                    self.assertEqual(member["answer"], COLORS[world[query]])
                    order = family["reorder"] if index == 3 else family["order"]
                    self.assertEqual(member["order"], order)
                    self.assertEqual(member["source"], source_text(world, order))
                    self.assertEqual(member["symbolic"], source_text(world, order, True))
                    self.assertEqual(bd.parse_source(member["source"]), world)
                    self.assertEqual(bd.parse_source(member["symbolic"], True), world)
                original, target, nuisance, reorder = row["members"]
                self.assertNotEqual(original["answer"], target["answer"])
                self.assertEqual(original["answer"], nuisance["answer"])
                self.assertEqual(original["answer"], reorder["answer"])
                self.assertEqual(sum(a != b for a, b in zip(base, target["world"])), 2)
                self.assertEqual(sum(a != b for a, b in zip(base, nuisance["world"])), 2)
            for index in range(4):
                self.assertEqual(Counter(row["members"][index]["answer"] for row in rows), Counter(COLORS))
        self.assertGreater(len(donor_patterns), 1)
        self.assertGreater(len(nuisance_patterns), 8)

    def test_source_identity_is_query_and_role_independent_for_repeated_world_views(self):
        identities, appearances = {}, defaultdict(set)
        for rows in self.partitions.values():
            for row in rows:
                for member in row["members"]:
                    identity = (tuple(member["world"]), tuple(member["order"]))
                    strings = (member["source"], member["symbolic"])
                    if identity in identities:
                        self.assertEqual(strings, identities[identity])
                    identities[identity] = strings
                    appearances[identity].add((row["query_index"], member["variant"]))
        self.assertTrue(any(len({query for query, _ in entries}) == 8 for entries in appearances.values()))
        self.assertEqual(sum(len({role for _, role in entries}) > 1 for entries in appearances.values()), 321)
        self.assertEqual(len(identities), 2340)  # Independent source/view count for the frozen cohort.
        self.assertEqual(len(identities), self.manifest["audit"]["unique_source_order_identities"])

    def test_recorded_answer_frequencies_are_exact_without_assuming_uniform_object_priors(self):
        expected = {}
        for split, rows in self.partitions.items():
            frequency = {str(query): {member: [0] * 8 for member in MEMBERS} for query in range(8)}
            for row in rows:
                for member in row["members"]:
                    frequency[str(row["query_index"])][member["variant"]][COLORS.index(member["answer"])] += 1
            expected[split] = frequency
            for variant in MEMBERS:
                for query in range(8):
                    self.assertEqual(sum(frequency[str(query)][variant]), SPLITS[split])
                self.assertEqual([sum(frequency[str(q)][variant][color] for q in range(8)) for color in range(8)],
                                 [SPLITS[split]] * 8)
        self.assertEqual(self.manifest["audit"]["answer_frequencies_by_query_member"], expected)
        self.assertEqual(bd.validate_partitions(self.partitions)["answer_frequencies_by_query_member"], expected)

    def test_generation_is_deterministic_and_validator_requires_all_three_partitions(self):
        partitions, manifest = bd.generate_dataset()
        self.assertEqual(partitions, self.partitions)
        self.assertEqual(manifest, self.manifest)
        with self.assertRaises(ValueError):
            bd.validate_partitions({name: self.partitions[name] for name in ("train", "validation")})

    def test_validator_rejects_overlap_via_an_unscored_reserved_neighbor(self):
        poisoned = copy.deepcopy(self.partitions)
        family = self.manifest_families[0]
        scored = {tuple(member["world"]) for row in self.by_family[family["family"]] for member in row["members"]}
        unscored = sorted(neighbors(GOLDEN_BASES[0]) - scored)
        self.assertTrue(unscored)
        replacement = unscored[0]
        # Keep the entire validation family internally consistent except for
        # its forbidden reservation overlap with a training family.
        for row in poisoned["validation"][:8]:
            query = row["query_index"]
            worlds = (replacement, exchanged(replacement, query, row["partner"]),
                      exchanged(replacement, *row["nuisance_pair"]), replacement)
            for member, world in zip(row["members"], worlds):
                replace_member_world(member, world, query)
        with self.assertRaisesRegex(ValueError, "reservation overlap"):
            bd.validate_partitions(poisoned)

    def test_validator_rejects_query_dependent_orders_even_with_valid_rendered_text(self):
        poisoned = copy.deepcopy(self.partitions)
        row = poisoned["train"][1]
        original = row["members"][0]["order"]
        reorder = row["members"][3]["order"]
        changed = next(original[shift:] + original[:shift] for shift in range(1, 8)
                       if original[shift:] + original[:shift] != reorder)
        for member in row["members"][:3]:
            member["order"] = changed[:]
            member["source"] = source_text(member["world"], changed)
            member["symbolic"] = source_text(member["world"], changed, True)
        with self.assertRaisesRegex(ValueError, "order leaks query"):
            bd.validate_partitions(poisoned)

    def test_validator_rejects_balanced_target_rule_violation_missing_queries_and_bad_nuisance(self):
        poisoned = copy.deepcopy(self.partitions)
        row = poisoned["train"][0]
        base = tuple(row["members"][0]["world"])
        other = next(donor for donor in self.manifest_families[0]["donors"]
                     if donor not in (row["query_index"], row["partner"]))
        row["partner"] = other
        replace_member_world(row["members"][1], exchanged(base, row["query_index"], other), row["query_index"])
        with self.assertRaisesRegex(ValueError, "balanced|donor"):
            bd.validate_partitions(poisoned)
        poisoned = copy.deepcopy(self.partitions)
        poisoned["train"][1] = copy.deepcopy(poisoned["train"][0])
        with self.assertRaises(ValueError):
            bd.validate_partitions(poisoned)
        poisoned = copy.deepcopy(self.partitions)
        poisoned["train"][0]["nuisance_pair"] = [0, 1]
        with self.assertRaises(ValueError):
            bd.validate_partitions(poisoned)

    def test_serialized_files_are_hashed_complete_deterministic_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest = bd.write_dataset(first)
            other_manifest = bd.write_dataset(second)
            self.assertEqual(manifest, other_manifest)
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
                data = (Path(first) / name).read_bytes()
                self.assertEqual(data, (Path(second) / name).read_bytes())
                if name != "manifest.json":
                    self.assertEqual(manifest["files"][name],
                                     {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            loaded = bd.load_partitions(first, include_test=True)
            self.assertEqual(loaded, self.partitions)
            before = (Path(first) / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                bd.write_dataset(first)
            self.assertEqual((Path(first) / "manifest.json").read_bytes(), before)

    def test_default_loader_does_not_open_fresh_test_records_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            bd.write_dataset(directory)
            test = Path(directory) / "test.jsonl"
            test.unlink()
            fitting = bd.load_partitions(directory)
            self.assertEqual(set(fitting), {"train", "validation"})
            self.assertEqual(fitting, {name: self.partitions[name] for name in fitting})
            with self.assertRaises((OSError, ValueError)):
                bd.load_partitions(directory, include_test=True)
            train = Path(directory) / "train.jsonl"
            train.write_bytes(train.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                bd.load_partitions(directory)


if __name__ == "__main__":
    unittest.main()
