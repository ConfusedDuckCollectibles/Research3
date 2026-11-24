import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from latent_bridge.private_counterfactual import COLORS, OBJECTS, VARIANTS, VIEWS, changed, family_worlds, generate, load_partitions, pair_records, parse_symbolic, validate_record, write_dataset
from latent_bridge.counterfactual_runner import EVALUATED_ARMS, candidate_ids, competence_gate, flatten, oracle_accuracy, paired_family_scores, source_derangement, summarize


class CounterfactualDataTests(unittest.TestCase):
    def test_family_worlds_never_overlap_across_training_or_test(self):
        parts = generate(train_families=4, validation_families=3, test_families=3)
        owners = {}
        for name in ("train", "validation", "test_iid"):
            for row in parts[name]:
                validate_record(row)
                original = tuple(row["original"]["facts"][obj] for obj in OBJECTS)
                for world in family_worlds(original):
                    owner = (name, row["family_id"])
                    if world in owners:
                        self.assertEqual(owners[world], owner)
                    owners[world] = owner
        self.assertEqual(len(owners), (4 + 3 + 3) * 5)
        self.assertEqual(parts, generate(train_families=4, validation_families=3, test_families=3))

    def test_targeted_and_nuisance_edits_preserve_required_facts(self):
        row = pair_records(("red", "blue", "green", "yellow"))[0]
        validate_record(row)
        original = row["original"]["facts"]
        self.assertEqual([key for key in OBJECTS if original[key] != row["counterfactual"]["facts"][key]], ["apple"])
        self.assertEqual([key for key in OBJECTS if original[key] != row["nuisance"]["facts"][key]], ["book"])
        self.assertEqual(row["original"]["answer"], row["nuisance"]["answer"])
        self.assertNotEqual(row["original"]["answer"], row["counterfactual"]["answer"])
        corrupted = copy.deepcopy(row)
        double_edit = changed(changed(("red", "blue", "green", "yellow"), 0), 1)
        corrupted["counterfactual"] = pair_records(double_edit)[0]["original"]
        with self.assertRaisesRegex(ValueError, "only the queried fact"):
            validate_record(corrupted)

    def test_views_are_paired_and_source_family_shuffle_is_coherent(self):
        parts = generate(train_families=2, validation_families=2, test_families=3)
        base = parts["test_iid"]
        first_mapping = None
        for view in VIEWS:
            rows = parts["test_" + view]
            self.assertEqual([r["pair_id"] for r in rows], [r["pair_id"] for r in base])
            for row, original in zip(rows, base):
                for variant in VARIANTS:
                    self.assertEqual(row[variant]["facts"], original[variant]["facts"])
                self.assertEqual(row["query"] == original["query"], view in ("iid", "sender"))
                self.assertEqual(row["original"]["source"] == original["original"]["source"], view in ("iid", "query"))
            cases = flatten(rows)
            indices = source_derangement(cases)
            self.assertEqual(sorted(indices), list(range(len(cases))))
            mapping = {}
            for index, other in enumerate(indices):
                self.assertNotEqual(cases[index]["family"], cases[other]["family"])
                self.assertEqual(cases[index]["variant"], cases[other]["variant"])
                mapping.setdefault(cases[index]["family"], set()).add(cases[other]["family"])
            self.assertTrue(all(len(values) == 1 for values in mapping.values()))
            if first_mapping is None:
                first_mapping = mapping
            self.assertEqual(mapping, first_mapping)

    def test_loader_rejects_cross_split_family_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(directory, train_families=2, validation_families=2, test_families=2)
            path = Path(directory)
            (path / "validation.jsonl").write_text((path / "train.jsonl").read_text())
            with self.assertRaisesRegex(ValueError, "world reused"):
                load_partitions(directory)

    def test_structured_oracle_verifies_task_not_model_ability(self):
        parts = generate(train_families=2, validation_families=2, test_families=2)
        oracle = oracle_accuracy(parts)
        self.assertEqual(oracle["accuracy"], 1)
        self.assertIn("not an LM", oracle["role"])
        with self.assertRaises(ValueError):
            parse_symbolic("apple=red;apple=blue;cup=green;lamp=yellow")


class CounterfactualMetricsTests(unittest.TestCase):
    def test_candidate_set_is_fixed_and_rejects_tokenization_changes(self):
        class Tokenizer:
            def encode(self, value, add_special_tokens=False):
                return [COLORS.index(value.strip()) + 10]
        self.assertEqual(candidate_ids(Tokenizer()), list(range(10, 18)))
        class BadTokenizer:
            def encode(self, value, add_special_tokens=False):
                return [10, 11]
        with self.assertRaisesRegex(ValueError, "one distinct token"):
            candidate_ids(BadTokenizer())

    def test_gate_cannot_hide_a_failed_seed_by_averaging(self):
        gate = competence_gate([101, 202, 303], [0.95, 0.95, 0.75])
        self.assertGreater(gate["mean_accuracy"], 0.8)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["receiver_limited_seeds"], [303])
        self.assertFalse(gate["per_seed_passed"]["303"])

    @staticmethod
    def predictions():
        rows = []
        for seed in (101, 202, 303):
            for view in VIEWS:
                for condition in EVALUATED_ARMS:
                    for family in ("family0", "family1"):
                        for key, obj in enumerate(OBJECTS):
                            for variant in VARIANTS:
                                gold = (key + 1) % 8 if variant == "counterfactual" else key
                                choice = gold if condition == "latent" and family == "family0" else key
                                rows.append({"seed": seed, "view": view, "condition": condition, "family": family,
                                             "query_key": obj, "variant": variant, "choice": choice, "gold": gold})
        return rows

    def test_pair_score_requires_correct_answers_not_arbitrary_flips(self):
        rows = self.predictions()
        scores = paired_family_scores(rows)
        self.assertEqual(scores[(101, "iid", "latent", "family0")]["paired_both_correct"], 1)
        self.assertEqual(scores[(101, "iid", "constant", "family0")]["paired_both_correct"], 0)
        self.assertEqual(scores[(101, "iid", "constant", "family0")]["nuisance_invariance"], 1)
        changed_rows = copy.deepcopy(rows)
        for row in changed_rows:
            if row["variant"] == "counterfactual":
                row["choice"] = (row["gold"] + 2) % 8
        changed_scores = paired_family_scores(changed_rows)
        metric = changed_scores[(101, "iid", "latent", "family0")]
        self.assertEqual(metric["answer_change_rate"], 1)
        self.assertEqual(metric["paired_both_correct"], 0)

    def test_bootstrap_clusters_families_without_multiplying_views_or_seeds(self):
        result = summarize(self.predictions(), [101, 202, 303])
        self.assertEqual(result["uncertainty"]["families"], 2)
        self.assertEqual(result["uncertainty"]["bootstrap_resamples"], 2000)
        comparisons = [row for row in result["paired_comparisons"] if row["contrast"] == "latent_minus_shuffled"]
        self.assertEqual(len(comparisons), 4)
        for row in comparisons:
            self.assertEqual(row["mean_difference"], 0.5)
            self.assertEqual(row["paired_family_cluster_95_percentile_interval"], [0.0, 1.0])
            self.assertEqual(row["seed_difference_range"], [0.5, 0.5])
        self.assertTrue(all(row["mean_difference"] == 0 for row in result["paired_view_differences"]))
        with self.assertRaisesRegex(ValueError, "incomplete paired"):
            paired_family_scores(self.predictions()[1:])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional torch/transformers unavailable")
class SelectedHeadEquivalenceTests(unittest.TestCase):
    def test_selected_logits_and_bridge_gradients_equal_full_head_with_padding(self):
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.nn import functional as F
        from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM
        from latent_bridge.counterfactual_runner import selected_logits
        from latent_bridge.transformers_runner import make_bridge

        configurations = [
            (GPT2LMHeadModel, GPT2Config(vocab_size=32, n_positions=64, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)),
            (GPTNeoXForCausalLM, GPTNeoXConfig(vocab_size=32, max_position_embeddings=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, hidden_dropout=0, attention_dropout=0)),
        ]
        for cls, config in configurations:
            with self.subTest(model=config.model_type):
                torch.manual_seed(31)
                receiver = cls(config).eval()
                # Exercise head bias handling even though the real target head
                # has no bias; full and optimized paths must remain equivalent.
                receiver.get_output_embeddings().bias = torch.nn.Parameter(torch.randn(32), requires_grad=False)
                for parameter in receiver.parameters():
                    parameter.requires_grad_(False)
                bridge = make_bridge(12, 16, bottleneck=4, slots=2)
                hidden = torch.randn(2, 5, 12)
                source_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
                prefix = bridge(hidden, source_mask)
                sequences = [torch.cat((prefix[0], receiver.get_input_embeddings()(torch.tensor([1, 2, 3])))),
                             torch.cat((prefix[1], receiver.get_input_embeddings()(torch.tensor([4, 5, 6, 7, 8]))))]
                embeddings = pad_sequence(sequences, batch_first=True)
                mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
                ids = torch.arange(8) + 9
                optimized = selected_logits(receiver, embeddings, mask, ids)
                full = receiver(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
                expected = full[torch.arange(2), mask.sum(dim=1) - 1][:, ids]
                torch.testing.assert_close(optimized, expected, rtol=1e-5, atol=1e-6)
                target = torch.tensor([1, 6])
                parameters = tuple(bridge.parameters())
                optimized_grads = torch.autograd.grad(F.cross_entropy(optimized, target), parameters, retain_graph=True)
                expected_grads = torch.autograd.grad(F.cross_entropy(expected, target), parameters)
                for optimized_grad, expected_grad in zip(optimized_grads, expected_grads):
                    torch.testing.assert_close(optimized_grad, expected_grad, rtol=1e-4, atol=1e-6)
                bad_mask = mask.clone()
                bad_mask[0, 1] = 0
                with self.assertRaisesRegex(ValueError, "right-padded"):
                    selected_logits(receiver, embeddings, bad_mask, ids)


if __name__ == "__main__":
    unittest.main()
