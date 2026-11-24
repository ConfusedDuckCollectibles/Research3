import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from latent_bridge.private_counterfactual import OBJECTS, VARIANTS, VIEWS, query_text
from latent_bridge.query_conditioned_data import write_dataset as write_original
from latent_bridge.readout_data import EXCLUSION_HASH, exclusions, fresh_families, load_partitions, write_dataset
from latent_bridge.readout_probes import (LINEAR_ARMS, MESSAGE_ARMS, fit_preprocessing, flatten_source, freeze_selection,
    label_derangement, preprocess, public_head, secondary_trigger, summarize)
from latent_bridge.readout_reconstruction import compare_results
from latent_bridge.readout_study import validation_pair_accuracy


class ReadoutDataTests(unittest.TestCase):
    def test_fresh_test_excludes_all_previous_worlds_and_fitting_data_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_original(root / "old")
            result = write_dataset(root / "old", root / "fresh")
            parts = load_partitions(root / "fresh")
            self.assertEqual(result["excluded_worlds"], 2002)
            self.assertEqual(result["excluded_world_identities_sha256"], EXCLUSION_HASH)
            self.assertEqual(result["fresh_reserved_worlds"], 160)
            self.assertFalse(set(result["fresh_reserved_identities"]) & set(exclusions()))
            for name in ("train", "validation"):
                self.assertEqual((root / "old" / (name + ".jsonl")).read_bytes(), (root / "fresh" / (name + ".jsonl")).read_bytes())
            self.assertEqual(len(parts["test_iid"]), 128)
            for view in VIEWS:
                self.assertEqual([row["pair_id"] for row in parts["test_" + view]], [row["pair_id"] for row in parts["test_iid"]])
            result["fresh_reserved_identities"][0] = exclusions()[0]
            (root / "fresh" / "manifest.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "reserved identities"):
                load_partitions(root / "fresh")

    def test_fixed_greedy_capacity_is_not_silently_repacked(self):
        self.assertEqual(len(fresh_families(count=53)), 53)
        with self.assertRaisesRegex(ValueError, "fixed greedy"):
            fresh_families(count=54)

    def test_public_parser_is_text_only_and_question_interaction_is_explicit(self):
        for shifted in (False, True):
            for key in range(4):
                self.assertEqual(public_head(query_text(key, shifted)), key)
        with self.assertRaisesRegex(ValueError, "metadata fallback"):
            public_head("The answer is red; hidden query_key=apple")

    def test_label_derangement_preserves_question_identity_without_labels(self):
        cases = [{"family": family, "query": query_text(key), "variant": "original"}
                 for family in ("b", "a", "c") for key in range(4)]
        indices = label_derangement(cases)
        self.assertEqual(sorted(indices), list(range(len(cases))))
        for case, index in zip(cases, indices):
            other = cases[index]
            self.assertNotEqual(case["family"], other["family"])
            self.assertEqual(public_head(case["query"]), public_head(other["query"]))


class ReadoutPreprocessingTests(unittest.TestCase):
    def test_training_stats_do_not_depend_on_heldout_features_and_are_reversible(self):
        training = np.array([[1, 3, 0], [3, 7, 0], [5, 11, 0]], dtype=np.float32)
        statistics = fit_preprocessing(training)
        mean, scale = statistics["mean"].copy(), statistics["scale"].copy()
        heldout = np.array([[10000, -10000, 2]], dtype=np.float32)
        transformed = preprocess(heldout, statistics)
        restored = transformed * np.sqrt(3) * scale + mean
        np.testing.assert_allclose(restored, heldout, rtol=1e-6, atol=1e-4)
        np.testing.assert_array_equal(mean, statistics["mean"])
        np.testing.assert_array_equal(scale, statistics["scale"])
        self.assertEqual(float(scale[2]), 1.0)
        self.assertAlmostEqual(float(transformed[0, 2]), 2 / np.sqrt(3), places=6)

    def test_full_source_preserves_positions_and_rejects_overflow(self):
        states = np.arange(12, dtype=np.float32).reshape(3, 4)
        value = flatten_source(states, 5).reshape(5, 4)
        np.testing.assert_array_equal(value[:3], states)
        np.testing.assert_array_equal(value[3:], np.zeros((2, 4)))
        with self.assertRaisesRegex(ValueError, "overflow"):
            flatten_source(states, 2)


class ReconstructionTests(unittest.TestCase):
    @staticmethod
    def fixture():
        metric = {"restricted_ce": 1.5, "record_accuracy": 0.5, "records": 2}
        result = {"config": {"seeds": [101]}, "candidate_colors": ["red"], "candidate_token_ids": [1],
            "versions": {"torch": "2.6.0+cpu"}, "model_files": {"sender": "hash"}, "data_file_sha256": {"train": "hash"}, "data_manifest": {"seed": 1},
            "runs": [{"seed": 101, "selected_epochs": {"query_aware": 1}, "initial_bridge_state_sha256": {"query_aware": "same"},
                "shared_epoch_training_order_sha256": ["same"], "optimizer_steps_per_arm": 256, "parameters_per_arm": 10378,
                "selected_query_gates": {"query_aware": [0.02, -0.03]},
                "selected_validation": {"query_aware": metric}, "history": [{"epoch": 1, "arm": "query_aware", "validation": metric}]}]}
        rows = [{"seed": 101, "view": "iid", "condition": "query_aware", "family": "family0", "query_key": "apple", "variant": "original", "gold": 0, "choice": 1, "restricted_ce": 1.5}]
        return result, rows

    def test_gate_accepts_declared_small_numerical_variation_only(self):
        old, rows = self.fixture()
        new, new_rows = copy.deepcopy(old), copy.deepcopy(rows)
        new_rows[0]["restricted_ce"] += 1e-6
        new["runs"][0]["selected_query_gates"]["query_aware"][0] += 1e-7
        self.assertTrue(compare_results(old, new, rows, new_rows)["passed"])
        new_rows[0]["restricted_ce"] += 0.001
        result = compare_results(old, new, rows, new_rows)
        self.assertFalse(result["passed"])
        self.assertIn("prediction_ce_tolerance", result["failures"])

    def test_gate_rejects_changed_choices_epochs_nan_and_duplicate_rows(self):
        old, rows = self.fixture()
        new = copy.deepcopy(old)
        new["runs"][0]["selected_epochs"]["query_aware"] = 2
        altered = copy.deepcopy(rows)
        altered[0]["choice"] = 0
        altered[0]["restricted_ce"] = float("nan")
        altered.append(copy.deepcopy(altered[0]))
        result = compare_results(old, new, rows, altered)
        self.assertFalse(result["passed"])
        self.assertIn("seed:101:selected_epochs", result["failures"])
        self.assertIn("predicted_choice_or_gold_mismatch", result["failures"])
        self.assertIn("prediction_ce_tolerance", result["failures"])
        self.assertTrue(any(reason.startswith("duplicate_prediction") for reason in result["failures"]))


class ReadoutSelectionTests(unittest.TestCase):
    def test_trigger_uses_every_validation_seed_and_freeze_requires_all_checkpoints(self):
        seeds = [101, 202, 303]
        trigger = secondary_trigger({101: 0.99, 202: 0.79, 303: 0.99}, seeds)
        self.assertTrue(trigger["triggered"])
        self.assertFalse(trigger["decision_uses_fresh_test"])
        selected = {(seed, "linear", arm): {"selected_epoch": 1, "state_sha256": "test"} for seed in seeds for arm in LINEAR_ARMS}
        with self.assertRaisesRegex(ValueError, "every declared"):
            freeze_selection(selected, seeds, trigger)
        selected.update({(seed, "mlp", arm): {"selected_epoch": 1, "state_sha256": "test"} for seed in seeds for arm in MESSAGE_ARMS})
        self.assertTrue(freeze_selection(selected, seeds, trigger)["all_probe_checkpoints_fixed"])
        with self.assertRaisesRegex(ValueError, "every seed"):
            secondary_trigger({101: 1.0}, seeds)
        self.assertFalse(secondary_trigger(dict.fromkeys(seeds, 0.8), seeds)["triggered"])

    def test_validation_trigger_requires_both_answers_not_flips(self):
        cases = [{"family": "family", "query": query_text(0), "variant": variant,
                  "member": {"answer": answer}} for variant, answer in (("original", "red"), ("counterfactual", "blue"))]
        self.assertEqual(validation_pair_accuracy(cases, [0, 1]), 1)
        self.assertEqual(validation_pair_accuracy(cases, [2, 3]), 0)

    def test_primary_is_accessibility_gap_with_family_cluster_uncertainty(self):
        predictions = []
        conditions = ["linear_message_aware", "linear_message_blind", "linear_message_aware_wrong_source", "linear_pooled_aware", "linear_full_source", "receiver_query_aware"]
        for seed in (101, 202, 303):
            for view in VIEWS:
                for condition in conditions:
                    for family in ("a", "b"):
                        for key, obj in enumerate(OBJECTS):
                            for variant in VARIANTS:
                                gold = key + 1 if variant == "counterfactual" else key
                                choice = gold if condition == "linear_message_aware" and family == "a" else key
                                predictions.append({"seed": seed, "view": view, "condition": condition, "family": family,
                                    "query_key": obj, "variant": variant, "choice": choice, "gold": gold})
        result = summarize(predictions, [101, 202, 303])
        self.assertEqual(result["uncertainty"]["families"], 2)
        primary = next(row for row in result["paired_comparisons"] if row["view"] == "iid" and row["contrast"] == "primary_accessibility_gap")
        self.assertEqual(primary["mean_difference"], 0.5)
        self.assertEqual(primary["paired_family_cluster_95_percentile_interval"], [0.0, 1.0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
class ReadoutTorchTests(unittest.TestCase):
    def test_conditional_heads_can_select_message_coordinates_and_match_budgets(self):
        import torch
        from latent_bridge.readout_probes import make_probe
        probe = make_probe(16)
        self.assertEqual(sum(value.numel() for value in probe.parameters()), 544)
        self.assertEqual(sum(value.numel() for value in make_probe(16, "mlp").parameters()), 3232)
        self.assertEqual(sum(value.numel() for value in make_probe(1536).parameters()), 49184)
        with torch.no_grad():
            for head in probe.heads:
                head.weight.zero_()
                head.bias.zero_()
            probe.heads[0].weight[0, 0] = 1
            probe.heads[1].weight[0, 1] = 1
        features = torch.zeros(2, 16)
        features[:, :2] = torch.tensor([2.0, -3.0])
        output = probe(features, torch.tensor([0, 1]))
        self.assertEqual(output[:, 0].tolist(), [2.0, -3.0])
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(float(probe.heads[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(probe.heads[1].weight.grad.abs().sum()), 0)
        self.assertIsNone(probe.heads[2].weight.grad)

    def test_pooled_features_are_before_layernorm_and_frozen_training_has_no_encoder_gradients(self):
        import torch
        from latent_bridge.query_conditioned import make_query_bridge, normalize_question
        from latent_bridge.readout_probes import make_probe, pool_before_compressor
        torch.manual_seed(73)
        bridge = make_query_bridge(12, 16, bottleneck=8, slots=2).eval()
        for value in bridge.parameters():
            value.requires_grad_(False)
        hidden = torch.randn(3, 5, 12)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        question = normalize_question(torch.randn(3, 12))
        with torch.no_grad():
            pooled = pool_before_compressor(bridge, hidden, mask, question)
            message = bridge.encode(hidden, mask, question)
            torch.testing.assert_close(bridge.compressor(pooled), message)
            changed_padding = hidden.clone()
            changed_padding[~mask.bool()] += 1000
            torch.testing.assert_close(pool_before_compressor(bridge, changed_padding, mask, question), pooled)
        probe = make_probe(16)
        loss = probe(message.reshape(3, 16), torch.tensor([0, 1, 2])).square().mean()
        loss.backward()
        self.assertTrue(all(value.grad is None for value in bridge.parameters()))
        self.assertFalse(message.requires_grad)
        self.assertTrue(any(value.grad is not None for value in probe.parameters()))


if __name__ == "__main__":
    unittest.main()
