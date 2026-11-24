import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from latent_bridge.private_counterfactual import COLORS, OBJECTS, VARIANTS, VIEWS, generate as old_generate, query_text
from latent_bridge.query_conditioned_data import generate, identity_hash, load_partitions, prior_world_exclusions, reserved_identities, write_dataset
from latent_bridge.query_conditioned import EVALUATED_ARMS, advancement_targets, summarize, wrong_public_question


class QueryConditionedDataTests(unittest.TestCase):
    def test_full_dataset_is_fresh_and_keeps_all_family_members_disjoint(self):
        exclusions = prior_world_exclusions()
        self.assertEqual(exclusions["prior_counterfactual_worlds"], 960)
        self.assertEqual(exclusions["prior_pilot_worlds"], 112)
        self.assertEqual(exclusions["prior_study_overlap_worlds"], 30)
        self.assertEqual(exclusions["excluded_worlds"], 1042)
        self.assertEqual(exclusions["excluded_world_identities_sha256"], "4ef5acf31e0db8cc8fe507b7dcbbc16fcd93623f6072c66379c8323236eb366c")
        parts = generate()
        reserved = reserved_identities(parts)
        prior = set(exclusions["excluded_world_identities"])
        seen = set()
        for name, expected in (("train", 640), ("validation", 160), ("test_iid", 160)):
            identities = set(reserved[name])
            self.assertEqual(len(identities), expected)
            self.assertFalse(prior & identities)
            self.assertFalse(seen & identities)
            seen.update(identities)
        self.assertEqual(len(seen), 960)
        self.assertEqual(len(parts["train"]), 512)
        self.assertEqual(len(parts["validation"]), 128)
        for view in VIEWS:
            self.assertEqual(len(parts["test_" + view]), 128)
            for row, iid in zip(parts["test_" + view], parts["test_iid"]):
                self.assertEqual(row["pair_id"], iid["pair_id"])
                self.assertEqual([row[v]["world_id"] for v in VARIANTS], [iid[v]["world_id"] for v in VARIANTS])

    def test_loader_rejects_previous_world_even_with_recomputed_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(directory, train_families=2, validation_families=2, test_families=2)
            path = Path(directory)
            parts = load_partitions(path)
            self.assertEqual(manifest["new_reserved_worlds"], 30)
            old_rows = old_generate(train_families=2, validation_families=2, test_families=2)["train"]
            content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in old_rows)
            (path / "train.jsonl").write_text(content)
            manifest["partitions"]["train"]["sha256"] = hashlib.sha256(content.encode()).hexdigest()
            parts["train"] = old_rows
            identities = reserved_identities(parts)["train"]
            manifest["reserved_worlds_by_partition"]["train"] = {"worlds": len(identities), "identities": identities, "sha256": identity_hash(identities)}
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "prior pretrained-study world"):
                load_partitions(path)

    def test_loader_rejects_exclusion_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(directory, train_families=2, validation_families=2, test_families=2)
            manifest["prior_world_exclusions"]["excluded_world_identities"] = []
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fixed studies"):
                load_partitions(directory)

    def test_wrong_public_question_is_label_free_cyclic_template_preserving(self):
        for shifted in (False, True):
            questions = [query_text(key, shifted) for key in range(len(OBJECTS))]
            for key, question in enumerate(questions):
                self.assertEqual(wrong_public_question(question), questions[(key + 1) % len(OBJECTS)])
                current = question
                for _ in OBJECTS:
                    current = wrong_public_question(current)
                self.assertEqual(current, question)
        with self.assertRaises(ValueError):
            wrong_public_question("The color is red")


class QueryConditionedMetricsTests(unittest.TestCase):
    @staticmethod
    def predictions():
        rows = []
        for seed in (101, 202, 303):
            for view in VIEWS:
                for condition in EVALUATED_ARMS:
                    for family in ("family0", "family1"):
                        for key, obj in enumerate(OBJECTS):
                            for variant in VARIANTS:
                                gold = (key + 1) % len(COLORS) if variant == "counterfactual" else key
                                choice = gold if condition == "query_aware" and family == "family0" else key
                                rows.append({"seed": seed, "view": view, "condition": condition,
                                    "family": family, "query_key": obj, "variant": variant, "choice": choice, "gold": gold})
        return rows

    def test_primary_contrast_uses_paired_family_clusters_and_new_comparator(self):
        result = summarize(self.predictions(), [101, 202, 303])
        self.assertEqual(result["primary_contrast"], {"view": "iid", "metric": "paired_both_correct", "contrast": "query_aware_minus_query_blind"})
        self.assertEqual(result["uncertainty"]["families"], 2)
        primary = [row for row in result["paired_comparisons"] if row["contrast"] == "query_aware_minus_query_blind"]
        self.assertEqual(len(primary), 4)
        for row in primary:
            self.assertEqual(row["mean_difference"], 0.5)
            self.assertEqual(row["paired_family_cluster_95_percentile_interval"], [0.0, 1.0])
            self.assertEqual(row["seed_difference_range"], [0.5, 0.5])
        self.assertTrue(all(row["mean_difference"] == 0 for row in result["paired_view_differences"]))

    def test_advancement_requires_both_targets_and_every_text_gate(self):
        result = summarize(self.predictions(), [101, 202, 303])
        self.assertFalse(advancement_targets(result, {"passed": True})["passed"])
        iid = next(row for row in result["conditions"] if row["view"] == "iid" and row["condition"] == "query_aware")
        iid["metrics"]["paired_both_correct"]["mean"] = 0.8
        iid["metrics"]["nuisance_invariance"]["mean"] = 0.9
        self.assertTrue(advancement_targets(result, {"passed": True})["passed"])
        self.assertFalse(advancement_targets(result, {"passed": False})["passed"])
        iid["metrics"]["nuisance_invariance"]["mean"] = 0.899
        self.assertFalse(advancement_targets(result, {"passed": True})["passed"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
class QueryConditionedBridgeTests(unittest.TestCase):
    def test_normalization_is_population_layernorm_not_unit_l2(self):
        import torch
        from latent_bridge.query_conditioned import constant_question, normalize_question
        vector = torch.tensor([1.0, 2.0, 4.0, 8.0])
        expected = (vector - vector.mean()) / torch.sqrt(((vector - vector.mean()) ** 2).mean() + 1e-5)
        actual = normalize_question(vector)
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(float(actual.mean()), 0, places=6)
        self.assertAlmostEqual(float(actual.norm()), 2, places=5)
        fixed = constant_question(12, 101)
        torch.testing.assert_close(fixed, constant_question(12, 101))
        self.assertFalse(torch.equal(fixed, constant_question(12, 202)))
        self.assertGreater(float(fixed.var(unbiased=False)), 0.999)

    def test_zero_gate_equivalence_and_nonzero_gate_gradients(self):
        import torch
        from latent_bridge.query_conditioned import make_query_bridge, normalize_question
        torch.manual_seed(19)
        bridge = make_query_bridge(12, 16, bottleneck=4, slots=2)
        hidden = torch.randn(3, 5, 12)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        question = normalize_question(torch.randn(3, 12)).requires_grad_(True)
        other = normalize_question(torch.randn(3, 12))
        actual = bridge(hidden, mask, question)
        torch.testing.assert_close(actual, bridge(hidden, mask, other), rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertGreater(float(bridge.gates.grad.abs().sum()), 1e-8)
        self.assertGreater(float(bridge.queries.grad.abs().sum()), 1e-8)
        self.assertEqual(float(question.grad.abs().sum()), 0)
        bridge.zero_grad()
        question.grad = None
        with torch.no_grad():
            bridge.gates.copy_(torch.tensor([0.4, -0.3]))
        changed = bridge(hidden, mask, question)
        self.assertFalse(torch.allclose(changed, bridge(hidden, mask, other)))
        changed.square().sum().backward()
        self.assertGreater(float(question.grad.abs().sum()), 1e-8)

    def test_controls_route_only_permitted_inputs_and_preserve_pair_invariance(self):
        import torch
        from latent_bridge.query_conditioned import condition_prefixes, constant_question, make_query_bridge, normalize_question
        from latent_bridge.transformers_runner import constant_observation
        torch.manual_seed(27)
        bridge = make_query_bridge(12, 16, 4, 2)
        with torch.no_grad():
            bridge.gates.copy_(torch.tensor([0.5, -0.4]))
        fixed_hidden, fixed_mask = constant_observation(12, 101)
        fixed_question = constant_question(12, 101)
        actual = normalize_question(torch.randn(12))
        wrong = normalize_question(torch.randn(12))
        hidden, shuffled = torch.randn(5, 12), torch.randn(7, 12)
        cases = [{"hidden": hidden, "shuffled_hidden": shuffled, "question_vector": actual, "wrong_question_vector": wrong},
                 {"hidden": hidden + 10, "shuffled_hidden": shuffled + 10, "question_vector": actual, "wrong_question_vector": wrong}]
        route = lambda rows, condition: condition_prefixes(rows, condition, bridge, fixed_hidden, fixed_mask, fixed_question)
        blind = route(cases, "query_blind")
        changed_public = [{**case, "question_vector": wrong, "wrong_question_vector": actual} for case in cases]
        torch.testing.assert_close(blind, route(changed_public, "query_blind"), rtol=0, atol=0)
        constant = route(cases, "private_constant")
        torch.testing.assert_close(constant[0], constant[1], rtol=0, atol=0)
        # No private, answer or identity field is even required by this control.
        query_only_cases = [{"question_vector": actual}, {"question_vector": actual}]
        torch.testing.assert_close(constant, route(query_only_cases, "private_constant"), rtol=0, atol=0)
        self.assertFalse(torch.allclose(constant, route(changed_public, "private_constant")))
        torch.testing.assert_close(route(cases, "text"), constant)
        torch.testing.assert_close(route(cases, "symbolic"), constant)
        # Interventions change one encoder input, keeping the other untouched.
        swapped_source = [{**case, "hidden": case["shuffled_hidden"]} for case in cases]
        torch.testing.assert_close(route(cases, "aware_shuffled"), route(swapped_source, "query_aware"))
        torch.testing.assert_close(route(cases, "aware_wrong_question"), route(changed_public, "query_aware"))
        zero = route(cases, "aware_zero")
        torch.testing.assert_close(zero[0], zero[1], rtol=0, atol=0)
        self.assertIsNone(route(cases, "frozen_no_channel"))
        # Arbitrary metadata does not alter message construction.
        with_metadata = [{**case, "target": 7, "facts": {"apple": "purple"}, "query_key": "lamp", "family": "not-a-feature"} for case in cases]
        torch.testing.assert_close(route(with_metadata, "query_aware"), route(cases, "query_aware"), rtol=0, atol=0)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional torch/transformers unavailable")
class QueryConditionedSelectedHeadTests(unittest.TestCase):
    def test_gated_bridge_gradients_match_full_vocabulary_head_with_padding(self):
        import torch
        from torch.nn import functional as F
        from torch.nn.utils.rnn import pad_sequence
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        from latent_bridge.counterfactual_runner import selected_logits
        from latent_bridge.query_conditioned import make_query_bridge, normalize_question
        torch.manual_seed(41)
        config = GPTNeoXConfig(vocab_size=32, max_position_embeddings=32, hidden_size=16,
                              intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                              hidden_dropout=0, attention_dropout=0)
        receiver = GPTNeoXForCausalLM(config).eval()
        receiver.get_output_embeddings().bias = torch.nn.Parameter(torch.randn(32), requires_grad=False)
        for parameter in receiver.parameters():
            parameter.requires_grad_(False)
        bridge = make_query_bridge(12, 16, 4, 2)
        with torch.no_grad():
            bridge.gates.copy_(torch.tensor([0.3, -0.2]))
        prefix = bridge(torch.randn(2, 5, 12), torch.ones(2, 5), normalize_question(torch.randn(2, 12)))
        sequences = [torch.cat((prefix[0], receiver.get_input_embeddings()(torch.tensor([1, 2, 3])))),
                     torch.cat((prefix[1], receiver.get_input_embeddings()(torch.tensor([4, 5, 6, 7, 8]))))]
        embeddings = pad_sequence(sequences, batch_first=True)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        ids = torch.arange(8) + 9
        selected = selected_logits(receiver, embeddings, mask, ids)
        full = receiver(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
        expected = full[torch.arange(2), mask.sum(dim=1) - 1][:, ids]
        torch.testing.assert_close(selected, expected, rtol=1e-5, atol=1e-6)
        parameters = tuple(bridge.parameters())
        targets = torch.tensor([1, 6])
        actual_grads = torch.autograd.grad(F.cross_entropy(selected, targets), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad(F.cross_entropy(expected, targets), parameters)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        gate_index = [name for name, _ in bridge.named_parameters()].index("gates")
        self.assertGreater(float(actual_grads[gate_index].abs().sum()), 1e-8)


if __name__ == "__main__":
    unittest.main()
