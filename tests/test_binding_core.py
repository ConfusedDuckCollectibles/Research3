"""Focused guards and tiny local-model tests; no pretrained downloads or outcomes."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from latent_bridge import binding_core as core
from latent_bridge.binding_data import (COLORS, MEMBERS, OBJECTS, canonical_id,
    public_question, render_source, swap_world)
from latent_bridge.binding_study import FrozenFeatures

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def example_quartet():
    world = tuple(range(8))
    order, reorder = list(range(8)), list(reversed(range(8)))
    variants = [world, swap_world(world, 0, 1), swap_world(world, 2, 3), world]
    members = []
    for label, private in zip(MEMBERS, variants):
        member_order = reorder if label == "R" else order
        members.append({"variant": label, "world": list(private),
                        "world_id": canonical_id(private), "order": member_order,
                        "source": render_source(private, member_order),
                        "symbolic": render_source(private, member_order, True),
                        "answer": COLORS[private[0]]})
    return {"family": "fixture", "query_index": 0, "query": public_question(0),
            "partner": 1, "nuisance_pair": [2, 3], "members": members}


def validation_rows():
    rows = []
    for family in range(32):
        for query in range(8):
            for variant in MEMBERS:
                gold = (query + (variant == "T")) % 8
                rows.append({"family": f"family-{family:02d}", "query_index": query,
                             "variant": variant, "choice": gold, "gold": gold,
                             "restricted_ce": .1})
    return rows


def selected_fit(seed, arm):
    return {
        "seed": seed, "arm": arm, "complete": True, "selected_epoch": 2,
        "state_sha256": f"selected-{seed}-{arm}",
        "initial_state_sha256": f"initial-{seed}", "updates": 1024,
        "epoch_order_sha256": core.order_hashes(core.epoch_orders(seed)),
        "query_gates": [0., 0.], "checkpoint": {"path": f"{seed}/{arm}.npz", "sha256": "file", "bytes": 1},
        "validation": core.validation_metrics(validation_rows()),
    }


class WordTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        tokens = re.findall(r"[A-Za-z]+|[^\s]", text)
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
        return [self.vocabulary[token] for token in tokens]


class BindingPureGuards(unittest.TestCase):
    def test_identical_orders_and_exact_budget(self):
        for seed in core.SEEDS:
            orders = core.epoch_orders(seed)
            self.assertEqual(len(orders), 8)
            self.assertEqual(orders, core.epoch_orders(seed))
            self.assertTrue(all(sorted(order) == list(range(512)) for order in orders))
            self.assertEqual(sum(len(order) // core.QUARTETS_PER_BATCH for order in orders), 1024)
            self.assertEqual(len(set(core.order_hashes(orders))), 8)

    def test_selection_uses_earliest_minimum_ce_only(self):
        history = [
            {"epoch": 1, "validation_ce": 2., "paired_accuracy": 1.},
            {"epoch": 2, "validation_ce": 1., "paired_accuracy": 0.},
            {"epoch": 3, "validation_ce": 1., "paired_accuracy": 1.},
        ]
        self.assertEqual(core.select_epoch(history), 2)
        with self.assertRaises(ValueError):
            core.select_epoch([{"epoch": 1, "validation_ce": float("nan")}])

    def test_each_text_seed_must_pass_and_no_latent_before_gate(self):
        fits = [selected_fit(seed, "text") for seed in core.SEEDS]
        self.assertTrue(core.competence_gate(fits)["passed"])
        fits[0]["validation"]["metrics"]["target_paired_both_correct"] = 204 / 256
        gate = core.competence_gate(fits)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_seeds"], [101])
        for arm in core.ARMS:
            if arm != "text":
                with self.assertRaises(ValueError):
                    core.authorize_fit(arm, fits, gate)
        with self.assertRaises(ValueError):
            core.competence_gate(fits[:2])
        with self.assertRaises(ValueError):
            core.authorize_fit("aware_ce", fits[:2], {"passed": True})
        core.authorize_fit("text", [], None)

    def test_all_three_gate_metrics_are_required(self):
        for metric in ("target_paired_both_correct", "nuisance_stability", "reorder_stability"):
            fits = [selected_fit(seed, "text") for seed in core.SEEDS]
            fits[1]["validation"]["metrics"][metric] = .5
            self.assertFalse(core.competence_gate(fits)["passed"])

    def test_validation_covers_all_families_and_queries(self):
        rows = validation_rows()
        report = core.validation_metrics(rows)
        self.assertEqual(report["quartets"], 256)
        self.assertEqual(report["counts"]["target_paired_both_correct"], 256)
        broken = copy.deepcopy(rows)
        for row in broken:
            if row["family"] == "family-00" and row["query_index"] == 7:
                row["query_index"] = 8
        with self.assertRaises(ValueError):
            core.validation_metrics(broken)
        with self.assertRaises(ValueError):
            core.validation_metrics(rows[:-1])

    def test_complete_persisted_selection_precedes_test_permission(self):
        fits = [selected_fit(seed, arm) for seed in core.SEEDS for arm in core.ARMS]
        gate = core.competence_gate([row for row in fits if row["arm"] == "text"])
        with self.assertRaises(ValueError):
            core.freeze_selection(fits[:-1], gate)
        for key, value in (("updates", 1023), ("initial_state_sha256", "different")):
            bad = copy.deepcopy(fits)
            bad[-1][key] = value
            with self.assertRaises(ValueError):
                core.freeze_selection(bad, gate)
        selected = core.freeze_selection(fits, gate)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selection.json"
            with self.assertRaises(ValueError):
                core.require_test_authorized(path, fits, gate)
            path.write_text(json.dumps(selected))
            self.assertEqual(core.require_test_authorized(path, fits, gate), selected)
            changed = dict(selected, fresh_test_model_cases_loaded=True)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                core.require_test_authorized(path, fits, gate)

    def test_token_preflight_checks_actual_ids_and_candidate_order(self):
        data = {"fixture": [example_quartet()]}
        tokenizers = {"sender": WordTokenizer(), "receiver": WordTokenizer()}
        report = core.tokenizer_preflight(data, tokenizers)
        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate_colors"], list(COLORS))
        self.assertEqual(report["candidate_token_ids"],
                         [tokenizers["receiver"].encode(" " + color)[0] for color in COLORS])
        class BoundaryTokenizer(WordTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                return ids + ([999] if text.startswith("The apple is red.") else [])
        with self.assertRaisesRegex(ValueError, "multiset"):
            core.tokenizer_preflight(data, {"sender": BoundaryTokenizer(), "receiver": WordTokenizer()})
        class CollidingTokenizer(WordTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text in (" red", " blue"):
                    return [42]
                return super().encode(text, add_special_tokens)
        with self.assertRaisesRegex(ValueError, "candidate"):
            core.tokenizer_preflight(data, {"sender": WordTokenizer(), "receiver": CollidingTokenizer()})

    def test_receiver_separator_and_interventions_are_public_only(self):
        case = core.flattened_cases([example_quartet()])[0]
        self.assertEqual(core.RECEIVER_SEPARATOR, "\n")
        self.assertEqual(core.receiver_text(case, "text"),
                         case["member"]["source"] + "\n" + case["query"])
        for query in range(8):
            self.assertEqual(core.wrong_public_question(public_question(query)),
                             public_question((query + 1) % 8))
        quartets = []
        for family in ("z", "a", "m"):
            for query in range(8):
                quartets.append({"family": family, "query_index": query, "answer": "unused"})
        mapping = core.source_successor_map(quartets)
        for index, target in enumerate(mapping):
            self.assertNotEqual(quartets[index]["family"], quartets[target]["family"])
            self.assertEqual(quartets[index]["query_index"], quartets[target]["query_index"])
        altered = copy.deepcopy(quartets)
        for row in altered:
            row["answer"] = "a different unused answer"
        self.assertEqual(mapping, core.source_successor_map(altered))

    def test_stability_and_source_following_are_distinct(self):
        members = {row["variant"]: row for row in validation_rows()[:4]}
        for row in members.values():
            row["choice"] = 7
            row["source_gold"] = 7
        metric = core.quartet_metrics(members)
        self.assertEqual(metric["nuisance_stability"], 1.)
        self.assertEqual(metric["target_paired_both_correct"], 0.)
        self.assertEqual(metric["supplied_source_record_accuracy"], 1.)


@unittest.skipUnless(HAS_TORCH, "optional PyTorch is not installed")
class BindingTensorTests(unittest.TestCase):
    def test_exact_raw_message_objective_and_common_ce(self):
        import torch
        logits = torch.zeros((8, 8), dtype=torch.float64, requires_grad=True)
        targets = torch.arange(8)
        message = torch.zeros((2, 4, 2, 8), dtype=torch.float64)
        message[0, 1] = .25
        message[0, 2] = .5
        message[0, 3] = -.5
        message[1, 0] = -1
        message[1, 1] = 1
        message[1, 2:] = -1
        message = message.reshape(8, 2, 8).requires_grad_()
        ce, ce_terms = core.quartet_loss(logits, targets, message, False)
        joint, terms = core.quartet_loss(logits, targets, message, True)
        self.assertAlmostEqual(float(ce), math.log(8), places=12)
        self.assertAlmostEqual(float(joint - ce), .21875, places=12)
        self.assertEqual(float(ce_terms["restricted_ce"]), float(terms["restricted_ce"]))
        gradients = torch.autograd.grad(ce, logits, retain_graph=True)[0]
        joint_gradients = torch.autograd.grad(joint, logits, retain_graph=True)[0]
        self.assertTrue(torch.equal(gradients, joint_gradients))
        self.assertTrue(bool((gradients.abs().sum(dim=1) > 0).all()))
        message_gradient = torch.autograd.grad(joint, message)[0]
        self.assertGreater(float(message_gradient.abs().sum()), 0.)
        self.assertLessEqual(float(terms["target_distance"]), 4.)

    def test_auxiliary_gradcheck_uses_raw_coordinates(self):
        import torch
        torch.manual_seed(41)
        message = (.1 * torch.randn((4, 2, 8), dtype=torch.float64)).requires_grad_()
        logits = torch.randn((4, 8), dtype=torch.float64)
        targets = torch.tensor([0, 1, 0, 0])
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: core.quartet_loss(logits, targets, z, True)[0],
            (message,), eps=1e-6, atol=1e-5, rtol=1e-3))
        with self.assertRaises(ValueError):
            core.quartet_loss(logits, targets, torch.full_like(message, 1.1), True)

    def test_routing_controls_and_zero_gate_equivalence(self):
        import torch
        torch.manual_seed(7)
        bridge = core.make_query_bridge(12, 16)
        hidden, mask = torch.randn(4, 7, 12), torch.ones(4, 7)
        question = core.normalize_question(torch.randn(4, 12))
        fixed_hidden, fixed_mask = core.constant_observation(12, 101)
        fixed_question = core.constant_question(12, 101)
        def encoded(arm, h=hidden, q=question):
            return bridge.encode(*core.route_inputs(
                arm, h, mask, q, fixed_hidden, fixed_mask, fixed_question))
        self.assertTrue(torch.equal(encoded("aware_ce"), encoded("aware_aux")))
        self.assertTrue(torch.equal(encoded("aware_ce"), encoded("blind_ce")))
        encoded("aware_ce").square().sum().backward()
        self.assertGreater(float(bridge.gates.grad.abs().sum()), 0.)
        with torch.no_grad():
            bridge.gates.fill_(.4)
        self.assertTrue(torch.equal(encoded("blind_ce"), encoded("blind_ce", q=-question)))
        same_query = question[0:1].expand(4, -1)
        a = encoded("private_constant", q=same_query)
        b = encoded("private_constant", h=hidden * 10, q=same_query)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[0], a[1]))
        for arm in ("text", "symbolic"):
            self.assertTrue(torch.equal(a, encoded(arm, q=same_query)))

    def test_portable_checkpoint_round_trip(self):
        import torch
        torch.manual_seed(8)
        bridge = core.make_query_bridge(12, 16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bridge.npz"
            saved = core.save_state(path, bridge.state_dict())
            recovered = core.load_state(path)
            self.assertEqual(core.state_hash(recovered), saved["state_sha256"])
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(bridge.state_dict()))
            for name, tensor in bridge.state_dict().items():
                self.assertTrue(torch.equal(tensor, recovered[name]))

    def test_exact_source_and_prompt_cache_separate_reordering(self):
        import torch
        class Sender(torch.nn.Module):
            def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
                x = input_ids.float()
                positions = torch.arange(x.shape[1])[None, :].expand_as(x).float()
                return SimpleNamespace(last_hidden_state=torch.stack(
                    (x, positions, x + positions, x.square(), positions.square(), torch.ones_like(x)), dim=-1))
        class Receiver(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = torch.nn.Embedding(2048, 8)
            def get_input_embeddings(self):
                return self.embeddings
        cases = core.flattened_cases([example_quartet()])
        original, reordered = cases[0], cases[3]
        self.assertEqual(original["member"]["world_id"], reordered["member"]["world_id"])
        self.assertNotEqual(original["member"]["source"], reordered["member"]["source"])
        features = FrozenFeatures(Sender(), Receiver(), WordTokenizer(), WordTokenizer(),
                                  torch.arange(8), {})
        features.prepare([original, reordered], private=True, receiver_arms=("text",))
        self.assertEqual(len(features.sources), 2)
        self.assertEqual(len(features.static), 2)
        self.assertFalse(torch.equal(features.sources[original["member"]["source"]],
                                     features.sources[reordered["member"]["source"]]))
        self.assertEqual(features.encoding_sequences["source"], 2)
        self.assertEqual(features.encoding_sequences["question"], 1)


@unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "optional transformer dependencies unavailable")
class BindingTransformerTests(unittest.TestCase):
    def test_eight_selected_logits_and_gradients_match_full_head_with_bias(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(29)
        config = GPT2Config(vocab_size=64, n_embd=16, n_layer=1, n_head=2,
                            n_positions=32, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        receiver = GPT2LMHeadModel(config).double().eval()
        receiver.set_output_embeddings(torch.nn.Linear(16, 64, bias=True).double())
        embedding = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
        ids = torch.tensor([1, 9, 3, 12, 4, 15, 6, 21])
        selected = core.selected_logits(receiver, embedding, mask, ids)
        full = receiver(inputs_embeds=embedding, attention_mask=mask,
                        use_cache=False, return_dict=True).logits
        expected = full[torch.arange(2), mask.sum(dim=1) - 1][:, ids]
        self.assertTrue(torch.allclose(selected, expected, atol=1e-10, rtol=1e-10))
        head = receiver.get_output_embeddings()
        left = torch.autograd.grad(selected.square().sum(), (embedding, head.weight, head.bias), retain_graph=True)
        right = torch.autograd.grad(expected.square().sum(), (embedding, head.weight, head.bias))
        for actual, reference in zip(left, right):
            self.assertTrue(torch.allclose(actual, reference, atol=1e-10, rtol=1e-10))

    def test_frozen_receiver_passes_ce_gradients_to_actual_bridge(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(31)
        receiver = GPT2LMHeadModel(GPT2Config(
            vocab_size=64, n_embd=16, n_layer=1, n_head=2, n_positions=32,
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)).eval()
        for parameter in receiver.parameters():
            parameter.requires_grad_(False)
        bridge = core.make_query_bridge(12, 16)
        hidden = torch.randn(4, 6, 12)
        question = core.normalize_question(torch.randn(4, 12))
        message = bridge.encode(hidden, torch.ones(4, 6), question)
        prefix = bridge.decode(message)
        static = receiver.get_input_embeddings()(torch.tensor([[2, 3, 4]] * 4)).detach()
        embeddings = torch.cat((prefix, static), dim=1)
        logits = core.selected_logits(receiver, embeddings, torch.ones(4, 5), torch.arange(8))
        loss, _ = core.quartet_loss(logits, torch.tensor([0, 1, 0, 0]), message, True)
        loss.backward()
        self.assertGreater(float(bridge.expander.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(bridge.compressor[1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in receiver.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in receiver.parameters()))


if __name__ == "__main__":
    unittest.main()
