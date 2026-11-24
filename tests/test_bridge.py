import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from latent_bridge.core import Bridge, FrozenAgents, decode_latent, decode_packed, decode_symbols, encode_latent, encode_packed, encode_symbols, make_splits, train
from latent_bridge.transformers_runner import different_observation_indices, read_partitions


class DataTests(unittest.TestCase):
    def test_splits_disjoint_complete_reproducible(self):
        splits = make_splits()
        sets = {key: {tuple(row) for row in rows} for key, rows in splits.items()}
        self.assertEqual(sum(map(len, sets.values())), 4096)
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        for name in splits:
            np.testing.assert_array_equal(splits[name], make_splits()[name])

    def test_hf_data_rejects_cross_split_observation_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for index, name in enumerate(("train", "validation", "test")):
                path = Path(directory) / (name + ".jsonl")
                path.write_text(json.dumps({"id": str(index), "sender": "same private facts", "query": "question", "answer": "a"}) + "\n")
                paths[name] = path
            with self.assertRaisesRegex(ValueError, "observations"):
                read_partitions(paths)

    def test_hf_permutation_changes_observation_not_only_row(self):
        rows = [{"sender": sender} for sender in ["a", "a", "b", "b"]]
        indices = different_observation_indices(rows)
        self.assertEqual(sorted(indices), list(range(len(rows))))
        self.assertTrue(all(rows[i]["sender"] != rows[j]["sender"] for i, j in enumerate(indices)))
        with self.assertRaisesRegex(ValueError, "derangement"):
            different_observation_indices([{"sender": "a"}, {"sender": "a"}])


class CodecTests(unittest.TestCase):
    def test_all_symbol_worlds_roundtrip(self):
        for rows in make_splits().values():
            for row in rows:
                np.testing.assert_array_equal(decode_symbols(encode_symbols(row)), row)
                np.testing.assert_array_equal(decode_packed(encode_packed(row)), row)

    def test_latent_roundtrip_framing_and_exact_bytes(self):
        vector = np.array([0.1, 0.2, -0.3, 0.9])
        message = encode_latent(vector)
        self.assertEqual(len(message), 8 + 4 * len(vector))
        np.testing.assert_array_equal(decode_latent(message, 4), vector.astype(np.float32))
        with self.assertRaises(ValueError):
            decode_latent(message[:-1], 4)
        with self.assertRaises(ValueError):
            decode_latent(message, 8)
        with self.assertRaises(ValueError):
            decode_symbols(message)

    def test_malformed_values_rejected(self):
        for world in ([8, 0, 0, 0], [-1, 0, 0, 0], [0.5, 0, 0, 0], [1, 2, 3]):
            with self.assertRaises(ValueError):
                encode_symbols(world)
        with self.assertRaises(ValueError):
            encode_latent([np.nan])
        message = bytearray(encode_packed([0, 0, 0, 0]))
        message[-1] = 255
        with self.assertRaises(ValueError):
            decode_packed(bytes(message))


class LearningTests(unittest.TestCase):
    def test_gradients_against_central_differences(self):
        agents = FrozenAgents.create()
        bridge = Bridge(agents, latent_dim=3, hidden_dim=5)
        labels = make_splits()["train"][:4]
        states = agents.observe(labels)
        _, gradients = bridge.loss_and_gradients(states, labels)
        step = 1e-5
        for key, parameter in bridge.params.items():
            index = tuple(0 for _ in parameter.shape)
            original = parameter[index]
            parameter[index] = original + step
            plus, _ = bridge.loss_and_gradients(states, labels)
            parameter[index] = original - step
            minus, _ = bridge.loss_and_gradients(states, labels)
            parameter[index] = original
            self.assertAlmostEqual((plus - minus) / (2 * step), gradients[key][index], places=7, msg=key)

    def test_training_reduces_validation_loss_with_frozen_agents(self):
        agents = FrozenAgents.create()
        original_sender = agents.sender_weight.copy()
        original_receiver = agents.receiver_readout.copy()
        data = make_splits()
        bridge = Bridge(agents, latent_dim=16)
        validation_states = agents.observe(data["validation"][:64])
        validation_labels = data["validation"][:64]
        before, _ = bridge.loss_and_gradients(validation_states, validation_labels)
        info = train(bridge, agents.observe(data["train"][:256]), data["train"][:256], validation_states, validation_labels, epochs=12, seed=101)
        self.assertLess(info["best_validation_nll"], before)
        np.testing.assert_array_equal(agents.sender_weight, original_sender)
        np.testing.assert_array_equal(agents.receiver_readout, original_receiver)

    def test_receiver_has_only_message_and_query_readout(self):
        bridge = Bridge(FrozenAgents.create(), latent_dim=4)
        duplicated = np.zeros((2, 4))
        np.testing.assert_array_equal(bridge.decode(duplicated)[0], bridge.decode(duplicated)[1])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional torch/transformers unavailable")
class TransformerSmokeTests(unittest.TestCase):
    def test_answer_gradients_reach_bridge_without_updating_base(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from latent_bridge.transformers_runner import answer_loss, make_bridge
        torch.manual_seed(7)
        receiver = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2)).eval()
        for parameter in receiver.parameters():
            parameter.requires_grad_(False)
        bridge = make_bridge(12, 16, bottleneck=4, slots=2)
        hidden = torch.randn(1, 3, 12)
        mask = torch.ones(1, 3)
        tokens = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, 3, 4]])
        loss = answer_loss(receiver, bridge, hidden, mask, tokens, labels)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters()))
        self.assertTrue(all(p.grad is None for p in receiver.parameters()))


if __name__ == "__main__":
    unittest.main()
