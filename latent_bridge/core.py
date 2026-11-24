"""Small NumPy bridge and exact wire formats for an auditable controlled task."""
from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass

import numpy as np

FIELDS = 4
VALUES = 8
SEMANTIC_DIM = FIELDS * VALUES
HEADER = struct.Struct("<4sBBH")


def make_splits(seed: int = 23):
    """Split all 4,096 distinct worlds, never repeated rows across partitions."""
    worlds = np.array(list(itertools.product(range(VALUES), repeat=FIELDS)), dtype=np.uint8)
    order = np.random.default_rng(seed).permutation(len(worlds))
    a, b = int(len(worlds) * 0.6), int(len(worlds) * 0.8)
    return {"train": worlds[order[:a]], "validation": worlds[order[a:b]], "test": worlds[order[b:]]}


def one_hot(worlds):
    worlds = np.asarray(worlds)
    if worlds.ndim != 2 or worlds.shape[1] != FIELDS or np.any(worlds < 0) or np.any(worlds >= VALUES):
        raise ValueError("worlds must have four categorical attributes in [0, 8)")
    if not np.issubdtype(worlds.dtype, np.integer):
        raise ValueError("categorical attributes must be integers")
    return np.eye(VALUES)[worlds].reshape(len(worlds), SEMANTIC_DIM)


@dataclass
class FrozenAgents:
    sender_weight: np.ndarray
    receiver_readout: np.ndarray

    @classmethod
    def create(cls, seed=11, sender_dim=48, receiver_dim=40):
        if receiver_dim < SEMANTIC_DIM:
            raise ValueError("receiver dimension must support the constructed semantic basis")
        rng = np.random.default_rng(seed)
        sender = rng.normal(0, 0.6, (SEMANTIC_DIM, sender_dim))
        receiver, _ = np.linalg.qr(rng.normal(size=(receiver_dim, SEMANTIC_DIM)))
        return cls(sender, receiver)

    def observe(self, worlds):
        return np.tanh(one_hot(worlds) @ self.sender_weight)


class Bridge:
    """A learned sender compressor and receiver adapter around frozen agents.

    The sender observes the entire private world before the receiver chooses a
    query. The receiver can request any one of four attributes from the same
    transmitted vector. It has no other access to the world.
    """

    def __init__(self, agents: FrozenAgents, latent_dim=8, hidden_dim=64, seed=1):
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.agents = agents
        self.latent_dim = latent_dim
        dims = [agents.sender_weight.shape[1], hidden_dim, latent_dim, hidden_dim, agents.receiver_readout.shape[0]]
        rng = np.random.default_rng(seed)
        self.params = {}
        for i, (left, right) in enumerate(zip(dims, dims[1:]), 1):
            self.params[f"w{i}"] = rng.normal(0, (2 / (left + right)) ** 0.5, (left, right))
            self.params[f"b{i}"] = np.zeros(right)

    def encode(self, states):
        p = self.params
        h = np.tanh(states @ p["w1"] + p["b1"])
        return np.tanh(h @ p["w2"] + p["b2"])

    def decode(self, messages):
        messages = np.asarray(messages)
        if messages.ndim != 2 or messages.shape[1] != self.latent_dim or not np.isfinite(messages).all():
            raise ValueError("invalid latent message dimensions or non-finite values")
        p = self.params
        h = np.tanh(messages @ p["w3"] + p["b3"])
        receiver_states = h @ p["w4"] + p["b4"]
        return (receiver_states @ self.agents.receiver_readout).reshape(-1, FIELDS, VALUES)

    def loss_and_gradients(self, states, worlds):
        p = self.params
        h1 = np.tanh(states @ p["w1"] + p["b1"])
        z = np.tanh(h1 @ p["w2"] + p["b2"])
        h3 = np.tanh(z @ p["w3"] + p["b3"])
        output = h3 @ p["w4"] + p["b4"]
        logits = (output @ self.agents.receiver_readout).reshape(-1, FIELDS, VALUES)
        loss, dlogits = cross_entropy(logits, worlds, gradient=True)
        doutput = dlogits.reshape(len(worlds), -1) @ self.agents.receiver_readout.T
        grads = {"w4": h3.T @ doutput, "b4": doutput.sum(axis=0)}
        d3 = (doutput @ p["w4"].T) * (1 - h3 ** 2)
        grads.update(w3=z.T @ d3, b3=d3.sum(axis=0))
        d2 = (d3 @ p["w3"].T) * (1 - z ** 2)
        grads.update(w2=h1.T @ d2, b2=d2.sum(axis=0))
        d1 = (d2 @ p["w2"].T) * (1 - h1 ** 2)
        grads.update(w1=states.T @ d1, b1=d1.sum(axis=0))
        return loss, grads


def cross_entropy(logits, labels, gradient=False):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    rows = np.arange(len(labels))[:, None]
    fields = np.arange(FIELDS)[None, :]
    log_norm = np.log(exp.sum(axis=-1))
    loss = float(np.mean(log_norm - shifted[rows, fields, labels]))
    if not gradient:
        return loss
    probs[rows, fields, labels] -= 1
    return loss, probs / (len(labels) * FIELDS)


def score(logits, labels):
    correct = logits.argmax(axis=-1) == labels
    return {"attribute_accuracy": float(correct.mean()), "all_attributes_accuracy": float(correct.all(axis=1).mean()), "nll": cross_entropy(logits, labels)}


def train(bridge, train_states, train_labels, validation_states, validation_labels,
          epochs=180, batch_size=128, learning_rate=0.003, seed=1):
    """Fixed epoch budget; restore best validation NLL, never inspect test data."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    rng = np.random.default_rng(seed)
    first = {k: np.zeros_like(v) for k, v in bridge.params.items()}
    second = {k: np.zeros_like(v) for k, v in bridge.params.items()}
    best, checkpoint, best_epoch, step = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            _, grads = bridge.loss_and_gradients(train_states[idx], train_labels[idx])
            step += 1
            for key, grad in grads.items():
                first[key] = 0.9 * first[key] + 0.1 * grad
                second[key] = 0.999 * second[key] + 0.001 * grad * grad
                bridge.params[key] -= learning_rate * (first[key] / (1 - 0.9 ** step)) / (np.sqrt(second[key] / (1 - 0.999 ** step)) + 1e-8)
        val = cross_entropy(bridge.decode(bridge.encode(validation_states)), validation_labels)
        if val < best:
            best, best_epoch = val, epoch
            checkpoint = {k: v.copy() for k, v in bridge.params.items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "validation_nll": val})
    bridge.params = checkpoint
    return {"best_epoch": best_epoch, "best_validation_nll": best, "optimizer_steps": step, "history": history}


def frame(payload: bytes, kind: int):
    if len(payload) > 65535 or kind not in (1, 2, 3, 4):
        raise ValueError("unsupported payload or message kind")
    return HEADER.pack(b"LBR1", 1, kind, len(payload)) + payload


def unframe(message: bytes, kind: int):
    if len(message) < HEADER.size:
        raise ValueError("truncated header")
    magic, version, actual, size = HEADER.unpack(message[:HEADER.size])
    payload = message[HEADER.size:]
    if magic != b"LBR1" or version != 1 or actual != kind or size != len(payload):
        raise ValueError("invalid message header, kind, or payload length")
    return payload


def encode_latent(vector):
    vector = np.asarray(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("latent vector must be finite and one-dimensional")
    wire = vector.astype("<f4")
    if not np.isfinite(wire).all():
        raise ValueError("latent values overflow float32")
    return frame(wire.tobytes(), 1)


def decode_latent(message, expected_dim):
    payload = unframe(message, 1)
    if expected_dim <= 0 or len(payload) != expected_dim * 4:
        raise ValueError("latent payload has wrong dimension")
    result = np.frombuffer(payload, dtype="<f4").copy()
    if not np.isfinite(result).all():
        raise ValueError("non-finite latent payload")
    return result


def encode_symbols(world):
    world = np.asarray(world)
    one_hot(world.reshape(1, -1))
    return frame(world.astype(np.uint8).tobytes(), 2)


def decode_symbols(message):
    payload = unframe(message, 2)
    if len(payload) != FIELDS:
        raise ValueError("symbol payload must contain four attributes")
    world = np.frombuffer(payload, dtype=np.uint8).copy()
    one_hot(world.reshape(1, -1))
    return world


def encode_packed(world):
    world = decode_symbols(encode_symbols(world))
    bits = 0
    for value in world:
        bits = (bits << 3) | int(value)
    return frame(bits.to_bytes(2, "little"), 3)


def decode_packed(message):
    payload = unframe(message, 3)
    if len(payload) != 2:
        raise ValueError("packed payload must be two bytes")
    bits = int.from_bytes(payload, "little")
    if bits >= 4096:
        raise ValueError("nonzero padding bits")
    return np.array([(bits >> (3 * (3 - i))) & 7 for i in range(4)], dtype=np.uint8)


def encode_english(world):
    world = decode_symbols(encode_symbols(world))
    return frame(("The object has " + ", ".join(f"{name}={int(value)}" for name, value in zip("abcd", world)) + ".").encode("utf-8"), 4)
