"""Independent frozen-cohort, extraction, instrumentation and retention tests.

The generator golden strings were derived before model outcomes with an
independent JavaScript implementation of CPython's integer-seeded MT19937,
getrandbits rejection and shuffle (checked against Random(0).random()).
All model calls below use a tiny randomly initialized local GPTNeoX config.
No pretrained checkpoint, tokenizer, download, empirical cohort extraction,
training or measurement is invoked.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from latent_bridge import contextual_information as study


TREES = ("alder", "birch", "cedar", "elm", "fir", "maple", "oak", "willow")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white")
WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight")
GOLDEN_BASES = """
12053764 30246517 32540617 32574016 30471265 76254103 27641035 02567413
05763241 13067452 40673521 76150243 24576013 32165740 23175460 76304215
13724560 57241603 32046751 30264517 74506213 74532061 65720143 53401276
54026731 07534126 57210463 25071346 46501372 76420153 62514073 62407135
75631402 35701426 74125063 07312546 12753064 71264503 75304126 31562047
70541263 26341075 41573206 46512703 70214653 34120567 12074536 52340761
02637154 30751624 12603574 54602317 27504613 36257014 37160245 52367140
65703412 23475061 24506173 10564732 07456213 75106423 26451730 06371425
""".split()


class FixedTokenizer:
    """A deterministic local tokenizer for preflight logic, not model evidence."""
    def __init__(self):
        self.word_ids = {word: 100+i for i, word in enumerate(WORDS)}
        self.word_ids.update({word: 200+i for i, word in enumerate(COLORS)})
        self.other_ids = {}
        self.overrides = {}
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The fixed contract forbids added special tokens")
        self.calls.append(text)
        if text in self.overrides:
            return list(self.overrides[text])
        if text.startswith(" ") and text[1:] in self.word_ids:
            return [self.word_ids[text[1:]]]
        result = []
        for token in re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text):
            if token in self.word_ids:
                result.append(self.word_ids[token])
            else:
                if token not in self.other_ids:
                    self.other_ids[token] = 300+len(self.other_ids)
                result.append(self.other_ids[token])
        return result


class FrozenCohortContracts(unittest.TestCase):
    def test_all_golden_worlds_swaps_labels_and_exact_prompt_grammar(self):
        cohort = study.build_cohort()
        self.assertEqual(cohort["seed"], 41029)
        self.assertEqual(len(cohort["families"]), 64)
        self.assertEqual(cohort["generation"]["total_candidates"], 64)
        reserved = set()
        for index, (family, golden) in enumerate(zip(cohort["families"], GOLDEN_BASES)):
            expected = list(map(int, golden))
            target = index % 8
            partner = (target+1+(index//8)%7) % 8
            swapped = expected.copy()
            swapped[target], swapped[partner] = swapped[partner], swapped[target]
            fid = f"F{index:03d}"
            question = f"What color is the {TREES[target]} specimen?"
            self.assertEqual(family["family_id"], fid)
            self.assertEqual(family["index"], index)
            self.assertEqual(family["target_index"], target)
            self.assertEqual(family["partner_index"], partner)
            self.assertNotEqual(target, partner)
            self.assertEqual(family["question"], question)
            self.assertEqual(family["generation"]["draws"], 1)
            for name, permutation in (("original", expected), ("swapped", swapped)):
                world = family["worlds"][name]
                self.assertEqual(world["permutation"], permutation)
                self.assertEqual(world["world_id"], "".join(map(str, permutation)))
                self.assertEqual(world["answer_index"], permutation[target])
                self.assertEqual(world["answer"], COLORS[permutation[target]])
                prompt = "\n".join(
                    [f"Family {fid}.", "Private record:"]
                    + [f"{tree}: {COLORS[color]}" for tree, color in zip(TREES, permutation)]
                    + [f"Question: {question}", "Answer:"]
                )
                self.assertEqual(world["prompt"], prompt)
                self.assertNotIn("\r", prompt)
                self.assertFalse(prompt.endswith(("\n", " ")))
                self.assertNotIn(tuple(permutation), reserved)
                reserved.add(tuple(permutation))
            changes = [i for i in range(8) if expected[i] != swapped[i]]
            self.assertEqual(set(changes), {target, partner})
            self.assertEqual(sorted(expected), sorted(swapped))
            self.assertNotEqual(
                family["worlds"]["original"]["answer"],
                family["worlds"]["swapped"]["answer"],
            )
        self.assertEqual(len(reserved), 128)
        expected_lists = [list(p) for p in sorted(reserved)]
        raw = json.dumps(expected_lists, sort_keys=True, separators=(",", ":")).encode("ascii")
        self.assertEqual(cohort["reserved_worlds"], expected_lists)
        self.assertEqual(cohort["reserved_worlds_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(cohort, study.build_cohort())

    def test_plan_has_exact_order_and_shared_public_pair_fields(self):
        cohort = study.build_cohort()
        before = json.dumps(cohort, sort_keys=True)
        plan = study.build_plan(cohort)
        expected = [
            f"F{f:03d}/{world}/{probe}"
            for f in range(64)
            for world in ("original", "swapped")
            for probe in ("unique", "repeated")
        ]
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["expected_count"], 256)
        self.assertEqual([r["record_id"] for r in plan["planned_records"]], expected)
        self.assertEqual(len(set(expected)), 256)
        for start in range(0, 256, 4):
            rows = plan["planned_records"][start:start+4]
            self.assertEqual(len({r["family_id"] for r in rows}), 1)
            self.assertEqual(len({r["question"] for r in rows}), 1)
            self.assertEqual(len({r["target_index"] for r in rows}), 1)
            self.assertEqual(rows[0]["answer_index"], rows[1]["answer_index"])
            self.assertEqual(rows[2]["answer_index"], rows[3]["answer_index"])
            self.assertNotEqual(rows[0]["answer_index"], rows[2]["answer_index"])
        self.assertEqual(json.dumps(cohort, sort_keys=True), before)

    def test_token_preflight_checks_fixed_ids_and_all_pair_bags(self):
        cohort = study.build_cohort()
        tokenizer = FixedTokenizer()
        result = study.tokenize_cohort(tokenizer, cohort)
        self.assertTrue(result["passed"])
        self.assertEqual(result["probe_ids"]["unique"], list(range(100, 108)))
        self.assertEqual(result["probe_ids"]["repeated"], [100]*8)
        self.assertEqual(result["color_ids"], list(range(200, 208)))
        self.assertEqual(len(result["prompt_ids"]), 128)
        for f in range(64):
            a = result["prompt_ids"][f"F{f:03d}/original"]
            b = result["prompt_ids"][f"F{f:03d}/swapped"]
            self.assertEqual(len(a), len(b))
            self.assertEqual(Counter(a), Counter(b))
            self.assertLessEqual(len(a), 256)
        # Two independent probe lists are not permission to add BOS/EOS.
        self.assertEqual(tokenizer.calls[:8], [" "+word for word in WORDS])

    def test_preflight_rejects_token_replacements_duplicates_bag_and_length(self):
        cohort = study.build_cohort()
        for condition in ("split_probe", "duplicate_probe", "split_color", "bag", "long"):
            with self.subTest(condition=condition):
                tokenizer = FixedTokenizer()
                if condition == "split_probe":
                    tokenizer.overrides[" one"] = [100, 101]
                elif condition == "duplicate_probe":
                    tokenizer.overrides[" two"] = [100]
                elif condition == "split_color":
                    tokenizer.overrides[" white"] = [207, 208]
                else:
                    prompt = cohort["families"][0]["worlds"]["swapped"]["prompt"]
                    if condition == "bag":
                        original = tokenizer.encode(prompt)
                        tokenizer.overrides[prompt] = original[:-1]+[999]
                    else:
                        tokenizer.overrides[prompt] = list(range(257))
                with self.assertRaises(study.PreflightError):
                    study.tokenize_cohort(tokenizer, cohort)


def author_context(vocabulary):
    vocabulary = torch.as_tensor(vocabulary, dtype=torch.float32).clone()
    embedding = torch.nn.Embedding.from_pretrained(vocabulary.clone(), freeze=True)
    return SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, embedding_layer=embedding,
        vocab_embeds=vocabulary, target_norm=vocabulary.norm(dim=-1).mean().item(),
        adaptive_reg=1e-3, snap_ratio=.3,
    )


def offline_author_fixtures():
    # Centered isotropic full feature rank forces a nonzero-cost SO correction.
    h = np.array([
        [1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
        [-1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
        [0., -2/math.sqrt(6), 1/math.sqrt(12)],
        [0., 0., -3/math.sqrt(12)],
    ], dtype=np.float32)
    reflected = h @ np.diag(np.array([-1., 1., 1.], dtype=np.float32))
    repeated_vocabulary = np.array([
        [.7, .2, -.1], [-.3, .5, .8], [.1, -.6, .4], [.2, .1, .9],
    ], dtype=np.float32)
    return [
        ("proper_rotation", h, reflected, [0, 1, 2, 3]),
        ("repeated_ids", h, repeated_vocabulary, [0, 0, 0, 0]),
    ]


class AuthorExecutionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.function, cls.metadata = study.load_author_method()
        path = Path(study.VENDOR_SOURCE)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != (
            "415957df427eb79a4212e760af4d133cb4ce10cc4619f11447082117922f366d"
        ):
            raise AssertionError("Original author fixture source changed")
        parsed = ast.parse(raw.decode("utf-8"), filename=str(path))
        original_class = next(
            node for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "StateBridge"
        )
        node = next(
            item for item in original_class.body
            if isinstance(item, ast.FunctionDef) and item.name == "_align_hidden_sequence"
        )
        cls.expected_ast_hash = hashlib.sha256(
            ast.dump(node, annotate_fields=True, include_attributes=True).encode("utf-8")
        ).hexdigest()
        namespace = {"torch": torch, "F": functional}
        # Independent extraction: compile the original method without the loader.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        cls.baseline = staticmethod(namespace["_align_hidden_sequence"])
        cls.function = staticmethod(cls.function)

    def test_unmodified_author_output_and_return_capture_on_both_boundaries(self):
        self.assertEqual(self.metadata["method_ast_sha256"], self.expected_ast_hash)
        self.assertFalse(self.metadata["body_modified"])
        self.assertFalse(self.metadata["application_imported"])
        for label, source, vocabulary, ids in offline_author_fixtures():
            with self.subTest(fixture=label), torch.no_grad():
                context = author_context(vocabulary)
                hidden = torch.tensor(source).unsqueeze(0)
                tokens = torch.tensor([ids], dtype=torch.long)
                expected = self.baseline(context, hidden.clone(), tokens)
                direct = self.function(context, hidden.clone(), tokens)
                old_profile = sys.getprofile()
                def prior_profile(frame, event, arg):
                    return None
                try:
                    sys.setprofile(prior_profile)
                    captured_output, captured = study.invoke_with_capture(
                        self.function, context, hidden.clone(), tokens
                    )
                    self.assertIs(sys.getprofile(), prior_profile)
                finally:
                    sys.setprofile(old_profile)
                self.assertEqual(expected.numpy().tobytes(), direct.numpy().tobytes())
                self.assertEqual(expected.numpy().tobytes(), captured_output.numpy().tobytes())
                self.assertEqual(captured_output.dtype, torch.float32)
                self.assertEqual(set(captured), set(study.AUTHOR_LOCALS))
                self.assertGreater(torch.det(captured["R"]).item(), .999)
                if label == "proper_rotation":
                    cross = captured["H_w"].T @ captured["E_w"]
                    self.assertLess(torch.det(cross).item(), 0.0)

    def test_capture_restores_profiler_when_original_method_raises(self):
        class BrokenEmbedding(torch.nn.Module):
            def forward(self, token_ids):
                raise RuntimeError("fixed synthetic embedding failure")
        context = author_context(np.eye(3, dtype=np.float32))
        context.embedding_layer = BrokenEmbedding()
        previous = sys.getprofile()
        def prior_profile(frame, event, arg):
            return None
        try:
            sys.setprofile(prior_profile)
            with self.assertRaisesRegex(RuntimeError, "fixed synthetic embedding failure"):
                study.invoke_with_capture(
                    self.function, context, torch.ones(1, 2, 3),
                    torch.tensor([[0, 1]], dtype=torch.long),
                )
            self.assertIs(sys.getprofile(), prior_profile)
        finally:
            sys.setprofile(previous)

    def test_capture_keys_on_exact_code_object_not_same_function_name(self):
        def foreign_factory():
            def _align_hidden_sequence():
                Hc = torch.tensor([[999.]], dtype=torch.float32)
                return Hc
            return _align_hidden_sequence
        foreign = foreign_factory()
        def outer_factory():
            def _align_hidden_sequence(context, hidden, token_ids):
                foreign()
                return hidden+1
            return _align_hidden_sequence
        outer = outer_factory()
        returned, captured = study.invoke_with_capture(
            outer, None, torch.tensor([[2.]]), None
        )
        self.assertTrue(torch.equal(returned, torch.tensor([[3.]])))
        self.assertEqual(captured, {})

    def test_public32_root_mean_covariance_and_white_rows_match_original_locals(self):
        for label, source, vocabulary, ids in offline_author_fixtures():
            with self.subTest(fixture=label), torch.no_grad():
                context = author_context(vocabulary)
                public, parameters = study.public32_arrays(context.embedding_layer, ids)
                _, captured = study.invoke_with_capture(
                    self.function, context, torch.tensor(source).unsqueeze(0),
                    torch.tensor([ids], dtype=torch.long),
                )
                for name, local in (
                    ("mean", "mu_E"), ("covariance", "Cov_E"),
                    ("root", "sqrt_Cov_E"), ("target_white", "E_w"),
                ):
                    self.assertEqual(public[name].dtype, np.dtype("float32"))
                    self.assertEqual(public[name].shape, tuple(captured[local].shape))
                    self.assertEqual(public[name].tobytes(), captured[local].numpy().tobytes())
                self.assertEqual(parameters["target_norm"], context.target_norm)
                self.assertEqual(parameters["lam"], float(np.float32(.001)))
                self.assertEqual(parameters["anchor_gain"], float(np.float32(.3)))
                self.assertEqual(parameters["complement"], float(np.float32(1-.3)))


def tiny_model():
    torch.manual_seed(71237)
    config = GPTNeoXConfig(
        vocab_size=64, hidden_size=12, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=3,
        max_position_embeddings=64, rotary_pct=1.0,
        attention_dropout=0.0, hidden_dropout=0.0,
        bos_token_id=0, eos_token_id=1, use_cache=True,
    )
    config._attn_implementation = "eager"
    model = GPTNeoXForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


class CachedExtractionContracts(unittest.TestCase):
    def test_cached_states_predict_the_forced_token_and_match_full_causal_forward(self):
        model = tiny_model()
        prompt = [1, 2, 3]
        probe = list(range(5, 13))
        colors = list(range(20, 28))
        captured = []
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value.detach().clone())
        handle = model.gpt_neox.layers[-1].register_forward_hook(hook)
        try:
            with torch.no_grad():
                whole = model(
                    input_ids=torch.tensor([prompt+probe]),
                    attention_mask=torch.ones(1, len(prompt)+8, dtype=torch.long),
                    use_cache=False, return_dict=True,
                )
        finally:
            handle.remove()
        positions = np.arange(len(prompt)-1, len(prompt)+7)
        expected_source = captured[0][0, positions].numpy()
        expected_logits = whole.logits[0, positions].numpy()
        with patch.object(model, "forward", wraps=model.forward) as calls:
            actual = study.extract_sequence(model, prompt, probe, colors)
        self.assertEqual(calls.call_count, 8)
        self.assertEqual(actual["model_forward_calls"], 8)
        self.assertFalse(actual["sampling"])
        self.assertFalse(actual["cache_reused_across_sequences"])
        self.assertEqual(actual["source"].shape, (8, 12))
        self.assertEqual(actual["source"].dtype, np.dtype("float32"))
        np.testing.assert_allclose(actual["source"], expected_source, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            actual["first_color_logits"], expected_logits[0, colors],
            rtol=2e-5, atol=2e-6,
        )
        for step, call in enumerate(calls.call_args_list):
            ids = call.kwargs.get("input_ids")
            if ids is None:
                ids = call.args[0]
            expected_consumed = prompt if step == 0 else [probe[step-1]]
            self.assertEqual(ids.tolist(), [expected_consumed])
            self.assertEqual(actual["steps"][step]["consumed_ids"], expected_consumed)
            self.assertEqual(actual["steps"][step]["predicts_token_id"], probe[step])
            self.assertEqual(actual["steps"][step]["attention_mask_length"], len(prompt)+step)
            self.assertEqual(actual["steps"][step]["has_input_cache"], step > 0)
        self.assertEqual(actual["steps"][-1]["consumed_ids"], [probe[6]])
        logits64 = expected_logits.astype(np.float64)
        maximum = logits64.max(axis=1)
        log_norm = maximum + np.log(np.exp(logits64-maximum[:, None]).sum(axis=1))
        expected_log_prob = logits64[np.arange(8), probe]-log_norm
        np.testing.assert_allclose(actual["forced_log_probabilities"], expected_log_prob, atol=2e-6)
        self.assertAlmostEqual(actual["forced_log_probability_sum"], float(expected_log_prob.sum()), places=5)

        repeated = study.extract_sequence(model, prompt, [probe[0]]*8, colors)
        self.assertEqual(actual["source"][0].tobytes(), repeated["source"][0].tobytes())
        self.assertEqual(
            actual["first_color_logits"].tobytes(),
            repeated["first_color_logits"].tobytes(),
        )

    def test_extraction_removes_its_hook_on_a_model_exception(self):
        model = tiny_model()
        layer = model.gpt_neox.layers[-1]
        before = tuple(layer._forward_hooks)
        with patch.object(model, "forward", side_effect=RuntimeError("fixed forward failure")):
            with self.assertRaisesRegex(RuntimeError, "fixed forward failure"):
                study.extract_sequence(model, [1, 2], list(range(5, 13)), list(range(20, 28)))
        self.assertEqual(tuple(layer._forward_hooks), before)


class ImmutableEvidenceContracts(unittest.TestCase):
    def test_bundle_preserves_canonical_bits_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            writer = study.EvidenceWriter(directory)
            try:
                state = np.array(
                    [[0., 1., 2.], [-0., -1., -2.]], dtype=">f4"
                )[:, ::-1]
                ids = np.array([0, 1, 2**32-1], dtype=np.uint32)
                refs = writer.bundle(
                    "bundles/fixed_source.npz", {"state": state, "ids": ids}, "source"
                )
                path = directory / "bundles/fixed_source.npz"
                original_npz = path.read_bytes()
                original_manifest = (directory / "array_manifest.jsonl").read_bytes()
                row = json.loads(original_manifest)
                expected = np.array(state, dtype="<f4", order="C")
                expected_bytes = expected.tobytes()
                self.assertEqual(row["stage"], "source")
                self.assertEqual(row["sha256"], hashlib.sha256(original_npz).hexdigest())
                self.assertEqual(row["bytes"], len(original_npz))
                self.assertEqual(refs["state"]["dtype"], "<f4")
                self.assertEqual(refs["state"]["shape"], [2, 3])
                self.assertEqual(refs["state"]["raw_bytes"], 24)
                self.assertEqual(refs["state"]["raw_sha256"], hashlib.sha256(expected_bytes).hexdigest())
                with np.load(path, allow_pickle=False) as saved:
                    self.assertEqual(saved["state"].tobytes(), expected_bytes)
                    self.assertEqual(saved["state"].dtype.str, "<f4")
                    self.assertEqual(saved["state"].shape, (2, 3))
                    self.assertEqual(saved["ids"].tobytes(), ids.astype("<u4").tobytes())
                with self.assertRaises(FileExistsError):
                    writer.bundle("bundles/fixed_source.npz", {"state": np.ones((2, 3))}, "final")
                self.assertEqual(path.read_bytes(), original_npz)
                self.assertEqual((directory / "array_manifest.jsonl").read_bytes(), original_manifest)
                writer.bundle("bundles/fixed_final.npz", {"state": expected+1}, "final")
                self.assertEqual(path.read_bytes(), original_npz)
                rows = (directory / "array_manifest.jsonl").read_text().splitlines()
                self.assertEqual(len(rows), 2)
                self.assertEqual(json.loads(rows[1])["stage"], "final")
            finally:
                writer.close()


class CollisionAndSummaryContracts(unittest.TestCase):
    @staticmethod
    def family():
        return {
            "family_id": "F000", "question": "Which value belongs to the target?",
            "worlds": {"original": {"answer_index": 0}, "swapped": {"answer_index": 1}},
        }

    @classmethod
    def compare(cls, left_source, right_source, right_final=None):
        final = np.array([[1., 0.], [-1., 0.]], dtype=np.float32)
        return study.analyze_pair(
            cls.family(), "unique",
            {
                "original": {"source": left_source, "author_final_f32": final,
                             "inverse_status": "rejected"},
                "swapped": {"source": right_source, "author_final_f32":
                            final.copy() if right_final is None else right_final,
                            "inverse_status": "rejected"},
            },
            np.eye(2, dtype=np.float32),
        )

    def test_preexisting_signed_zero_and_new_gram_loss_are_distinct(self):
        left = np.array([[0., 1.], [-0., -1.]], dtype=np.float32)
        right = np.array([[-0., 1.], [0., -1.]], dtype=np.float32)
        preexisting = self.compare(left, right)
        self.assertTrue(preexisting["exact_source_comparison"]["numeric_values_equal"])
        self.assertFalse(preexisting["exact_source_comparison"]["raw_bits_equal"])
        self.assertTrue(preexisting["preexisting_source_collision"])
        self.assertFalse(preexisting["alignment_created_collision"])
        self.assertFalse(preexisting["exact_gram_loss_collision"])
        self.assertFalse(preexisting["channels"]["raw_state_f32"]["byte_collision"])
        self.assertTrue(preexisting["channels"]["author_f32"]["byte_collision"])
        # Rejected inverses never remove otherwise valid final bytes.
        self.assertEqual(preexisting["status"], "compared")

        source = np.array([[1., 0.], [-1., 0.]], dtype=np.float32)
        scaled = self.compare(source, 2*source)
        self.assertFalse(scaled["preexisting_source_collision"])
        self.assertTrue(scaled["alignment_created_collision"])
        self.assertTrue(scaled["exact_gram_loss_collision"])
        rotated = self.compare(source, np.array([[0., 1.], [0., -1.]], dtype=np.float32))
        self.assertTrue(rotated["alignment_created_collision"])
        self.assertFalse(rotated["exact_gram_loss_collision"])
        invalid = self.compare(source, source, np.full((2, 2), np.nan, dtype=np.float32))
        self.assertEqual(invalid["status"], "missing_or_invalid")
        self.assertFalse(invalid["channels"]["author_f32"]["valid_pair"])
        self.assertIsNone(invalid["channels"]["author_f32"]["byte_collision"])

    def test_all_collisions_enter_fixed_population_bounds_including_preexisting(self):
        source = np.array([[1., 0.], [-1., 0.]], dtype=np.float32)
        preexisting = self.compare(source, source)
        created = self.compare(source, 2*source)
        distinct = self.compare(source, 2*source, np.zeros((2, 2), dtype=np.float32))
        rows = [preexisting, created, distinct]
        for index, row in enumerate(rows):
            row["pair_id"] = f"F{index:03d}/unique"
        summary = study.collision_summary(rows)
        actual = summary["channels"]["author_f32"]
        self.assertEqual(actual["confirmed_collisions"], 2)
        self.assertEqual(actual["confirmed_distinct"], 1)
        self.assertEqual(actual["missing_or_invalid"], 61)
        self.assertIsNone(actual["full_codebook_ceiling"])
        self.assertEqual(actual["full_codebook_ceiling_bounds"], [65/128, 126/128])
        self.assertAlmostEqual(actual["observed_only_ceiling"], 2/3)
        self.assertEqual(summary["preexisting_source_collision_ids"], ["F000/unique"])
        self.assertEqual(summary["alignment_created_witnesses"], ["F001/unique"])
        self.assertEqual(summary["exact_gram_loss_witnesses"], ["F001/unique"])

        complete = study.collision_summary(rows, expected_pairs=3)["channels"]["author_f32"]
        self.assertAlmostEqual(complete["full_codebook_ceiling"], 2/3)
        for bound in complete["full_codebook_ceiling_bounds"]:
            self.assertAlmostEqual(bound, 2/3)

    @staticmethod
    def accounting_fixture():
        records = []
        for planned in study.build_plan()["planned_records"]:
            recovery = {}
            for name in ("author_matched_public32", "author_ideal_public64",
                         "reference_f64", "reference_f64_cast_f32"):
                reference_success = name == "reference_f64" and planned["probe"] == "unique"
                recovery[name] = {
                    "status": "estimated" if reference_success else "rejected",
                    "reason": None if reference_success else "scale_rank",
                    "gram_error": {"finite": True, "relative_frobenius": 0.0}
                                  if reference_success else None,
                }
            records.append({
                **planned, "status": "completed", "recovery": recovery,
                "validation": {
                    "public32_contract": True,
                    "first_step_contract": True if planned["probe"] == "repeated" else "baseline_recorded",
                    "f64_reference_eligible": planned["probe"] == "unique",
                    "f64_reference_recovery_valid": True if planned["probe"] == "unique" else None,
                },
            })
        collisions = [
            {"pair_id": f"F{f:03d}/{probe}", "probe": probe,
             "channels": {}, "exact_source_comparison": {"gram_equal": True}}
            for f in range(64) for probe in ("unique", "repeated")
        ]
        return records, collisions

    def test_complete_negative_and_partial_primary_keep_128_denominator(self):
        records, collisions = self.accounting_fixture()
        negative = study.summarize_records(records, collisions, complete=True)
        self.assertTrue(negative["study_complete"])
        self.assertTrue(negative["required_validation_passed"])
        target = negative["primary_numerical_target"]
        self.assertTrue(target["available"])
        self.assertFalse(target["met"])
        self.assertEqual(target["successes"], 0)
        self.assertEqual(target["planned_denominator"], 128)

        unique = [r for r in records if r["probe"] == "unique"]
        for row in unique[:122]:
            row["recovery"]["author_matched_public32"] = {
                "status": "estimated", "reason": None,
                "gram_error": {"finite": True, "relative_frobenius": .001},
            }
        passing = study.summarize_records(records, collisions, complete=True)
        self.assertTrue(passing["primary_numerical_target"]["met"])
        self.assertEqual(passing["primary_numerical_target"]["successes"], 122)
        partial = study.summarize_records(
            [r for r in records if r is not unique[-1]], collisions, complete=True
        )
        self.assertFalse(partial["study_complete"])
        self.assertFalse(partial["primary_numerical_target"]["available"])
        self.assertIsNone(partial["primary_numerical_target"]["met"])
        self.assertEqual(partial["primary_numerical_target"]["success_fraction_planned"], 122/128)
        status = partial["cohorts"]["unique"]["recovery"]["author_matched_public32"]["statuses"]
        self.assertEqual(status["not_observed"], 1)
        self.assertEqual(status["rejected"], 5)

    def test_exact_pair_gate_requires_each_planned_comparison_once(self):
        records, collisions = self.accounting_fixture()
        for corruption in ("missing_row", "missing_comparison", "duplicate"):
            with self.subTest(corruption=corruption):
                altered = [dict(row) for row in collisions]
                if corruption == "missing_row":
                    altered.pop()
                elif corruption == "missing_comparison":
                    altered[-1].pop("exact_source_comparison")
                else:
                    altered[-1] = dict(altered[0])
                result = study.summarize_records(records, altered, complete=True)
                check = result["required_validation"]["exact_pair_arithmetic"]
                self.assertFalse(check["passed"])
                self.assertFalse(result["required_validation_passed"])
                # A genuine validation failure is still a completed record set.
                self.assertTrue(result["study_complete"])


class RecoveryBoundaryAndFailureContracts(unittest.TestCase):
    def test_primary_inverse_receives_only_received_values_and_public_context(self):
        from latent_bridge import alignment_core as core
        from latent_bridge import information_recovery as inverse

        source = np.eye(8, 128, dtype=np.float32)
        pre = source.astype(np.float64) + .25
        final32 = np.full((8, 128), 2., dtype=np.float32)
        reference_final = np.full((8, 128), 3., dtype=np.float64)
        mean = np.full((1, 128), .25, dtype=np.float32)
        eye = np.eye(128, dtype=np.float32)
        zero_white = np.zeros((8, 128), dtype=np.float32)
        public_arrays = {"mean": mean, "covariance": eye, "root": eye,
                         "target_white": zero_white}
        metadata = lambda: {
            "reference": {"rank": 8}, "centered_reference": {"rank": 7},
            "anchor_geometry_supported": True,
        }
        context32 = SimpleNamespace(metadata=metadata, public_name="matched32")
        context64 = SimpleNamespace(metadata=metadata, public_name="ideal64")
        group = {
            "context32": context32, "context64": context64,
            "arrays32": public_arrays,
            "arrays64": {"reference": np.eye(8, 128, dtype=np.float64)},
            "parameters32": {"lam": .001}, "parameters64": {"lam": .001},
            "refs": {"public32_"+key: {"bundle": "public.npz", "member": key}
                     for key in public_arrays},
        }
        captured = {
            "Hc": torch.from_numpy(source-source.mean(axis=0, keepdims=True)),
            "H_w": torch.from_numpy(source.copy()),
            "E_w": torch.from_numpy(zero_white.copy()),
            "mu_H": torch.from_numpy(source.mean(axis=0, keepdims=True)),
            "mu_E": torch.from_numpy(mean.copy()),
            "Cov_H": torch.from_numpy(eye.copy()),
            "Cov_E": torch.from_numpy(eye.copy()),
            "R": torch.from_numpy(eye.copy()),
            "sqrt_Cov_E": torch.from_numpy(eye.copy()),
            "aligned_flat": torch.from_numpy(pre.astype(np.float32)),
            "flat": torch.from_numpy(final32.copy()),
            # Evaluation-only values; a decoder must never receive them.
            "top_ids": torch.arange(8, dtype=torch.long)+12345,
        }
        extracted = {
            "source": source, "first_color_logits": np.arange(8, dtype=np.float32),
            "color_probabilities": np.full(8, .125, dtype=np.float64),
            "forced_logits": np.zeros(8, dtype=np.float32),
            "forced_log_probabilities": np.full(8, -1., dtype=np.float64),
            "native_prediction_index": 0, "model_forward_calls": 8,
        }
        dense = SimpleNamespace(
            pre=pre, white=pre, sender_gram=np.eye(8), target_white=pre, diagnostics={}
        )
        processed = {
            "final": reference_final, "calibrated": reference_final,
            "ids": np.arange(8), "margins": np.ones(8), "target_norm": 1.,
        }
        cohort = study.build_cohort()
        planned = study.build_plan(cohort)["planned_records"][0]
        tokenization = {
            "prompt_ids": {"F000/original": [1, 2]},
            "probe_ids": {"unique": list(range(8))}, "color_ids": list(range(8)),
        }

        class RecordingWriter:
            def __init__(self, fail_final):
                self.fail_final = fail_final
                self.calls = []
            def bundle(self, path, arrays, stage):
                self.calls.append((path, tuple(arrays), stage))
                if self.fail_final and len(self.calls) == 2:
                    raise OSError("fixed final evidence failure")
                return {name: {"bundle": path, "member": name} for name in arrays}

        for fail_final in (False, True):
            with self.subTest(final_retention_failure=fail_final):
                calls = []
                # Exactly two public arguments are accepted by this sentinel.
                def no_truth_decoder(received, public_context):
                    calls.append((received.copy(), public_context))
                    return {
                        "status": "rejected", "reason": "scale_rank",
                        "gram": None, "arrays": {}, "diagnostics": {},
                    }
                writer = RecordingWriter(fail_final)
                pair_cache = {}
                with patch.object(study, "extract_sequence", return_value=extracted), \
                     patch.object(study, "invoke_with_capture",
                                  return_value=(torch.from_numpy(final32).unsqueeze(0), captured)), \
                     patch.object(core, "dense_reference", return_value=dense), \
                     patch.object(core, "postprocess", return_value=processed), \
                     patch.object(inverse, "recover_final", side_effect=no_truth_decoder):
                    row = study.process_record(
                        planned, cohort["families"][0], tokenization, None, None, None,
                        {"unique": group}, np.eye(8, 128), writer, {}, pair_cache,
                    )
                self.assertEqual(len(calls), 4)
                for (received, context), expected_value, expected_dtype, expected_context in zip(
                    calls, (2., 2., 3., 3.),
                    (np.float32, np.float32, np.float64, np.float32),
                    (context32, context64, context64, context64),
                ):
                    self.assertIs(context, expected_context)
                    self.assertEqual(received.dtype, np.dtype(expected_dtype))
                    self.assertTrue(np.all(received == expected_value))
                self.assertTrue(all(r["status"] == "rejected" for r in row["recovery"].values()))
                self.assertEqual(
                    [call[0] for call in writer.calls],
                    ["bundles/F000_original_unique_source.npz",
                     "bundles/F000_original_unique.npz"],
                )
                entry = pair_cache[("F000", "unique")]["original"]
                self.assertIn("source", entry)
                if fail_final:
                    self.assertEqual(row["status"], "error")
                    self.assertIn("source", row["arrays"])
                    self.assertNotIn("author_final_f32", entry)
                    self.assertNotIn("direct_gram_f32", entry)
                    self.assertNotIn("reference_final_cast_f32", entry)
                    self.assertNotIn("reduced_gram_f64", entry)
                    self.assertEqual(row["operational_errors"][-1]["phase"], "retention")
                else:
                    self.assertEqual(row["status"], "completed")
                    self.assertEqual(row["operational_errors"], [])
                    self.assertIn("author_final_f32", entry)
                    self.assertIn("author_final_f32", row["arrays"])


if __name__ == "__main__":
    unittest.main()
