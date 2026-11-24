"""Focused numerical and failure-retention gates not duplicated by codec tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from latent_bridge.alignment_core import (
    ALGORITHMS, Unsupported, canonical_alignment, helmert, postprocess)
from latent_bridge.alignment_numerical import Study, case, timing_coverage


def whiten_direct(rows, lam, eps):
    centered = rows - rows.mean(axis=0)
    covariance = centered.T @ centered / len(rows) + lam * np.eye(rows.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    inverse = (vectors * np.maximum(values, eps) ** -0.5) @ vectors.T
    return centered @ inverse


class AlignmentFocusedTests(unittest.TestCase):
    def test_floored_dewhite_matches_explicit_feature_covariance(self):
        u = helmert(3)
        a = np.zeros((2, 5))
        a[0, 3], a[1, 4] = np.sqrt(2.0), np.sqrt(0.5)
        b = np.zeros((2, 5))
        b[0, 0], b[1, 1] = np.sqrt(3e-7), np.sqrt(9e-6)
        s, r = u @ a, u @ b + np.array([0.2, -0.1, 0.3, 0.0, 0.0])
        for lam in (0.0, 1e-8, 1e-6):
            with self.subTest(lam=lam):
                value = canonical_alignment(s, r, lam=lam, eps=1e-6)
                rc = r - r.mean(axis=0)
                covariance = rc.T @ rc / len(r) + lam * np.eye(5)
                eig, vec = np.linalg.eigh(covariance)
                square_root = (vec * np.sqrt(np.maximum(eig, 1e-6))) @ vec.T
                expected = value.white @ square_root + r.mean(axis=0)
                np.testing.assert_allclose(value.pre, expected, rtol=1e-8, atol=1e-10)

    def test_cross_rank_zero_with_nonzero_larger_receiver_rank(self):
        u = helmert(4)
        s = u[:, :1] @ np.array([[0., 0, 0, 0, 2.]])
        r = u[:, 1:3] @ np.array([[1., 0, 0, 0, 0], [0, 1., 0, 0, 0]])
        value = canonical_alignment(s, r, lam=1e-3, eps=1e-6)
        self.assertEqual(value.diagnostics["source"]["rank"], 1)
        self.assertEqual(value.diagnostics["receiver"]["rank"], 2)
        self.assertEqual(value.diagnostics["cross"]["rank"], 0)
        self.assertFalse(value.diagnostics["unique"])
        sw = whiten_direct(s, 1e-3, 1e-6)
        rw = whiten_direct(r, 1e-3, 1e-6)
        np.testing.assert_allclose(value.white @ value.white.T, sw @ sw.T,
                                   rtol=1e-8, atol=1e-10)
        optimum = np.linalg.svd(sw.T @ rw, compute_uv=False).sum()
        self.assertAlmostEqual(float(np.sum(value.white * rw)), float(optimum), delta=1e-10)

    def test_required_failure_and_postprocess_exception_remain_recorded(self):
        u = helmert(3)
        s = u @ np.array([[2., 0.3, 0, 0, 0], [0.1, 1., 0, 0, 0]])
        r = u @ np.array([[1., 0, 1., 0, 0], [0.2, 0, 0, 1., 0]])
        vocab = np.vstack([r, np.eye(5), -np.eye(5)])
        fixture = case("unit-required", "unit", s, r, vocab,
                       np.arange(3, dtype=np.uint32), 2, 2, 2, 0)
        with tempfile.TemporaryDirectory() as directory:
            study = Study(Path(directory))
            try:
                def unsupported(*args, **kwargs):
                    raise Unsupported("injected_required_guard")
                with patch.dict(ALGORITHMS, {"dual": unsupported}):
                    row, results, _ = study.evaluate(fixture)
                self.assertEqual(row["status"], "failed")
                self.assertNotIn("dual", results)
                self.assertIn("dual_required_coverage_or_expected_rejection",
                              row["required_check_failures"])
                self.assertEqual(study.support[("unit", "dual", "unsupported")], 1)
                self.assertEqual(study.support[("unit", "dual", "supported")], 0)
            finally:
                study.close()
        with tempfile.TemporaryDirectory() as directory:
            study = Study(Path(directory))
            calls = 0
            try:
                def fail_first_postprocess(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise ValueError("injected_postprocessing_failure")
                    return postprocess(*args, **kwargs)
                with patch("latent_bridge.alignment_numerical.postprocess",
                           side_effect=fail_first_postprocess):
                    row, results, processed = study.evaluate(fixture)
                self.assertEqual(row["status"], "failed")
                self.assertNotIn("dense", results)
                self.assertNotIn("dense", processed)
                self.assertEqual(row["methods"]["dense"]["status"], "error")
                self.assertEqual(study.support[("unit", "dense", "error")], 1)
                self.assertEqual(study.support[("unit", "dense", "supported")], 0)
                self.assertIn("thin", results)  # Other safe independent calls continued.
            finally:
                study.close()

    def test_timing_gate_rejects_missing_duplicate_and_wrong_order(self):
        phases = [
            ("core", ("dense", "thin", "dual", "canonical")),
            ("postprocess", ("common",)),
            ("gram_encode", ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32")),
            ("gram_decode", ("dual/float64", "dual/float32", "canonical/float64", "canonical/float32")),
            ("direct_encode", ("float64", "float32")),
            ("direct_decode", ("float64", "float32")),
            ("token_encode", ("uint32_ids",)),
            ("token_decode", ("embedding_lookup",)),
        ]
        rows = []
        for phase, names in phases:
            for repetition in range(-1, 8):
                rotation = 0 if repetition == -1 else repetition % len(names)
                ordered = names[rotation:] + names[:rotation]
                for position, name in enumerate(ordered):
                    rows.append({"case_id": "T-unit", "phase": phase, "name": name,
                                 "repetition": repetition, "order_position": position,
                                 "duration_ns": 100, "error": None})
        prepared = [({"id": "T-unit"}, {}, {}, True)]
        self.assertEqual(len(rows), 171)
        self.assertTrue(timing_coverage(prepared, rows)["passed"])
        self.assertFalse(timing_coverage(prepared, rows[:-1])["passed"])
        self.assertFalse(timing_coverage(prepared, rows + [rows[0]])["passed"])
        wrong_order = [dict(row) for row in rows]
        wrong_order[0]["order_position"] = 1
        self.assertFalse(timing_coverage(prepared, wrong_order)["passed"])
        failed = [dict(row) for row in rows]
        failed[-1]["error"] = "injected"
        self.assertFalse(timing_coverage(prepared, failed)["passed"])


if __name__ == "__main__":
    unittest.main()
