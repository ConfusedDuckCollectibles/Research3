"""Independent fixed-fixture tests for the public-only numerical inverse.

No model, original cohort extraction, network, timing experiment, or fitting is
performed. Closed forms and exact integer witnesses are separate from the
implementation's solver and floating-point Gram calculations.
"""
from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from latent_bridge.information_recovery import (
    PublicContext,
    array_errors,
    exact_centered_gram_pair,
    helmert,
    inverse_spectrum,
    public64,
    rank_diagnostics,
    recover_final,
    source_grams,
)


def independent_anchor(pre, public):
    """Specification-level row calibration/anchoring on tiny fixed arrays."""
    pre = np.asarray(pre, dtype=np.float64)
    norms = np.sqrt(np.sum(pre * pre, axis=1))
    calibrated = pre * (
        public.target_norm / np.maximum(norms, public.row_floor)
    )[:, None]
    table_norms = np.sqrt(np.sum(public.vocab * public.vocab, axis=1))
    directions = public.vocab / np.maximum(
        table_norms, public.cosine_floor
    )[:, None]
    ids = np.argmax(calibrated @ directions.T, axis=1)
    final = public.complement * calibrated + public.anchor_gain * public.vocab[ids]
    return final, calibrated, ids


def two_row_case(g=2.0, rho=1.0, lam=1e-3, extra_zero=False):
    """Closed form: R=rho*(e1,e2), source reduced Gram=(g)."""
    table = rho * np.array([[1., 0., 0.], [0., 1., 0.]])
    if extra_zero:
        table = np.vstack((table, np.zeros(3)))
    public = public64(table, [0, 1], lam=lam)
    y = rho * rho / (rho * rho / 2.0 + lam)
    x = g / (g / 2.0 + lam)
    m = math.sqrt(x / y)
    pre = rho * np.array([
        [(1.0 + m) / 2.0, (1.0 - m) / 2.0, 0.0],
        [(1.0 - m) / 2.0, (1.0 + m) / 2.0, 0.0],
    ])
    final, calibrated, ids = independent_anchor(pre, public)
    return public, pre, final, calibrated, ids


class InverseAlgebraContracts(unittest.TestCase):
    def test_closed_form_full_rank_round_trip_and_input_immutability(self):
        public, pre, final, calibrated, ids = two_row_case(g=2.0)
        before = final.tobytes()
        result = recover_final(final, public)
        self.assertEqual(result["status"], "estimated", result["reason"])
        self.assertIsNone(result["reason"])
        np.testing.assert_allclose(result["gram"], [[2.0]], rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(result["arrays"]["pre"], pre, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(result["arrays"]["calibrated"], calibrated, atol=2e-15)
        np.testing.assert_array_equal(result["arrays"]["anchor_ids"], ids)
        self.assertTrue(result["diagnostics"]["not_a_fidelity_claim"])
        self.assertEqual(final.tobytes(), before)


    def test_nondiagonal_matrix_gram_round_trip_in_fixed_three_row_basis(self):
        # Broad any-orthogonal-alignment identity, without a production forward
        # or a fitted Procrustes optimum. Two independent source eigendirections
        # exercise the off-diagonal inverse and the Helmert column order.
        table = np.zeros((3, 5))
        table[0, 0], table[1, 1], table[2, 2] = 1., 2., 3.
        public = public64(table, [0, 1, 2], lam=1.0)
        h = np.array([
            [1/math.sqrt(2), 1/math.sqrt(6)],
            [-1/math.sqrt(2), 1/math.sqrt(6)],
            [0., -2/math.sqrt(6)],
        ])
        rotation = np.array([[math.sqrt(3)/2, -.5], [.5, math.sqrt(3)/2]])
        feature_rows = np.zeros((2, 5))
        feature_rows[0, 3], feature_rows[1, 4] = 1., 1.
        # f(2)=6/5 and f(6)=2 when K=3, lambda=1.
        white = h @ rotation @ np.diag(np.sqrt([6/5, 2.])) @ feature_rows
        pre = white @ public.root + public.mean
        final, _, _ = independent_anchor(pre, public)
        result = recover_final(final, public)
        self.assertEqual(result["status"], "estimated", result["reason"])
        expected = np.array([[3., -math.sqrt(3)], [-math.sqrt(3), 5.]])
        np.testing.assert_allclose(result["gram"], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(result["arrays"]["pre"], pre, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(result["arrays"]["anchor_ids"], [2, 2, 2])

    def test_active_row_clamp_still_recovers_scales_from_known_mean(self):
        public, pre, final, calibrated, _ = two_row_case(g=4e-16, rho=1e-8)
        self.assertTrue(np.all(np.linalg.norm(pre, axis=1) < public.row_floor))
        self.assertTrue(
            np.all(np.linalg.norm(calibrated, axis=1) < public.target_norm)
        )
        result = recover_final(final, public)
        self.assertEqual(result["status"], "estimated", result["reason"])
        np.testing.assert_allclose(result["gram"], [[4e-16]], rtol=2e-10, atol=1e-27)
        np.testing.assert_allclose(
            result["arrays"]["scales"],
            np.full(2, public.row_floor / public.target_norm),
            rtol=2e-12, atol=2e-12,
        )

    def test_zero_vocabulary_row_is_supported_but_nonzero_tiny_row_is_not(self):
        public, _, final, _, _ = two_row_case(extra_zero=True)
        result = recover_final(final, public)
        self.assertEqual(result["status"], "estimated", result["reason"])
        np.testing.assert_allclose(result["gram"], [[2.0]], rtol=2e-10, atol=1e-12)
        tiny_table = np.vstack((np.eye(3)[:2], [0., 0., 5e-13]))
        tiny = public64(tiny_table, [0, 1])
        rejected = recover_final(final, tiny)
        self.assertEqual(rejected["status"], "unsupported")
        self.assertEqual(rejected["reason"], "tiny_vocabulary_rows")
        self.assertIsNone(rejected["gram"])

    def test_centered_reference_normalization_collision_is_not_repaired(self):
        public = public64(np.array([[1., 0., 0.], [-1., 0., 0.]]), [0, 1], lam=1.0)
        first = np.array([[1., 0., 0.], [-1., 0., 0.]])
        second = first / 2.0
        f1, _, _ = independent_anchor(first, public)
        f2, _, _ = independent_anchor(second, public)
        self.assertEqual(f1.tobytes(), f2.tobytes())
        # Both are attainable: Y=1, M=1 or 1/2, giving source G=2 or 2/7.
        self.assertNotEqual(2.0, 2.0 / 7.0)
        result = recover_final(f1, public)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "scale_rank")
        self.assertIsNone(result["gram"])
        self.assertIn("scales", result["arrays"])
        self.assertNotIn("white_gram", result["arrays"])

    def test_anchor_inconsistency_and_nonpositive_scales_reject(self):
        public = public64(np.eye(3)[:2], [0, 1])
        cases = [
            (np.array([[.2, .1, 0.], [0., 1., 0.]]), "anchor_self_consistency"),
            (np.array([[-.7, .3, 0.], [0., 1., 0.]]), "nonpositive_scales"),
        ]
        for final, expected in cases:
            with self.subTest(reason=expected):
                result = recover_final(final, public)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["reason"], expected)
                self.assertIsNone(result["gram"])
                self.assertIn("calibrated", result["arrays"])

    def test_out_of_range_white_gram_stays_a_domain_failure(self):
        public = public64(np.eye(3)[:2], [0, 1], lam=1.0)
        # Y=2/3, M=sqrt(3/(2/3)) produces X=3, outside K=2.
        m = math.sqrt(4.5)
        pre = np.array([
            [(1+m)/2, (1-m)/2, 0.],
            [(1-m)/2, (1+m)/2, 0.],
        ])
        final, _, _ = independent_anchor(pre, public)
        result = recover_final(final, public)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "spectral_domain")
        self.assertIsNone(result["gram"])
        self.assertGreater(result["arrays"]["white_eigenvalues"][0], 2.0)
        self.assertIn("pre", result["arrays"])

    def test_actual_public_root_is_solved_without_resymmetrization(self):
        base = public64(np.eye(3)[:2], [0, 1], lam=1.0, alpha=0.0)
        root = np.array([[1., .2, 0.], [0., 1., 0.], [0., 0., 1.]])
        public = PublicContext(
            base.vocab, base.probe_ids, base.mean, root, base.target_norm,
            1.0, 0.0, 1.0,
        )
        # A deliberately nonsymmetric declared public root isolates solve order.
        # H and the row below are specified independently of production helmert().
        h = np.array([[1.], [-1.]]) / math.sqrt(2.0)
        white = h @ np.array([[math.sqrt(.5), -math.sqrt(.5), 0.]])
        pre = white @ root + base.mean
        final, _, _ = independent_anchor(pre, public)
        result = recover_final(final, public)
        self.assertEqual(result["status"], "estimated", result["reason"])
        np.testing.assert_allclose(result["arrays"]["rewhitened"], white, atol=2e-14)
        np.testing.assert_allclose(result["gram"], [[2.0]], rtol=2e-12, atol=2e-12)
        wrong = np.linalg.solve(((root+root.T)/2).T, (pre-base.mean).T).T
        self.assertGreater(np.max(np.abs(wrong-white)), 1e-3)

    def test_inverse_spectrum_exact_endpoints_and_no_clipping(self):
        values, reason = inverse_spectrum(np.array([0., .5, 1., 1.5]), 2, 1.0)
        self.assertIsNone(reason)
        np.testing.assert_allclose(values, [0., 2/3, 2., 6.], rtol=0, atol=1e-15)
        near = np.array([np.nextafter(2., 0.)])
        estimate, reason = inverse_spectrum(near, 2, 1.0)
        self.assertIsNone(reason)
        self.assertEqual(estimate[0], float(2**54-2))
        for bad, expected in [
            (np.array([np.nextafter(0., -1.)]), "spectral_domain"),
            (np.array([2.]), "spectral_domain"),
            (np.array([3.]), "spectral_domain"),
            (np.array([np.nan]), "nonfinite_spectrum"),
            (np.array([np.inf]), "nonfinite_spectrum"),
        ]:
            with self.subTest(value=bad, reason=expected):
                before = bad.tobytes()
                estimate, reason = inverse_spectrum(bad, 2, 1.0)
                self.assertIsNone(estimate)
                self.assertEqual(reason, expected)
                self.assertEqual(bad.tobytes(), before)
        self.assertEqual(
            inverse_spectrum(np.array([.5]), 2, .25, 1.0),
            (None, "unsupported_covariance_floor_branch"),
        )
        estimate, reason = inverse_spectrum(np.array([1.]), 2, 1e308)
        self.assertIsNone(estimate)
        self.assertEqual(reason, "nonfinite_reconstruction")

    def test_strict_rank_cutoff_and_zero_reference_errors(self):
        threshold = np.finfo(np.float64).eps * 3
        result = rank_diagnostics(np.diag([1., threshold, 2*threshold]))
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["cutoff"], threshold)
        zero = array_errors(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual(zero["relative_frobenius"], 0.0)
        nonzero = array_errors(np.eye(2), np.zeros((2, 2)))
        self.assertFalse(nonzero["finite"])
        self.assertIsNone(nonzero["relative_frobenius"])
        self.assertEqual(nonzero["reason"], "zero_reference_norm")
        underflow = array_errors(np.array([1e-200]), np.array([0.]))
        self.assertEqual(underflow["absolute_frobenius"], 0.0)
        self.assertIsNone(underflow["relative_frobenius"])
        self.assertFalse(underflow["finite"])
        self.assertEqual(underflow["reason"], "zero_reference_norm")

    def test_public_context_is_owned_immutable_and_truth_is_not_an_input(self):
        table = np.eye(3)[:2].copy()
        public = public64(table, [0, 1])
        expected = public.vocab.copy()
        table[:] = 99
        np.testing.assert_array_equal(public.vocab, expected)
        for array in (public.vocab, public.root, public.mean, public.directions):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertEqual(tuple(inspect.signature(recover_final).parameters), ("final", "public"))
        for forbidden in ("source", "true_gram", "true_pre", "true_anchor_ids", "answer"):
            with self.subTest(forbidden=forbidden), self.assertRaises(TypeError):
                recover_final(np.eye(3)[:2], public, **{forbidden: np.zeros((2, 3))})


class ExactGramContracts(unittest.TestCase):
    def test_orientation_and_translation_change_values_but_preserve_exact_gram(self):
        left = np.array([[1., 2., 0.], [-1., -2., 0.]], dtype=np.float32)
        right = np.array([[2., -1., 0.], [-2., 1., 0.]], dtype=np.float32) + 8
        result = exact_centered_gram_pair(left, right)
        self.assertFalse(result["numeric_values_equal"])
        self.assertFalse(result["raw_bits_equal"])
        self.assertTrue(result["gram_equal"])
        self.assertEqual(result["left_gram_integer_sha256"], result["right_gram_integer_sha256"])

    def test_signed_zero_does_not_establish_source_information_loss(self):
        left = np.array([[0., 1.], [-0., -1.]], dtype=np.float32)
        right = np.array([[-0., 1.], [0., -1.]], dtype=np.float32)
        result = exact_centered_gram_pair(left, right)
        self.assertTrue(result["numeric_values_equal"])
        self.assertFalse(result["raw_bits_equal"])
        self.assertTrue(result["gram_equal"])

    def test_integer_witness_detects_difference_hidden_by_float64_accumulation(self):
        left = np.array([[2.**60, 1.], [-2.**60, -1.]], dtype=np.float32)
        right = np.array([[2.**60, 2.], [-2.**60, -2.]], dtype=np.float32)
        a = left.astype(np.float64)
        b = right.astype(np.float64)
        self.assertTrue(np.array_equal(a @ a.T, b @ b.T))
        exact = exact_centered_gram_pair(left, right)
        self.assertFalse(exact["gram_equal"])
        self.assertFalse(exact["numeric_values_equal"])
        self.assertNotEqual(exact["left_gram_integer_sha256"], exact["right_gram_integer_sha256"])

    def test_subnormal_and_invalid_exact_inputs_are_not_downcast_or_lost(self):
        tiny = np.array([1], dtype=np.uint32).view(np.float32)[0]
        left = np.array([[tiny], [-tiny]], dtype=np.float32)
        right = np.zeros((2, 1), dtype=np.float32)
        exact = exact_centered_gram_pair(left, right)
        self.assertFalse(exact["gram_equal"])
        self.assertEqual(exact["common_exponent"], -149)
        self.assertEqual(exact["denominator_power_of_two"], 298)
        self.assertEqual(exact["denominator_k_squared"], 4)
        for bad in (
            left.astype(np.float64),
            np.array([[np.nan], [0.]], dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
        ):
            with self.subTest(shape=bad.shape, dtype=str(bad.dtype)), self.assertRaises(ValueError):
                exact_centered_gram_pair(left, bad)

    def test_evaluation_gram_preserves_centered_geometry(self):
        source = np.array([[3., 4., 1.], [1., 0., 1.]], dtype=np.float32)
        result = source_grams(source)
        np.testing.assert_array_equal(result["centered"], [[1., 2., 0.], [-1., -2., 0.]])
        np.testing.assert_array_equal(result["full"], [[5., -5.], [-5., 5.]])
        np.testing.assert_allclose(result["reduced"], [[10.]], rtol=1e-15, atol=1e-15)
        h = helmert(2)
        np.testing.assert_allclose(h @ result["reduced"] @ h.T, result["full"], atol=2e-15)


if __name__ == "__main__":
    unittest.main()
