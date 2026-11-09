"""Independent NumPy-only checks for the PMR normalization contract.

Expected Hamiltonians and normalized values are specified directly below.
These tests deliberately do not import the sparse simulator or its codec.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest
import warnings

import numpy as np


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from pmr_model import PMRModel
from value import ValueOracle


class NormalizationTests(unittest.TestCase):
    """Keep reconstruction, VALUE, and the weighted operator independent."""

    def assert_contract(
        self,
        model: PMRModel,
        expected_hamiltonian: np.ndarray,
        expected_shift: float,
        expected_alpha: float,
        expected_value: np.ndarray,
        *,
        atol: float = 1e-12,
    ) -> ValueOracle:
        self.assertAlmostEqual(model.shift, expected_shift, places=14)
        self.assertAlmostEqual(model.alpha, expected_alpha, places=14)
        np.testing.assert_allclose(
            model.hamiltonian(), expected_hamiltonian, atol=atol, rtol=0
        )
        expected_centered = expected_hamiltonian - expected_shift * np.eye(
            expected_hamiltonian.shape[0]
        )
        np.testing.assert_allclose(
            model.shifted_hamiltonian(), expected_centered, atol=atol, rtol=0
        )
        # Real-component division keeps the test reference valid for subnormals.
        expected_target = np.empty_like(expected_centered, dtype=np.complex128)
        expected_target.real = expected_centered.real / expected_alpha
        expected_target.imag = expected_centered.imag / expected_alpha
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            value = ValueOracle(model)
            actual_target = model.target_block()
        np.testing.assert_allclose(value.table(), expected_value, atol=atol, rtol=0)
        np.testing.assert_allclose(actual_target, expected_target, atol=atol, rtol=0)

        # Independently form sum_mu (b_mu/alpha) P_mu diag(VALUE_mu).
        # This is checked against the directly specified H, not model.hamiltonian().
        indices = np.arange(expected_hamiltonian.shape[0])
        identity = np.eye(len(indices), dtype=np.complex128)
        weighted_operator = np.zeros_like(identity)
        for row, mask in enumerate(model.masks):
            permutation = identity[:, indices ^ mask]
            weighted_operator += (model.bounds[row] / model.alpha) * (
                permutation @ np.diag(value.table()[row])
            )
        np.testing.assert_allclose(
            weighted_operator, expected_target, atol=atol, rtol=0
        )
        self.assertTrue(np.isfinite(actual_target).all())
        self.assertTrue(np.isfinite(value.table()).all())
        return value

    def test_x_y_z_have_independent_targets_and_values(self) -> None:
        fixtures = (
            (
                "X", (0, 1), [[0, 0], [1, 1]],
                [[0, 1], [1, 0]], [[0, 0], [1, 1]], [0, 1],
            ),
            (
                "Y", (0, 1), [[0, 0], [1j, -1j]],
                [[0, -1j], [1j, 0]], [[0, 0], [1j, -1j]], [0, 1],
            ),
            (
                "Z", (0,), [[1, -1]],
                [[1, 0], [0, -1]], [[1, -1]], [1],
            ),
        )
        for label, masks, coefficients, expected_h, expected_v, bounds in fixtures:
            with self.subTest(pauli=label):
                model = PMRModel.from_coefficients(1, masks, coefficients)
                self.assert_contract(
                    model, np.asarray(expected_h, dtype=complex), 0, 1,
                    np.asarray(expected_v, dtype=complex), atol=0,
                )
                np.testing.assert_array_equal(model.bounds, bounds)
                self.assertEqual(model.masks, masks)

    def test_unsorted_masks_keep_the_zero_row_in_place(self) -> None:
        model = PMRModel.from_coefficients(1, (1, 0), [[1, 1], [0, 0]])
        self.assertEqual(model.masks, (1, 0))
        self.assertEqual(model.move_count, 2)
        np.testing.assert_array_equal(model.bounds, [1, 0])
        self.assert_contract(
            model, np.array([[0, 1], [1, 0]], dtype=complex), 0, 1,
            np.array([[1, 1], [0, 0]], dtype=complex), atol=0,
        )

    def test_zero_coefficient_row_can_have_positive_preparation_weight(self) -> None:
        model = PMRModel.from_coefficients(
            1, (0, 1), [[0, 0], [1, 1]], bounds=[3, 2]
        )
        np.testing.assert_array_equal(model.bounds, [3, 2])
        np.testing.assert_allclose(model.bounds / model.alpha, [.6, .4], atol=0, rtol=0)
        self.assertEqual(float(np.sum(model.bounds / model.alpha)), 1)
        self.assert_contract(
            model, np.array([[0, 1], [1, 0]], dtype=complex), 0, 5,
            np.array([[0, 0], [.5, .5]], dtype=complex),
        )

    def test_inactive_nonzero_mask_is_not_pruned(self) -> None:
        model = PMRModel.from_coefficients(1, (1, 0), [[0, 0], [1, -1]])
        self.assertEqual(model.masks, (1, 0))
        np.testing.assert_array_equal(model.bounds, [0, 1])
        self.assert_contract(
            model, np.diag([1, -1]).astype(complex), 0, 1,
            np.array([[0, 0], [1, -1]], dtype=complex), atol=0,
        )

    def test_zero_centered_targets_require_explicit_positive_bounds(self) -> None:
        for scalar in (0, 1, 3, -2):
            with self.subTest(scalar=scalar):
                with self.assertRaises(ValueError) as caught:
                    PMRModel.from_coefficients(1, (0,), [[scalar, scalar]])
                message = str(caught.exception).lower()
                self.assertIn("normalization", message)
                self.assertIn("bounds", message)
                model = PMRModel.from_coefficients(
                    1, (0,), [[scalar, scalar]], bounds=[2]
                )
                self.assert_contract(
                    model, scalar * np.eye(2, dtype=complex), scalar, 2,
                    np.zeros((1, 2), dtype=complex), atol=0,
                )
                np.testing.assert_array_equal(model.bounds / model.alpha, [1])

    def test_user_shift_and_bounds_apply_after_centering(self) -> None:
        model = PMRModel.from_coefficients(
            1, (0,), [[5, 1]], shift=2, bounds=[4]
        )
        self.assert_contract(
            model, np.diag([5, 1]).astype(complex), 2, 4,
            np.array([[.75, -.25]], dtype=complex), atol=0,
        )
        with self.assertRaises(ValueError):
            PMRModel.from_coefficients(1, (0,), [[5, 1]], shift=2, bounds=[2])

    def test_complex_two_qubit_operator_is_independent(self) -> None:
        coefficients = [
            [3 + 4j, 3 - 4j, -4 + 3j, -4 - 3j],
            [3, -1, 1, 0],
            [1j, 2j, -2j, -1j],
        ]
        expected_h = np.array(
            [[3, 3 - 4j, 0, -1j],
             [3 + 4j, -1, -2j, 0],
             [0, 2j, 1, -4 - 3j],
             [1j, 0, -4 + 3j, 0]], dtype=complex,
        )
        expected_v = np.array(
            [[.6 + .8j, .6 - .8j, -.8 + .6j, -.8 - .6j],
             [1, -1, 0, -.5],
             [.5j, 1j, -1j, -.5j]], dtype=complex,
        )
        for bounds in (None, [5, 2, 2]):
            with self.subTest(bounds=bounds):
                model = PMRModel.from_coefficients(
                    2, (1, 0, 3), coefficients, bounds=bounds
                )
                self.assertEqual(model.masks, (1, 0, 3))
                np.testing.assert_array_equal(model.bounds, [5, 2, 2])
                self.assert_contract(model, expected_h, 1, 9, expected_v)

    def test_existing_example_is_preserved_to_declared_tolerance(self) -> None:
        model = PMRModel.from_coefficients(
            1, (0, 1), [[1.5, -.5], [.3 + .4j, .3 - .4j]]
        )
        self.assert_contract(
            model, np.array([[1.5, .3 - .4j], [.3 + .4j, -.5]]), .5, 1.5,
            np.array([[1, -1], [.6 + .8j, .6 - .8j]]),
        )

    def test_exact_and_loose_bounds_give_distinct_correct_normalizations(self) -> None:
        h = np.array([[1, 3 - 4j], [3 + 4j, -1]], dtype=complex)
        for bounds, alpha, expected_v in (
            ([1, 5], 6, [[1, -1], [.6 + .8j, .6 - .8j]]),
            ([2, 10], 12, [[.5, -.5], [.3 + .4j, .3 - .4j]]),
        ):
            with self.subTest(bounds=bounds):
                model = PMRModel.from_coefficients(
                    1, (0, 1), [[1, -1], [3 + 4j, 3 - 4j]], bounds=bounds
                )
                np.testing.assert_array_equal(model.bounds, bounds)
                self.assert_contract(model, h, 0, alpha, np.array(expected_v))

    def test_rounding_does_not_make_a_complex_bound_nonconservative(self) -> None:
        z = .3 + .4j
        squared_magnitude = Fraction.from_float(z.real) ** 2 + Fraction.from_float(z.imag) ** 2
        self.assertGreater(squared_magnitude, Fraction(1, 4))
        model = PMRModel.from_coefficients(1, (0, 1), [[0, 0], [z, z.conjugate()]])
        self.assertGreaterEqual(Fraction.from_float(float(model.bounds[1])) ** 2, squared_magnitude)
        with self.assertRaises(ValueError):
            PMRModel.from_coefficients(
                1, (0, 1), [[0, 0], [z, z.conjugate()]], bounds=[0, .5]
            )

    def test_nonzero_rows_cannot_use_zero_or_insufficient_bounds(self) -> None:
        for bound in (0, 1e-16):
            for atol in (1e-12, 1e-3):
                with self.subTest(bound=bound, atol=atol):
                    with self.assertRaises(ValueError):
                        PMRModel.from_coefficients(
                            1, (0, 1), [[1, -1], [1e-15, 1e-15]],
                            bounds=[1, bound], atol=atol,
                        )

    def test_tiny_nonzero_values_are_not_hidden_by_matrix_tolerance(self) -> None:
        tiny = 1e-15
        model = PMRModel.from_coefficients(
            1, (0, 1), [[1, -1], [tiny, tiny]], bounds=[1, 1]
        )
        value = self.assert_contract(
            model, np.array([[1, tiny], [tiny, -1]], dtype=complex), 0, 2,
            np.array([[1, -1], [tiny, tiny]], dtype=complex), atol=0,
        )
        self.assertEqual(model.hamiltonian()[1, 0], tiny)
        self.assertEqual(value.normalized_value(1, 0), tiny)
        self.assertEqual(model.target_block()[1, 0], tiny / 2)
        self.assertNotEqual(value.normalized_value(1, 0), 0)
        self.assertNotEqual(model.target_block()[1, 0], 0)

    def test_automatic_bounds_preserve_a_tiny_active_row(self) -> None:
        tiny = 1e-15
        model = PMRModel.from_coefficients(1, (0, 1), [[1, -1], [tiny, tiny]])
        self.assertEqual(model.bounds[1], tiny)
        self.assertEqual(ValueOracle(model).normalized_value(1, 0), 1)
        self.assertNotEqual(model.target_block()[1, 0], 0)

    def test_subnormal_model_normalizes_without_complex_division_overflow(self) -> None:
        small = 1e-320
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = PMRModel.from_coefficients(1, (0,), [[small, -small]], shift=0)
            value = self.assert_contract(
                model, np.diag([small, -small]).astype(complex), 0, small,
                np.array([[1, -1]], dtype=complex), atol=0,
            )
        self.assertEqual(model.alpha, small)
        self.assertEqual(value.normalized_value(0, 0), 1)

    def test_subnormal_complex_row_has_a_finite_conservative_bound(self) -> None:
        small = np.nextafter(0.0, 1.0)
        z = complex(small, small)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = PMRModel.from_coefficients(1, (0, 1), [[0, 0], [z, z.conjugate()]])
            value = ValueOracle(model)
        self.assertEqual(model.bounds[1], 2 * small)
        np.testing.assert_array_equal(value.table()[1], [.5 + .5j, .5 - .5j])
        np.testing.assert_array_equal(model.target_block(), [[0, .5 - .5j], [.5 + .5j, 0]])

    def test_value_normalization_underflow_to_zero_is_rejected(self) -> None:
        small = np.nextafter(0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "(?i)(underflow|range)"):
            model = PMRModel.from_coefficients(1, (0, 1), [[0, 0], [small, small]], bounds=[0, 2])
            ValueOracle(model)

    def test_total_alpha_normalization_underflow_to_zero_is_rejected(self) -> None:
        small = np.nextafter(0.0, 1.0)
        # c/b remains nonzero and both preparation weights are representable,
        # but division by total alpha=3 would erase the off-diagonal entry.
        with self.assertRaisesRegex(ValueError, "(?i)(underflow|range)"):
            model = PMRModel.from_coefficients(1, (0, 1), [[1, -1], [small, small]], bounds=[2, 1])
            model.target_block()

    def test_positive_zero_row_preparation_weight_underflow_is_rejected(self) -> None:
        small = np.nextafter(0.0, 1.0)
        # The zero coefficient row is still assigned a positive preparation
        # weight; silently losing that weight would change the declared model.
        with self.assertRaisesRegex(ValueError, "(?i)(underflow|range)"):
            PMRModel.from_coefficients(1, (0, 1), [[1, -1], [0, 0]], bounds=[2, small])

    def test_weighted_value_underflow_is_rejected_even_when_divisions_survive(self) -> None:
        small = float(np.nextafter(0.0, 1.0))
        row_bound = .7
        alpha = 1.2 + row_bound
        normalized_component = small / row_bound
        target_component = small / alpha
        weight = row_bound / alpha
        # Separate divisions are representable, but the actual weighted-VALUE
        # multiplication can still erase a component after an extra rounding.
        self.assertNotEqual(normalized_component, 0)
        self.assertNotEqual(target_component, 0)
        self.assertGreater(weight, 0)
        self.assertEqual(weight * normalized_component, 0)
        for coefficient in (complex(small, 0), complex(0, small)):
            with self.subTest(coefficient=coefficient):
                with self.assertRaisesRegex(ValueError, "(?i)(weighted VALUE|underflow)"):
                    PMRModel.from_coefficients(
                        1, (0, 1),
                        [[1, -1], [coefficient, coefficient.conjugate()]],
                        bounds=[1.2, row_bound],
                    )

    def test_large_valid_bounds_preserve_nonzero_normalized_entries(self) -> None:
        model = PMRModel.from_coefficients(1, (0, 1), [[1, -1], [1, 1]], bounds=[1e307, 1e307])
        value = ValueOracle(model)
        self.assertTrue(np.isfinite(value.table()).all())
        self.assertEqual(value.normalized_value(1, 0), 1 / 1e307)
        self.assertEqual(model.target_block()[1, 0], 1 / 2e307)
        self.assertNotEqual(model.target_block()[1, 0], 0)

    def test_representable_subnormal_normalized_values_are_supported(self) -> None:
        bound = np.finfo(float).max
        expected = 1 / bound
        self.assertGreater(expected, 0)
        self.assertLess(expected, np.finfo(float).tiny)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = PMRModel.from_coefficients(1, (0,), [[1, -1]], bounds=[bound])
            value = ValueOracle(model)
            target = model.target_block()
        np.testing.assert_array_equal(value.table(), [[expected, -expected]])
        np.testing.assert_array_equal(target, [[expected, 0], [0, -expected]])
        self.assertNotEqual(value.normalized_value(0, 0), 0)
        self.assertNotEqual(target[0, 0], 0)

    def test_nonfinite_coefficients_are_rejected(self) -> None:
        for bad in (np.nan, np.inf, -np.inf, complex(1, np.inf)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0, 1), [[1, -1], [bad, np.conjugate(bad)]])

    def test_invalid_shifts_are_rejected_before_casting(self) -> None:
        for bad in (np.nan, np.inf, -np.inf, .5 + 3j, np.complex128(.5 + 3j)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0,), [[1, -1]], shift=bad)

    def test_invalid_bounds_are_rejected_before_casting(self) -> None:
        for bad in (np.nan, np.inf, -np.inf, -1, 2 + 3j, np.complex128(2 + 3j)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0,), [[1, -1]], bounds=np.array([bad]))

    def test_value_tolerance_must_be_finite_nonnegative_and_real(self) -> None:
        model = PMRModel.from_coefficients(1, (0,), [[1, -1]])
        for bad in (-1e-12, np.nan, np.inf, -np.inf, 1e-12 + 1j, np.complex128(1e-12 + 1j)):
            with self.subTest(atol=bad):
                with self.assertRaises(ValueError):
                    ValueOracle(model, atol=bad)

    def test_total_normalization_overflow_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PMRModel.from_coefficients(1, (0, 1), [[1, -1], [1, 1]], bounds=[1e308, 1e308])

    def test_centering_overflow_is_rejected(self) -> None:
        large = np.finfo(float).max
        with self.assertRaises(ValueError):
            PMRModel.from_coefficients(1, (0,), [[large, -large]], shift=-large)

    def test_no_finite_complex_magnitude_bound_is_rejected(self) -> None:
        large = np.finfo(float).max
        z = complex(large, large)
        with self.assertRaises(ValueError):
            PMRModel.from_coefficients(1, (0, 1), [[0, 0], [z, z.conjugate()]])

    def test_constant_extreme_diagonals_center_without_overflow_or_loss(self) -> None:
        for scalar in (np.finfo(float).max, np.nextafter(0.0, 1.0)):
            with self.subTest(scalar=scalar):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    model = PMRModel.from_coefficients(1, (0,), [[scalar, scalar]], bounds=[1])
                    target = model.target_block()
                self.assertEqual(model.shift, scalar)
                np.testing.assert_array_equal(model.centered_coefficients, [[0, 0]])
                np.testing.assert_array_equal(target, np.zeros((2, 2)))

    def test_core_imports_do_not_load_quantum_or_sparse_backends(self) -> None:
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); import pmr_model, value; "
            "assert not any(n == 'qrisp' or n.startswith('qrisp.') for n in sys.modules); "
            "assert not any(n in sys.modules for n in ('prep', 'state', 'r_block', 'rotation_codec', 'main'))"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(CODE_DIRECTORY)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
