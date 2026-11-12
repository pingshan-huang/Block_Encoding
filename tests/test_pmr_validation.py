"""Structural and absolute-Hermiticity regression checks for Milestone 1B.

These tests concern the NumPy mathematical model, not sparse-state pruning
or fixed-point VALUE encoding.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import warnings

import numpy as np


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from pmr_model import PMRModel


class InputValidationTests(unittest.TestCase):
    """Check accepted PMR shapes, mask labels, and Hermiticity tolerance inputs."""

    def test_python_and_numpy_integer_labels_are_supported(self) -> None:
        """Accept Python and NumPy integers while preserving mask order and reconstruction."""
        for n_qubits in (1, np.int32(1), np.int64(1), np.uint64(1)):
            with self.subTest(n_qubits=type(n_qubits).__name__):
                model = PMRModel.from_coefficients(
                    n_qubits, (np.int64(1), np.uint32(0)),
                    [[1j, -1j], [1, -1]], shift=0,
                )
                self.assertEqual(model.masks, (1, 0))
                np.testing.assert_array_equal(
                    model.hamiltonian(), [[1, -1j], [1j, -1]]
                )

    def test_qubit_count_must_be_a_positive_integer(self) -> None:
        """Reject nonpositive counts, booleans, and non-integer types."""
        invalid = (0, -1, True, np.bool_(True), 1.0, 1.5, "1", None)
        for n_qubits in invalid:
            with self.subTest(n_qubits=n_qubits):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(n_qubits, (0,), [[1, -1]])

    def test_masks_are_not_silently_coerced_to_integers(self) -> None:
        """Reject booleans and non-integer masks instead of silently converting them."""
        invalid = (True, np.bool_(True), 1.0, 1.5, "1", 1 + 0j, None)
        for mask in invalid:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0, mask), [[1, -1], [1, 1]])

    def test_masks_must_be_nonempty_in_range_and_include_diagonal(self) -> None:
        """Require a nonempty mask list, masks within the system width, and the diagonal move."""
        fixtures = (
            ((), []),
            ((0, -1), [[1, -1], [1, 1]]),
            ((0, 2), [[1, -1], [1, 1]]),
            ((1,), [[1, 1]]),
        )
        for masks, coefficients in fixtures:
            with self.subTest(masks=masks):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, masks, coefficients)

    def test_duplicate_masks_are_unsupported_even_if_the_sum_is_hermitian(self) -> None:
        """Report repeated masks as unsupported, even when their sum is Hermitian."""
        # The two mask-1 rows individually disagree under reversal, but their
        # sum gives the Hermitian matrix [[0, 3], [3, 0]]. Representation policy,
        # rather than non-Hermiticity of that sum, is the reason for rejection.
        coefficients = np.array([[0, 0], [1, 2], [2, 1]], dtype=complex)
        summed_matrix = np.array([[0, 3], [3, 0]], dtype=complex)
        np.testing.assert_array_equal(summed_matrix, summed_matrix.conj().T)
        with self.assertRaises(ValueError) as caught:
            PMRModel.from_coefficients(1, (0, 1, 1), coefficients)
        message = str(caught.exception).lower()
        self.assertTrue("unique" in message or "duplicate" in message)
        self.assertNotIn("hermiticity failed", message)
        self.assertNotIn("non-hermitian", message)

    def test_coefficients_have_exact_row_and_system_dimensions(self) -> None:
        """Reject tables without one row per mask and one column per basis state."""
        invalid = ([], [1, -1], [[1]], [[1, -1, 0]],
                   [[1, -1], [1, -1]], [[[1, -1]]], [[1], [-1, 0]])
        for coefficients in invalid:
            with self.subTest(coefficients=coefficients):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0,), coefficients)

    def test_hermiticity_tolerance_must_be_a_finite_nonnegative_real_scalar(self) -> None:
        """Reject invalid tolerances during construction and later Hermiticity checks."""
        model = PMRModel.from_coefficients(1, (0,), [[1, -1]])
        invalid = (-1, np.nan, np.inf, -np.inf, 1j, [1e-12], [])
        for atol in invalid:
            with self.subTest(atol=atol):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, (0,), [[1, -1]], atol=atol)
                with self.assertRaises(ValueError):
                    model.validate_hermiticity(atol=atol)


class HermiticityTests(unittest.TestCase):
    """Check the absolute residual rule for diagonal and complex off-diagonal rows."""

    def test_diagonal_and_off_diagonal_use_the_same_absolute_residual(self) -> None:
        """Apply the same absolute residual boundary to diagonal and off-diagonal rows."""
        tolerance = 0.125
        # Each fixture has max |c(x xor mu) - conjugate(c(x))| = 0.125.
        # The diagonal imaginary component is half the residual.
        fixtures = (
            ((0,), [[-0.5 + 0.0625j, 0.5]], [1]),
            ((0, 1), [[-0.5, 0.5], [0.5 + 0.125j, 0.5]], [1, 1]),
        )
        for masks, coefficients, bounds in fixtures:
            with self.subTest(masks=masks):
                model = PMRModel.from_coefficients(
                    1, masks, coefficients, bounds=bounds, shift=0, atol=tolerance,
                )
                model.validate_hermiticity(atol=np.float64(tolerance))
                smaller_tolerance = np.nextafter(tolerance, 0)
                with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
                    PMRModel.from_coefficients(
                        1, masks, coefficients, bounds=bounds, shift=0,
                        atol=smaller_tolerance,
                    )
                with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
                    model.validate_hermiticity(atol=smaller_tolerance)

    def test_relative_tolerance_cannot_hide_large_amplitude_disagreement(self) -> None:
        """Use the requested absolute tolerance even when coefficients are large."""
        coefficients = [[0, 0], [1e8, 1e8 + 1]]
        # Default np.allclose would accept a residual of 1 at this magnitude.
        with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
            PMRModel.from_coefficients(1, (0, 1), coefficients)
        model = PMRModel.from_coefficients(1, (0, 1), coefficients, atol=1)
        model.validate_hermiticity(atol=1)
        with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
            model.validate_hermiticity(atol=0.5)

    def test_zero_tolerance_accepts_exact_complex_hermiticity(self) -> None:
        """Accept exact conjugate pairs at zero tolerance and reconstruct their Hamiltonian."""
        model = PMRModel.from_coefficients(
            1, (1, 0), [[3 + 4j, 3 - 4j], [2, -2]], atol=0, shift=0,
        )
        model.validate_hermiticity(atol=0)
        np.testing.assert_array_equal(
            model.hamiltonian(), [[2, 3 - 4j], [3 + 4j, -2]]
        )

    def test_nonzero_diagonal_imaginary_part_is_rejected_at_zero_tolerance(self) -> None:
        """Reject even a tiny imaginary diagonal component when the tolerance is zero."""
        with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
            PMRModel.from_coefficients(
                1, (0,), [[1 + 1e-15j, -1]], bounds=[2], atol=0,
            )

    def test_complex_residual_uses_magnitude_not_separate_component_limits(self) -> None:
        """Check the residual's magnitude, not its real and imaginary parts separately."""
        # Both residual components fit in 1, but their complex magnitude is sqrt(2).
        with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
            PMRModel.from_coefficients(
                1, (0, 1), [[0, 0], [1 + 1j, 0]], bounds=[0, 2], atol=1,
            )

    def test_overflowing_residual_is_rejected_without_runtime_warnings(self) -> None:
        """Report residual overflow as a Hermiticity error without a RuntimeWarning."""
        largest = np.finfo(float).max
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "[Hh]ermiticity"):
                PMRModel.from_coefficients(
                    1, (0, 1), [[0, 0], [largest, -largest]],
                    bounds=[0, largest], shift=0,
                )


if __name__ == "__main__":
    unittest.main()
