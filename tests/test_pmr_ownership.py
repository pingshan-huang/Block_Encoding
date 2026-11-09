"""Public-API ownership regression checks for Milestone 1C.

Each mutation either fails or changes an independent result. No test attempts
to bypass the API through private attributes or arbitrary Python internals.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from pmr_model import PMRModel
from value import ValueOracle


class OwnershipTests(unittest.TestCase):
    def fixture(self) -> tuple[PMRModel, ValueOracle]:
        model = PMRModel.from_coefficients(
            1, (1, 0), [[3 + 4j, 3 - 4j], [2, -2]],
            bounds=[5, 2], shift=0,
        )
        return model, ValueOracle(model)

    def assert_unchanged(self, model: PMRModel, oracle: ValueOracle) -> None:
        expected_coefficients = np.array([[3 + 4j, 3 - 4j], [2, -2]])
        expected_hamiltonian = np.array([[2, 3 - 4j], [3 + 4j, -2]])
        expected_value = np.array([[0.6 + 0.8j, 0.6 - 0.8j], [1, -1]])
        self.assertEqual(model.n_qubits, 1)
        self.assertEqual(model.masks, (1, 0))
        self.assertEqual(model.shift, 0)
        self.assertEqual(model.alpha, 7)
        np.testing.assert_array_equal(model.coefficients, expected_coefficients)
        np.testing.assert_array_equal(model.centered_coefficients, expected_coefficients)
        np.testing.assert_array_equal(model.bounds, [5, 2])
        np.testing.assert_array_equal(model.hamiltonian(), expected_hamiltonian)
        np.testing.assert_array_equal(model.shifted_hamiltonian(), expected_hamiltonian)
        np.testing.assert_allclose(
            model.target_block(), expected_hamiltonian / 7, atol=1e-15, rtol=0,
        )
        np.testing.assert_allclose(
            model.normalized_coefficients(), expected_value, atol=1e-15, rtol=0,
        )
        np.testing.assert_allclose(oracle.table(), expected_value, atol=1e-15, rtol=0)
        for row in range(2):
            for x in range(2):
                self.assertLessEqual(
                    abs(oracle.normalized_value(row, x) - expected_value[row, x]),
                    1e-15,
                )
        # A new oracle must agree with the cache made before the attempted write.
        np.testing.assert_array_equal(ValueOracle(model).table(), oracle.table())

    def attempt_array_mutation(self, array: np.ndarray) -> None:
        try:
            array.setflags(write=True)
        except ValueError:
            return
        try:
            array[...] = 99
        except (ValueError, TypeError):
            # Read-only views are also an acceptable public result.
            pass

    def test_factory_takes_ownership_without_changing_caller_flags(self) -> None:
        coefficients = np.array([[3 + 4j, 3 - 4j], [2, -2]])
        bounds = np.array([5.0, 2.0])
        masks = [1, 0]
        model = PMRModel.from_coefficients(
            1, masks, coefficients, bounds=bounds, shift=0,
        )
        oracle = ValueOracle(model)
        self.assertTrue(coefficients.flags.writeable)
        self.assertTrue(bounds.flags.writeable)
        coefficients[:] = 0
        bounds[:] = 100
        masks.reverse()
        self.assert_unchanged(model, oracle)

    def test_direct_initializer_takes_ownership_of_all_array_arguments(self) -> None:
        coefficients = np.array([[3 + 4j, 3 - 4j], [2, -2]])
        centered = coefficients.copy()
        bounds = np.array([5.0, 2.0])
        masks = [1, 0]
        shift = np.array(0.0)
        model = PMRModel(1, masks, coefficients, centered, bounds, shift)
        oracle = ValueOracle(model)
        self.assertTrue(coefficients.flags.writeable)
        self.assertTrue(centered.flags.writeable)
        self.assertTrue(bounds.flags.writeable)
        coefficients[:] = 7
        centered[:] = 8
        bounds[:] = 9
        shift[...] = 10
        masks[:] = [0, 1]
        self.assert_unchanged(model, oracle)

    def test_read_only_constructor_inputs_keep_their_original_flags(self) -> None:
        for direct in (False, True):
            with self.subTest(direct=direct):
                coefficients = np.array([[3 + 4j, 3 - 4j], [2, -2]])
                centered = coefficients.copy()
                bounds = np.array([5.0, 2.0])
                for array in (coefficients, centered, bounds):
                    array.setflags(write=False)
                if direct:
                    model = PMRModel(1, (1, 0), coefficients, centered, bounds, 0)
                else:
                    model = PMRModel.from_coefficients(
                        1, (1, 0), coefficients, bounds=bounds, shift=0,
                    )
                oracle = ValueOracle(model)
                for array in (coefficients, centered, bounds):
                    self.assertFalse(array.flags.writeable)
                    array.setflags(write=True)
                    array[...] = 0
                self.assert_unchanged(model, oracle)

    def test_strided_constructor_views_do_not_alias_the_model(self) -> None:
        for direct in (False, True):
            with self.subTest(direct=direct):
                coefficient_storage = np.zeros((2, 4), dtype=complex)
                coefficient_storage[:, ::2] = [[3 + 4j, 3 - 4j], [2, -2]]
                centered_storage = coefficient_storage.copy()
                bound_storage = np.array([5.0, 0, 2, 0])
                coefficients = coefficient_storage[:, ::2]
                centered = centered_storage[:, ::2]
                bounds = bound_storage[::2]
                for array in (coefficients, centered, bounds):
                    array.setflags(write=False)
                if direct:
                    model = PMRModel(1, (1, 0), coefficients, centered, bounds, 0)
                else:
                    model = PMRModel.from_coefficients(
                        1, (1, 0), coefficients, bounds=bounds, shift=0,
                    )
                oracle = ValueOracle(model)
                for array in (coefficients, centered, bounds):
                    self.assertFalse(array.flags.writeable)
                coefficient_storage[:] = 0
                centered_storage[:] = 0
                bound_storage[:] = 0
                self.assert_unchanged(model, oracle)

    def test_exposed_arrays_remain_independent_after_enabling_writes(self) -> None:
        model, oracle = self.fixture()
        for name in ("coefficients", "centered_coefficients", "bounds"):
            with self.subTest(name=name):
                self.attempt_array_mutation(getattr(model, name))
                self.assert_unchanged(model, oracle)

    def test_views_of_exposed_arrays_cannot_change_model_storage(self) -> None:
        model, oracle = self.fixture()
        transforms = {
            "view": lambda array: array.view(),
            "slice": lambda array: array[:1],
            "real": lambda array: array.real,
            "imag": lambda array: array.imag,
            "asarray": np.asarray,
            "flattened_view": lambda array: array.reshape(-1),
        }
        for name in ("coefficients", "centered_coefficients", "bounds"):
            for kind, transform in transforms.items():
                with self.subTest(name=name, kind=kind):
                    self.attempt_array_mutation(transform(getattr(model, name)))
                    self.assert_unchanged(model, oracle)

    def test_exposed_array_metadata_changes_cannot_change_model_shapes_or_dtypes(self) -> None:
        model, oracle = self.fixture()
        for name in ("coefficients", "centered_coefficients", "bounds"):
            with self.subTest(name=name):
                exposed = getattr(model, name)
                try:
                    exposed.shape = (exposed.size,)
                    exposed.dtype = np.uint8
                except (ValueError, TypeError):
                    pass
                self.assert_unchanged(model, oracle)
                expected_dtype = np.float64 if name == "bounds" else np.complex128
                self.assertEqual(getattr(model, name).dtype, expected_dtype)

    def test_returned_matrices_and_normalized_arrays_are_independent(self) -> None:
        model, oracle = self.fixture()
        for name in ("hamiltonian", "shifted_hamiltonian", "target_block",
                     "normalized_coefficients"):
            with self.subTest(name=name):
                returned = getattr(model, name)()
                self.attempt_array_mutation(returned)
                self.assert_unchanged(model, oracle)

    def test_value_table_and_its_views_cannot_change_the_oracle_cache(self) -> None:
        model, oracle = self.fixture()
        for transform in (lambda array: array, lambda array: array.view(),
                          lambda array: array.real, lambda array: array.imag):
            self.attempt_array_mutation(transform(oracle.table()))
            self.assert_unchanged(model, oracle)
        scalar = oracle.normalized_value(0, 0)
        self.assertIsInstance(scalar, complex)
        scalar += 100
        self.assert_unchanged(model, oracle)

    def test_value_model_access_does_not_expose_writable_model_storage(self) -> None:
        model, oracle = self.fixture()
        for name in ("coefficients", "centered_coefficients", "bounds"):
            with self.subTest(name=name):
                self.attempt_array_mutation(getattr(oracle.model, name))
                self.assert_unchanged(model, oracle)

    def test_public_model_fields_cannot_be_rebound(self) -> None:
        model, oracle = self.fixture()
        replacements = {
            "n_qubits": 2,
            "masks": (0, 1),
            "coefficients": np.zeros((2, 2)),
            "centered_coefficients": np.zeros((2, 2)),
            "bounds": np.array([10, 10]),
            "shift": 1,
        }
        for name, replacement in replacements.items():
            with self.subTest(name=name):
                with self.assertRaises((AttributeError, TypeError)):
                    setattr(model, name, replacement)
                self.assert_unchanged(model, oracle)

    def test_value_model_and_tolerance_cannot_be_rebound(self) -> None:
        model, oracle = self.fixture()
        other_model = PMRModel.from_coefficients(1, (0,), [[1, -1]])
        for name, replacement in (("model", other_model), ("atol", 10)):
            with self.subTest(name=name):
                with self.assertRaises((AttributeError, TypeError)):
                    setattr(oracle, name, replacement)
                self.assert_unchanged(model, oracle)

    def test_dataclass_replace_retains_public_fields_and_owns_replacements(self) -> None:
        model, oracle = self.fixture()
        clone = replace(model)
        self.assertIsNot(clone, model)
        clone_oracle = ValueOracle(clone)
        for name in ("coefficients", "centered_coefficients", "bounds"):
            self.attempt_array_mutation(getattr(clone, name))
        self.assert_unchanged(clone, clone_oracle)
        self.assert_unchanged(model, oracle)

        replacement_bounds = np.array([10.0, 2.0])
        changed = replace(model, bounds=replacement_bounds)
        self.assertTrue(replacement_bounds.flags.writeable)
        replacement_bounds[:] = 100
        np.testing.assert_array_equal(changed.bounds, [10, 2])
        self.assertEqual(changed.alpha, 12)
        np.testing.assert_allclose(
            ValueOracle(changed).table(), [[0.3 + 0.4j, 0.3 - 0.4j], [1, -1]],
            atol=1e-15, rtol=0,
        )
        np.testing.assert_allclose(
            changed.target_block(), np.array([[2, 3 - 4j], [3 + 4j, -2]]) / 12,
            atol=1e-15, rtol=0,
        )
        self.assert_unchanged(model, oracle)

    def test_dataclass_asdict_keeps_public_schema_and_returns_independent_arrays(self) -> None:
        model, oracle = self.fixture()
        expected_names = ["n_qubits", "masks", "coefficients",
                          "centered_coefficients", "bounds", "shift"]
        self.assertEqual([field.name for field in fields(model)], expected_names)
        exported = asdict(model)
        self.assertEqual(list(exported), expected_names)
        reconstructed = PMRModel(**exported)
        reconstructed_oracle = ValueOracle(reconstructed)
        for name in ("coefficients", "centered_coefficients", "bounds"):
            self.attempt_array_mutation(exported[name])
        exported["masks"] = (0, 1)
        exported["shift"] = 100
        self.assert_unchanged(model, oracle)
        self.assert_unchanged(reconstructed, reconstructed_oracle)

    def test_model_identity_equality_retains_dataclass_behavior(self) -> None:
        model, _ = self.fixture()
        self.assertIs(model == model, True)
        self.assertIs(model != model, False)
        self.assertIs(model == object(), False)
        self.assertIs(model != object(), True)


if __name__ == "__main__":
    unittest.main()
