"""Check that Antonio's modular interface uses the reviewed shared PMR core.

These bounded integration checks retain the original modular test suite and
add independent Hamiltonian references. Full-unitary comparison uses only
seven qubits; the larger example uses nine-qubit statevectors. No Aer backend
or sparse-simulator pruning is involved.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from qiskit.quantum_info import Operator


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = ROOT_DIRECTORY / "code"
MODULAR_DIRECTORY = ROOT_DIRECTORY / "Block Encoding Circuit"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

from pmr_model import PMRModel
from qiskit_oracles import QiskitAnglePhaseCodec, QiskitRTable
from value import ValueOracle


def _load_module(name: str, path: Path):
    """Register a uniquely named module before evaluating its dataclasses."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load test entry point {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temporary_module_aliases(names: tuple[str, ...]):
    """Restore every flat module name so the two main.py files cannot collide."""
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_interfaces():
    """Load both original entry points without changing other tests' imports."""
    dependencies = (
        "_shared_core", "model", "value_oracle", "angle_phase_codec",
        "r_oracle", "prep_oracle", "move_oracle", "flag_rotation_oracle",
    )
    with _temporary_module_aliases(("main", *dependencies)):
        core_main = _load_module(
            "_pmr_modular_test_core_main", CODE_DIRECTORY / "main.py"
        )
        sys.modules["main"] = core_main
        monolithic = _load_module(
            "_pmr_modular_test_monolithic", CODE_DIRECTORY / "qiskit_fig5.py"
        )
        for dependency in dependencies:
            sys.modules[dependency] = _load_module(
                f"_pmr_modular_test_{dependency}",
                MODULAR_DIRECTORY / f"{dependency}.py",
            )
        modular = _load_module(
            "_pmr_modular_test_main", MODULAR_DIRECTORY / "main.py"
        )
    return modular, monolithic


MODULAR, MONOLITHIC = _load_interfaces()


class ModularCoreIntegrationTests(unittest.TestCase):
    """Verify that the compatibility facade preserves shared core guarantees."""

    def test_model_and_codec_are_shared_classes(self) -> None:
        """Require one model and codec implementation behind both entry points."""
        self.assertIs(MODULAR.PMRModel, PMRModel)
        self.assertIs(MONOLITHIC.PMRModel, PMRModel)
        self.assertIs(MODULAR.AnglePhaseCodec, QiskitAnglePhaseCodec)
        self.assertIs(MONOLITHIC.AnglePhaseCodec, QiskitAnglePhaseCodec)
        self.assertTrue(issubclass(MODULAR.ValueOracle, ValueOracle))
        self.assertTrue(issubclass(MODULAR.LowercaseROracle, QiskitRTable))

    def test_validation_rejects_unsupported_inputs(self) -> None:
        """Keep unique-mask, finite-input, exact-zero, and bound-coverage checks."""
        invalid = (
            ((0, 1, 1), [[1, -1], [0.5, 0.5], [0.5, 0.5]], None,
             "duplicate move masks"),
            ((0, 1), [[1, -1], [1, 2]], None, "Hermiticity failed"),
            ((0, 1), [[1, -1], [np.nan, np.nan]], None,
             "coefficients must be finite"),
            ((0, 1), [[1, -1], [1e-300, 1e-300]], [1, 0],
             "exactly zero centered row"),
            ((0, 1), [[1, -1], [0.5, 0.5]], [1, 0.25],
             "bound is smaller"),
        )
        for masks, coefficients, bounds, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    MODULAR.PMRModel.from_coefficients(
                        1, masks, coefficients, bounds=bounds, shift=0
                    )

    def test_hermiticity_is_numerical_and_absolute_only(self) -> None:
        """Keep tolerated residuals while rejecting a large-scale relative mismatch."""
        model = MODULAR.PMRModel.from_coefficients(
            1, (0,), [[1 + 2e-13j, -1]], shift=0, atol=1e-12
        )
        self.assertEqual(model.coefficients[0, 0].imag, 2e-13)
        with self.assertRaisesRegex(ValueError, "Hermiticity failed"):
            MODULAR.PMRModel.from_coefficients(
                1, (0, 1), [[1, -1], [1e8, 1e8 + 1]], atol=1e-12
            )

    def test_zero_rows_keep_their_distinct_preparation_weights(self) -> None:
        """A zero bound is inactive; a positive-bound zero row still contributes alpha."""
        model = MODULAR.PMRModel.from_coefficients(
            2, (2, 0, 1), [[1] * 4, [4] * 4, [0] * 4], bounds=[1, 0, 2]
        )
        self.assertEqual(model.masks, (2, 0, 1))
        self.assertEqual(model.alpha, 3)
        value = MODULAR.ValueOracle(model)
        np.testing.assert_array_equal(value.table(), [[1] * 4, [0] * 4, [0] * 4])
        for row in (1, 2):
            for source in range(4):
                self.assertEqual(value.value(row, source), 0)
        np.testing.assert_allclose(
            MODULAR.PrepOracle(model).target_state(),
            [np.sqrt(1 / 3), 0, np.sqrt(2 / 3), 0], atol=1e-15, rtol=0,
        )

    def test_all_zero_target_requires_explicit_positive_normalization(self) -> None:
        """Reject zero automatic alpha and retain the documented explicit-bounds case."""
        with self.assertRaisesRegex(ValueError, "zero automatic normalization"):
            MODULAR.PMRModel.from_coefficients(1, (0, 1), [[3, 3], [0, 0]])
        model = MODULAR.PMRModel.from_coefficients(
            1, (0, 1), [[3, 3], [0, 0]], bounds=[0, 2]
        )
        self.assertEqual(model.alpha, 2)
        np.testing.assert_array_equal(MODULAR.ValueOracle(model).table(), np.zeros((2, 2)))

    def test_public_arrays_and_views_cannot_mutate_cached_values(self) -> None:
        """Mutation through constructor inputs, arrays, views, or table copies stays local."""
        coefficients = np.array([[1, -1], [1j, -1j]], dtype=complex)
        bounds = np.array([1.0, 1.0])
        expected = coefficients.copy()
        model = MODULAR.PMRModel.from_coefficients(
            1, (0, 1), coefficients, bounds=bounds, shift=0
        )
        value = MODULAR.ValueOracle(model)
        r = MODULAR.LowercaseROracle(
            value, MODULAR.AnglePhaseCodec(theta_bits=2, phase_bits=2)
        )
        expected_words = r.words.copy()
        self.assertTrue(coefficients.flags.writeable)
        self.assertTrue(bounds.flags.writeable)
        coefficients[:] = 0
        bounds[:] = 9
        exposed = (
            model.coefficients, model.centered_coefficients, model.bounds,
            model.normalized_coefficients(), value.table(), r.words,
        )
        for array in exposed:
            for candidate in (array, array.view()):
                try:
                    candidate.setflags(write=True)
                    candidate.flat[0] = 7
                except ValueError:
                    pass  # A genuinely read-only result also protects the model.
        np.testing.assert_array_equal(model.coefficients, expected)
        np.testing.assert_array_equal(model.centered_coefficients, expected)
        np.testing.assert_array_equal(model.bounds, [1, 1])
        np.testing.assert_array_equal(value.table(), expected)
        self.assertEqual(value.value(1, 0), 1j)
        np.testing.assert_array_equal(r.words, expected_words)

    def test_tiny_classical_value_is_not_pruned(self) -> None:
        """Keep a representable tiny VALUE before the circuit's explicit quantization."""
        tiny = 1e-300
        model = MODULAR.PMRModel.from_coefficients(
            1, (0, 1), [[0, 0], [tiny, tiny]], bounds=[0, 1], shift=0
        )
        value = MODULAR.ValueOracle(model)
        self.assertNotEqual(value.value(1, 0), 0)
        self.assertEqual(value.value(1, 0), tiny)
        self.assertEqual(value.normalized_value(1, 0), tiny)


class ModularCircuitIntegrationTests(unittest.TestCase):
    """Compare the two front ends against explicit matrices with exact-grid values."""

    def test_complex_block_and_complete_unitary_match(self) -> None:
        """Check source indexing, centering, row order, and every circuit input state."""
        model = MODULAR.PMRModel.from_coefficients(
            1, (1, 0), [[1j, -1j], [3, 1]], bounds=[1, 1]
        )
        hamiltonian = np.array([[3, -1j], [1j, 1]], dtype=complex)
        expected = (hamiltonian - 2 * np.eye(2)) / 2
        modular = MODULAR.build_fig5_circuit(
            model, theta_bits=2, phase_bits=2, barriers=False
        )
        monolithic = MONOLITHIC.build_qiskit_fig5_circuit(
            model, theta_bits=2, phase_bits=2, barriers=False
        )
        self.assertEqual(modular.circuit.num_qubits, 7)
        np.testing.assert_allclose(
            MODULAR.encoded_block_from_statevector(modular, max_qubits=7),
            expected, atol=2e-12, rtol=0,
        )
        np.testing.assert_allclose(
            MONOLITHIC.encoded_block_from_statevector(monolithic, max_qubits=7),
            expected, atol=2e-12, rtol=0,
        )
        unitary = Operator(modular.circuit).data
        np.testing.assert_allclose(
            unitary, Operator(monolithic.circuit).data,
            atol=2e-12, rtol=0,
        )
        np.testing.assert_allclose(
            unitary.conj().T @ unitary, np.eye(128), atol=2e-12, rtol=0
        )

    def test_zero_weight_and_positive_weight_zero_rows_encode_known_block(self) -> None:
        """Check zero-row handling, padded moves, and alpha against a written 4x4 matrix."""
        model = MODULAR.PMRModel.from_coefficients(
            2, (2, 0, 1), [[1] * 4, [4] * 4, [0] * 4], bounds=[1, 0, 2]
        )
        hamiltonian = np.array(
            [[4, 0, 1, 0], [0, 4, 0, 1], [1, 0, 4, 0], [0, 1, 0, 4]],
            dtype=complex,
        )
        expected = (hamiltonian - 4 * np.eye(4)) / 3
        modular = MODULAR.build_fig5_circuit(
            model, theta_bits=2, phase_bits=2, barriers=False
        )
        monolithic = MONOLITHIC.build_qiskit_fig5_circuit(
            model, theta_bits=2, phase_bits=2, barriers=False
        )
        self.assertEqual(modular.circuit.num_qubits, 9)
        for build, extract in (
            (modular, MODULAR.encoded_block_from_statevector),
            (monolithic, MONOLITHIC.encoded_block_from_statevector),
        ):
            with self.subTest(entry_point=extract.__module__):
                np.testing.assert_allclose(
                    extract(build, max_qubits=9), expected, atol=2e-12, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
