"""Bounded Qiskit checks against independent PMR matrices and truth tables.

Statevector checks use small angle registers. Full-unitary checks stay at seven
qubits or fewer; they include nonzero workspace and unused register states.
These tests use Qiskit's built-in quantum_info tools and do not require Aer.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIRECTORY))

from pmr_model import PMRModel
from value import ValueOracle
from qiskit_fig5 import build_qiskit_fig5_circuit, encoded_block_from_statevector
from qiskit_oracles import (
    QiskitAnglePhaseCodec,
    QiskitRTable,
    append_flag_rotation,
    append_move,
    append_prep,
    append_r,
    extract_encoded_block,
)


def independent_codes(value: complex, theta_bits: int, phase_bits: int) -> tuple[int, int]:
    """Quantize a supplied complex number without calling production VALUE or codec."""
    theta_levels = 2**theta_bits - 1
    phase_levels = 2**phase_bits
    magnitude = min(abs(value), 1.0)
    theta = 2 * np.arccos(magnitude)
    phase = np.angle(value) % (2 * np.pi) if magnitude else 0.0
    return (
        int(round(float(theta / np.pi * theta_levels))),
        int(round(float(phase / (2 * np.pi) * phase_levels))) % phase_levels,
    )


def independent_quantized_block(
    masks: tuple[int, ...],
    coefficients: np.ndarray,
    bounds: np.ndarray,
    shift: float,
    theta_bits: int,
    phase_bits: int,
) -> np.ndarray:
    """Form the finite-precision operator directly from the supplied PMR fixture."""
    dimension = coefficients.shape[1]
    expected = np.zeros((dimension, dimension), dtype=complex)
    alpha = float(sum(bounds))
    for row, mask in enumerate(masks):
        for x in range(dimension):
            centered = coefficients[row, x] - (shift if mask == 0 else 0)
            value = centered / bounds[row] if bounds[row] else 0j
            theta_code, phase_code = independent_codes(value, theta_bits, phase_bits)
            theta = theta_code * np.pi / (2**theta_bits - 1)
            phase = phase_code * 2 * np.pi / 2**phase_bits
            expected[x ^ mask, x] += (
                bounds[row] / alpha * np.cos(theta / 2) * np.exp(1j * phase)
            )
    return expected


def two_qubit_fixture() -> tuple[PMRModel, np.ndarray]:
    """Return an unsorted three-move model and its explicitly written Hamiltonian."""
    model = PMRModel.from_coefficients(
        2, (3, 0, 1),
        [[0.5, 0.5, 0.5, 0.5], [1, -1, 2, -2], [1j, -1j, 1j, -1j]],
        bounds=[0.5, 2, 1], shift=0,
    )
    expected = np.array(
        [[1, -1j, 0, 0.5], [1j, -1, 0.5, 0],
         [0, 0.5, 2, -1j], [0.5, 0, 1j, -2]], dtype=complex,
    )
    return model, expected


class EncodedBlockTests(unittest.TestCase):
    """Separate circuit-logic error from the known angle-quantization error."""

    def assert_block(
        self,
        model: PMRModel,
        expected_hamiltonian: np.ndarray,
        *,
        theta_bits: int = 2,
        phase_bits: int = 3,
    ) -> np.ndarray:
        """Compare the simulated block with independent ideal and quantized targets."""
        np.testing.assert_allclose(
            model.hamiltonian(), expected_hamiltonian, atol=1e-14, rtol=0,
        )
        build = build_qiskit_fig5_circuit(
            model, theta_bits=theta_bits, phase_bits=phase_bits, barriers=False,
        )
        actual = encoded_block_from_statevector(build, max_qubits=12)
        expected_quantized = independent_quantized_block(
            model.masks, model.coefficients, model.bounds, model.shift,
            theta_bits, phase_bits,
        )
        np.testing.assert_allclose(actual, expected_quantized, atol=2e-10, rtol=0)
        target = (expected_hamiltonian - model.shift * np.eye(model.system_dim)) / model.alpha
        error = np.linalg.norm(actual - target, ord=2)
        # Nearest rounding gives |delta theta| <= pi/(2L) and
        # |delta phase| <= pi/P. cos(theta/2) is 1/2-Lipschitz; row weights sum to 1.
        precision_bound = np.pi / (4 * (2**theta_bits - 1)) + np.pi / 2**phase_bits
        self.assertLessEqual(error, precision_bound + 2e-10)
        self.assertTrue(np.all(np.isfinite(actual)))
        return actual

    def test_pauli_x_y_z_and_negative_x(self) -> None:
        """Check signs, imaginary phases, and inactive rows against known Pauli matrices."""
        fixtures = (
            ([[0, 0], [1, 1]], [[0, 1], [1, 0]]),
            ([[0, 0], [1j, -1j]], [[0, -1j], [1j, 0]]),
            ([[1, -1], [0, 0]], [[1, 0], [0, -1]]),
            ([[0, 0], [-1, -1]], [[0, -1], [-1, 0]]),
        )
        for coefficients, hamiltonian in fixtures:
            with self.subTest(coefficients=coefficients):
                model = PMRModel.from_coefficients(1, (0, 1), coefficients, shift=0)
                actual = self.assert_block(model, np.array(hamiltonian), theta_bits=1, phase_bits=2)
                np.testing.assert_allclose(actual, hamiltonian, atol=2e-12, rtol=0)

    def test_complex_attenuated_coefficients_at_several_precisions(self) -> None:
        """Check complex phases and attenuation without confusing quantization with circuit error."""
        model = PMRModel.from_coefficients(
            1, (0, 1), [[1.5, -0.5], [0.3 + 0.4j, 0.3 - 0.4j]],
            bounds=[2, 1], shift=0.5,
        )
        hamiltonian = np.array([[1.5, 0.3 - 0.4j], [0.3 + 0.4j, -0.5]])
        for theta_bits, phase_bits in ((1, 2), (2, 3), (3, 4)):
            with self.subTest(theta_bits=theta_bits, phase_bits=phase_bits):
                self.assert_block(model, hamiltonian, theta_bits=theta_bits, phase_bits=phase_bits)

    def test_two_qubit_unsorted_masks_and_padded_move_state(self) -> None:
        """Reconstruct a complex two-qubit operator with three rows in a four-state move register."""
        model, hamiltonian = two_qubit_fixture()
        self.assertEqual(model.masks, (3, 0, 1))
        self.assertEqual(model.move_count, 3)
        self.assertEqual(model.move_dim, 4)
        self.assert_block(model, hamiltonian)

    def test_positive_bound_on_zero_row_keeps_its_preparation_weight(self) -> None:
        """A zero coefficient row with positive weight must not introduce a spurious diagonal."""
        model = PMRModel.from_coefficients(
            1, (0, 1), [[0, 0], [1, 1]], bounds=[3, 2], shift=0,
        )
        actual = self.assert_block(model, np.array([[0, 1], [1, 0]]))
        np.testing.assert_allclose(np.diag(actual), [0, 0], atol=2e-12, rtol=0)

    def test_all_zero_centered_target_with_explicit_positive_bounds(self) -> None:
        """A scalar Hamiltonian can encode the zero centered target using explicit bounds."""
        model = PMRModel.from_coefficients(1, (0,), [[3, 3]], bounds=[2])
        actual = self.assert_block(model, 3 * np.eye(2))
        np.testing.assert_allclose(actual, np.zeros((2, 2)), atol=2e-12, rtol=0)

    def test_different_valid_bounds_change_alpha_and_attenuation(self) -> None:
        """Both tight and loose bounds must agree with their own independently normalized target."""
        hamiltonian = np.array([[1, 3 - 4j], [3 + 4j, -1]])
        for bounds in ([1, 5], [2, 10]):
            with self.subTest(bounds=bounds):
                model = PMRModel.from_coefficients(
                    1, (0, 1), [[1, -1], [3 + 4j, 3 - 4j]], bounds=bounds, shift=0,
                )
                self.assert_block(model, hamiltonian)

    def test_invalid_masks_are_rejected_by_the_shared_model(self) -> None:
        """Circuit inputs retain the shared model's mask-width, integer, and uniqueness checks."""
        for masks in ((0, 2), (0, -1), (0, 0), (0, 1.0), (0, True)):
            with self.subTest(masks=masks):
                with self.assertRaises(ValueError):
                    PMRModel.from_coefficients(1, masks, [[1, -1], [1, 1]])


class PrimitiveUnitaryTests(unittest.TestCase):
    """Check complete small operators, including states not reached from zero ancillas."""

    def test_prep_and_inverse_on_the_full_move_space(self) -> None:
        """PREP is unitary, prepares the intended weights, and its inverse undoes every input."""
        model, _ = two_qubit_fixture()
        move = QuantumRegister(2, "move")
        circuit = QuantumCircuit(move)
        append_prep(circuit, move, model)
        unitary = Operator(circuit).data
        expected = np.sqrt(np.array([0.5, 2, 1, 0]) / 3.5)
        np.testing.assert_allclose(unitary[:, 0], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-12, rtol=0)
        append_prep(circuit, move, model, inverse=True)
        np.testing.assert_allclose(Operator(circuit).data, np.eye(4), atol=1e-12, rtol=0)

    def test_r_is_the_expected_xor_permutation_for_every_workspace_input(self) -> None:
        """The lookup must XOR into arbitrary workspace and leave padded move rows alone."""
        model, _ = two_qubit_fixture()
        codec = QiskitAnglePhaseCodec(theta_bits=1, phase_bits=2)
        table = QiskitRTable(ValueOracle(model), codec)
        workspace = QuantumRegister(3, "b")
        move = QuantumRegister(2, "move")
        system = QuantumRegister(2, "system")
        circuit = QuantumCircuit(workspace, move, system)
        append_r(circuit, workspace, move, system, table)
        expected = np.zeros((128, 128), dtype=complex)
        coefficients, bounds = model.coefficients, model.bounds
        for column in range(128):
            move_index = (column >> 3) & 3
            x = column >> 5
            word = 0
            if move_index < 3:
                value = coefficients[move_index, x] / bounds[move_index]
                theta_code, phase_code = independent_codes(value, 1, 2)
                word = theta_code | (phase_code << 1)
            expected[column ^ word, column] = 1
        np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12, rtol=0)
        append_r(circuit, workspace, move, system, table)
        np.testing.assert_allclose(Operator(circuit).data, np.eye(128), atol=2e-12, rtol=0)

    def test_move_matches_every_basis_state_and_is_self_inverse(self) -> None:
        """MOVE must implement the supplied masks while acting as identity on padded rows."""
        model, _ = two_qubit_fixture()
        move = QuantumRegister(2, "move")
        system = QuantumRegister(2, "system")
        circuit = QuantumCircuit(move, system)
        append_move(circuit, move, system, model)
        expected = np.zeros((16, 16), dtype=complex)
        for column in range(16):
            move_index, x = column & 3, column >> 2
            mask = (3, 0, 1, 0)[move_index]
            expected[move_index | ((x ^ mask) << 2), column] = 1
        np.testing.assert_allclose(Operator(circuit).data, expected, atol=1e-12, rtol=0)
        append_move(circuit, move, system, model)
        np.testing.assert_allclose(Operator(circuit).data, np.eye(16), atol=1e-12, rtol=0)

    def rotation_table(self, theta: float, phase: float, *, width: int = 1):
        """Supply one decoded word without allocating a large high-precision register."""
        data = SimpleNamespace(
            theta=theta, phase=phase,
            theta_code=1 if theta else 0, phase_code=1 if phase else 0,
        )
        return SimpleNamespace(
            workspace_bits=width,
            codec=SimpleNamespace(workspace_bits=width, decode_word=lambda word: data),
            active_words=lambda: {1},
        )

    def test_rotation_is_unitary_and_unused_workspace_words_are_identity(self) -> None:
        """R must preserve its specified full-space action, including inactive control patterns."""
        theta, phase = np.pi / 3, np.pi / 5
        flag, workspace = QuantumRegister(1, "flag"), QuantumRegister(2, "b")
        circuit = QuantumCircuit(flag, workspace)
        append_flag_rotation(circuit, flag, workspace, self.rotation_table(theta, phase, width=2))
        expected = np.eye(8, dtype=complex)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        expected[2:4, 2:4] = [[np.exp(1j * phase) * c, -np.exp(1j * phase) * s], [s, c]]
        actual = Operator(circuit).data
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(8), atol=1e-12, rtol=0)

    def test_tiny_nonzero_phase_is_not_discarded(self) -> None:
        """A decoded phase below np.isclose's threshold still changes the success amplitude."""
        phase = 2 * np.pi / 2**30
        flag, workspace = QuantumRegister(1, "flag"), QuantumRegister(1, "b")
        circuit = QuantumCircuit(flag, workspace)
        append_flag_rotation(circuit, flag, workspace, self.rotation_table(0, phase))
        actual = Operator(circuit).data
        expected = np.diag([1, 1, np.exp(1j * phase), 1])
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        self.assertNotEqual(actual[2, 2].imag, 0)

    def test_tiny_nonzero_theta_is_not_discarded(self) -> None:
        """A small encoded rotation must retain its nonzero flag-one amplitude."""
        theta = np.pi / (2**30 - 1)
        flag, workspace = QuantumRegister(1, "flag"), QuantumRegister(1, "b")
        circuit = QuantumCircuit(flag, workspace)
        append_flag_rotation(circuit, flag, workspace, self.rotation_table(theta, 0))
        actual = Operator(circuit).data
        self.assertNotEqual(actual[3, 2], 0)
        np.testing.assert_allclose(actual[3, 2], np.sin(theta / 2), atol=1e-14, rtol=0)

    def test_r_rotation_r_inverse_cleans_workspace_for_both_flag_inputs(self) -> None:
        """R leaves lookup bits unchanged, so unloading cleans workspace for arbitrary flag inputs."""
        model = PMRModel.from_coefficients(
            1, (0, 1), [[1, -1], [0.3 + 0.4j, 0.3 - 0.4j]], bounds=[2, 1], shift=0,
        )
        codec = QiskitAnglePhaseCodec(theta_bits=1, phase_bits=2)
        table = QiskitRTable(ValueOracle(model), codec)
        flag, workspace = QuantumRegister(1, "flag"), QuantumRegister(3, "b")
        move, system = QuantumRegister(1, "move"), QuantumRegister(1, "system")
        circuit = QuantumCircuit(flag, workspace, move, system)
        append_r(circuit, workspace, move, system, table)
        append_flag_rotation(circuit, flag, workspace, table)
        append_r(circuit, workspace, move, system, table)
        unitary = Operator(circuit).data
        dirty_rows = [i for i in range(64) if (i >> 1) & 7]
        for flag_input in (0, 1):
            for move_index in (0, 1):
                for x in (0, 1):
                    column = flag_input | (move_index << 4) | (x << 5)
                    np.testing.assert_allclose(unitary[dirty_rows, column], 0, atol=1e-12, rtol=0)


class RegisterAndPrecisionTests(unittest.TestCase):
    """Check public register assumptions and reject unsafe verification sizes early."""

    def test_extraction_uses_actual_qubit_positions_and_preserves_global_phase(self) -> None:
        """Extract the block with interleaved, reversed system qubits without phase correction."""
        qubits = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qubits)
        system = [qubits[3], qubits[1]]
        circuit.x(system[1])
        circuit.p(0.3, system[0])
        circuit.ry(0.4, qubits[0])
        circuit.global_phase = 0.23
        phase = np.exp(0.3j)
        expected = np.array([[0, 0, 1, 0], [0, 0, 0, phase],
                             [1, 0, 0, 0], [0, phase, 0, 0]], dtype=complex)
        expected *= np.cos(0.2) * np.exp(0.23j)
        actual = extract_encoded_block(circuit, system, max_qubits=4)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_default_workspace_is_rejected_before_simulation(self) -> None:
        """Building a default circuit is allowed, but its 31-qubit statevector must be guarded."""
        model = PMRModel.from_coefficients(1, (0, 1), [[0, 0], [1, 1]])
        build = build_qiskit_fig5_circuit(model, barriers=False)
        self.assertEqual(build.circuit.num_qubits, 31)
        with patch("qiskit_oracles.Statevector") as simulator, patch("qiskit_oracles.transpile") as lower:
            with self.assertRaises(ValueError):
                encoded_block_from_statevector(build)
            simulator.assert_not_called()
            simulator.from_int.assert_not_called()
            lower.assert_not_called()

    def test_precision_requires_supported_positive_integer_widths(self) -> None:
        """Reject lossy width coercions, uint64 overflow, and unsupported float precision."""
        for theta_bits, phase_bits in ((0, 2), (1, 0), (-1, 2), (True, 2),
                                       (1.5, 2), (2, "3"), (54, 1), (1, 54), (40, 30)):
            with self.subTest(theta_bits=theta_bits, phase_bits=phase_bits):
                with self.assertRaises(ValueError):
                    QiskitAnglePhaseCodec(theta_bits=theta_bits, phase_bits=phase_bits)
        valid = QiskitAnglePhaseCodec(theta_bits=np.int32(2), phase_bits=np.int64(3))
        self.assertEqual(valid.workspace_bits, 5)

    def test_register_width_mismatches_are_rejected(self) -> None:
        """Primitive builders must reject registers that cannot represent their model or codec."""
        model, _ = two_qubit_fixture()
        codec = QiskitAnglePhaseCodec(theta_bits=1, phase_bits=2)
        table = QiskitRTable(ValueOracle(model), codec)
        flag, workspace = QuantumRegister(1, "flag"), QuantumRegister(3, "b")
        move, system = QuantumRegister(2, "move"), QuantumRegister(2, "system")
        circuit = QuantumCircuit(flag, workspace, move, system)
        with self.assertRaises(ValueError):
            append_prep(circuit, [move[0]], model)
        with self.assertRaises(ValueError):
            append_r(circuit, workspace[:-1], move, system, table)
        with self.assertRaises(ValueError):
            append_r(circuit, workspace, [move[0]], system, table)
        with self.assertRaises(ValueError):
            append_move(circuit, move, [system[0]], model)
        with self.assertRaises(ValueError):
            append_flag_rotation(circuit, [flag[0], move[0]], workspace, table)

    def test_overlapping_and_foreign_qubits_are_rejected_before_appending(self) -> None:
        """Reject aliased controls and targets, or qubits outside the circuit."""
        model = PMRModel.from_coefficients(1, (0, 1), [[1, -1], [1, 1]])
        qubits, foreign = QuantumRegister(2, "q"), QuantumRegister(1, "other")
        circuit = QuantumCircuit(qubits)
        for move, system in (([qubits[0]], [qubits[0]]), ([qubits[0]], foreign)):
            with self.subTest(system=system):
                with self.assertRaises(ValueError):
                    append_move(circuit, move, system, model)
                self.assertEqual(len(circuit.data), 0)
        with self.assertRaises(ValueError):
            extract_encoded_block(circuit, [qubits[0], qubits[0]])

    def test_largest_supported_fields_do_not_spill_into_the_phase(self) -> None:
        """Large accepted fields stay packed correctly despite float-level rounding."""
        for theta_bits, phase_bits in ((1, 1), (52, 12), (12, 52), (32, 32)):
            with self.subTest(theta_bits=theta_bits, phase_bits=phase_bits):
                codec = QiskitAnglePhaseCodec(theta_bits, phase_bits)
                word = codec.encode_value(0j)
                theta_code, phase_code = codec.unpack(word)
                self.assertEqual(phase_code, 0)
                self.assertLessEqual(theta_code, codec.theta_levels)
                self.assertLess(word, 1 << codec.workspace_bits)
                for value in (1, -1, 1j, -1j):
                    word = codec.encode_value(value)
                    self.assertLess(word, 1 << codec.workspace_bits)
                    self.assertEqual(codec.unpack(word)[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
