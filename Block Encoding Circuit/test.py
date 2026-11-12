"""Unit and integration tests for the modular Qiskit oracle design."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from angle_phase_codec import AnglePhaseCodec
from flag_rotation_oracle import FlagRotationOracle
from inspect_oracle import build_oracle_circuit
from main import build_fig5_circuit, encoded_block_from_statevector, example_model
from move_oracle import MoveOracle
from prep_oracle import PrepOracle
from r_oracle import LowercaseROracle
from value_oracle import ValueOracle


class OracleTests(unittest.TestCase):
    """Test each oracle separately before testing the wired circuit."""

    def setUp(self) -> None:
        self.model = example_model()
        self.codec = AnglePhaseCodec(theta_bits=3, phase_bits=4)
        self.value = ValueOracle(self.model)
        self.r = LowercaseROracle(self.value, self.codec)

    def test_value_oracle(self) -> None:
        expected = np.asarray(
            [[1.0, -1.0], [0.6 + 0.8j, 0.6 - 0.8j]],
            dtype=np.complex128,
        )
        self.assertTrue(np.allclose(self.value.table(), expected))

    def test_prep_oracle(self) -> None:
        move = QuantumRegister(self.model.move_qubits, "move")
        circuit = QuantumCircuit(move)
        oracle = PrepOracle(self.model)
        oracle.append(circuit, move)
        actual = Statevector.from_instruction(circuit).data
        self.assertTrue(np.allclose(actual, oracle.target_state()))

    def test_lowercase_r_and_inverse(self) -> None:
        workspace = QuantumRegister(self.codec.workspace_bits, "b")
        move = QuantumRegister(self.model.move_qubits, "move")
        system = QuantumRegister(self.model.n_qubits, "system")
        circuit = QuantumCircuit(workspace, move, system)
        circuit.x(move[0])
        self.r.append(circuit, workspace, move, system)
        self.r.append_inverse(circuit, workspace, move, system)
        actual = Statevector.from_instruction(circuit)

        expected_index = 1 << len(workspace)
        expected = np.zeros(1 << circuit.num_qubits, dtype=np.complex128)
        expected[expected_index] = 1.0
        self.assertTrue(np.allclose(actual.data, expected))

    def test_flag_rotation_oracle_builds(self) -> None:
        flag = QuantumRegister(1, "flag")
        workspace = QuantumRegister(self.codec.workspace_bits, "b")
        circuit = QuantumCircuit(flag, workspace)
        FlagRotationOracle(self.r).append(circuit, flag, workspace)
        self.assertGreater(circuit.size(), 0)

    def test_move_oracle(self) -> None:
        move = QuantumRegister(self.model.move_qubits, "move")
        system = QuantumRegister(self.model.n_qubits, "system")
        circuit = QuantumCircuit(move, system)
        circuit.x(move[0])
        MoveOracle(self.model).append(circuit, move, system)
        state = Statevector.from_instruction(circuit)
        self.assertAlmostEqual(abs(state.data[3] - 1.0), 0.0)

    def test_complete_wired_circuit(self) -> None:
        build = build_fig5_circuit(
            self.model,
            theta_bits=self.codec.theta_bits,
            phase_bits=self.codec.phase_bits,
        )
        actual = encoded_block_from_statevector(build)
        expected_quantized = np.asarray(
            [
                [2.0 / 3.0, (np.sqrt(0.5) - 1j * np.sqrt(0.5)) / 3.0],
                [(np.sqrt(0.5) + 1j * np.sqrt(0.5)) / 3.0, -2.0 / 3.0],
            ],
            dtype=np.complex128,
        )
        self.assertTrue(np.allclose(actual, expected_quantized, atol=1e-12))

    def test_individual_oracle_inspector(self) -> None:
        for name in ("prep", "prep-dagger", "r", "r-dagger", "R", "move"):
            circuit = build_oracle_circuit(
                name,
                theta_bits=self.codec.theta_bits,
                phase_bits=self.codec.phase_bits,
            )
            self.assertIsNotNone(circuit)
            self.assertGreater(circuit.num_qubits, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
