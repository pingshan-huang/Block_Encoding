"""Unit tests for the modular Fig. 5 implementation."""

from __future__ import annotations

import unittest

import numpy as np

from flag_rotation import FlagRotationBlock
from main import build_fig5_circuit, example_model
from move import MoveBlock
from prep import PrepBlock
from r_compute import RComputeBlock
from value import ValueOracle


class Fig5CircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = example_model()
        self.value = ValueOracle(self.model)
        self.r_compute = RComputeBlock(self.value)

    def assert_unitary(self, matrix: np.ndarray, atol: float = 1e-12) -> None:
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        self.assertTrue(np.allclose(np.conjugate(matrix.T) @ matrix, identity, atol=atol))

    def test_prep(self) -> None:
        prep = PrepBlock(self.model)
        unitary = prep.unitary()
        self.assert_unitary(unitary)
        self.assertTrue(np.allclose(unitary[:, 0], prep.target_state()))
        self.assertAlmostEqual(float(np.linalg.norm(prep.target_state())), 1.0)

    def test_value_is_normalized_and_centered(self) -> None:
        table = self.value.table()
        self.assertTrue(np.all(np.abs(table) <= 1.0 + 1e-12))
        self.assertTrue(np.allclose(table[0], [1.0, -1.0]))

    def test_r_compute_and_uncompute(self) -> None:
        unitary = self.r_compute.unitary()
        self.assert_unitary(unitary)
        self.assertTrue(np.allclose(unitary @ unitary, np.eye(unitary.shape[0])))

    def test_flag_rotation_loads_normalized_value(self) -> None:
        block = FlagRotationBlock(self.r_compute)
        for data in self.r_compute.data_by_label.values():
            gate = block.single_qubit_unitary(data)
            self.assert_unitary(gate)
            self.assertAlmostEqual(abs(gate[0, 0] - data.normalized_value), 0.0, places=12)

    def test_move(self) -> None:
        move = MoveBlock(self.model)
        unitary = move.unitary()
        self.assert_unitary(unitary)
        self.assertTrue(np.allclose(unitary @ unitary, np.eye(unitary.shape[0])))

    def test_complete_block_encoding(self) -> None:
        build = build_fig5_circuit(self.model)
        self.assertTrue(
            np.allclose(build.encoded_block, self.model.target_block(), atol=2e-6)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
