"""Unit tests for the framework-independent modular implementation."""

from __future__ import annotations

import unittest

import numpy as np

from flag_rotation import FlagRotationBlock
from main import build_fig5_circuit, example_model, run_column
from move import MoveBlock
from prep import PrepBlock
from r_block import RBlock
from rotation_codec import AnglePhaseCodec
from value import ValueOracle


class CircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = example_model()
        self.value = ValueOracle(self.model)
        self.codec = AnglePhaseCodec(theta_bits=12, phase_bits=16)
        self.r = RBlock(self.value, self.codec)

    def assert_unitary(self, matrix: np.ndarray, atol: float = 1e-12) -> None:
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        self.assertTrue(np.allclose(matrix.conj().T @ matrix, identity, atol=atol))

    def test_prep(self) -> None:
        prep = PrepBlock(self.model)
        self.assert_unitary(prep.unitary())
        self.assertTrue(np.allclose(prep.unitary()[:, 0], prep.target_state()))

    def test_value_is_normalized_and_centered(self) -> None:
        table = self.value.table()
        self.assertTrue(np.all(np.abs(table) <= 1.0 + 1e-12))
        self.assertTrue(np.allclose(table[0], [1.0, -1.0]))

    def test_r_and_r_dagger(self) -> None:
        for move in range(self.model.move_dim):
            for x in range(self.model.system_dim):
                basis = (0, 0, move, x)
                loaded = self.r.apply_basis(*basis)
                self.assertEqual(self.r.apply_basis(*loaded), basis)

    def test_flag_rotation_uses_decoded_value(self) -> None:
        block = FlagRotationBlock(self.r)
        for move in range(self.model.move_count):
            for x in range(self.model.system_dim):
                data = self.r.decoded_rotation(move, x)
                gate = block.single_qubit_unitary(data)
                self.assert_unitary(gate)
                self.assertAlmostEqual(abs(gate[0, 0] - data.normalized_value), 0.0)

    def test_move(self) -> None:
        move = MoveBlock(self.model)
        self.assert_unitary(move.unitary())
        self.assertTrue(np.allclose(move.unitary() @ move.unitary(), np.eye(move.local_dim)))

    def test_workspace_is_clean_after_r_inverse(self) -> None:
        final_state = run_column(
            0,
            PrepBlock(self.model),
            self.r,
            FlagRotationBlock(self.r),
            MoveBlock(self.model),
        )
        self.assertTrue(all(workspace == 0 for _, workspace, _, _ in final_state))

    def test_complete_block_encoding(self) -> None:
        build = build_fig5_circuit(self.model)
        self.assertTrue(
            np.allclose(build.encoded_block, self.model.target_block(), atol=5e-5)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
