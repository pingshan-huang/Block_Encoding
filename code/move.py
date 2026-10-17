"""Qrisp controlled MOVE block for PMR bit-flip masks."""

from __future__ import annotations

import numpy as np
from qrisp import QuantumVariable, control, x as qx

from pmr_model import PMRModel


class MoveBlock:
    """Apply |mu>|x> -> |mu>|x xor mu>."""

    def __init__(self, model: PMRModel) -> None:
        self.model = model

    @property
    def local_dim(self) -> int:
        return self.model.move_dim * self.model.system_dim

    def unitary(self) -> np.ndarray:
        """Return MOVE on register order (m, s)."""
        move_dim = self.model.move_dim
        system_dim = self.model.system_dim
        result = np.zeros((self.local_dim, self.local_dim), dtype=np.complex128)

        for move in range(move_dim):
            mask = self.model.masks[move] if move < self.model.move_count else 0
            for x in range(system_dim):
                source = move * system_dim + x
                target = move * system_dim + (x ^ mask)
                result[target, source] = 1.0
        return result

    def apply(self, move: QuantumVariable, system: QuantumVariable) -> None:
        """Apply MOVE using Qrisp multi-control environments."""
        if move.size != self.model.move_qubits:
            raise ValueError(
                f"MOVE expected {self.model.move_qubits} move qubits, got {move.size}"
            )
        if system.size != self.model.n_qubits:
            raise ValueError(
                f"MOVE expected {self.model.n_qubits} system qubits, got {system.size}"
            )

        for move_index, mask in enumerate(self.model.masks):
            if mask == 0:
                continue
            with control(move.reg, ctrl_state=move_index):
                for bit in range(self.model.n_qubits):
                    if (mask >> bit) & 1:
                        qx(system[bit])

    def inverse(self) -> np.ndarray:
        # Every XOR move is its own inverse.
        return self.unitary()
