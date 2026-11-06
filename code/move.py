"""Framework-independent controlled MOVE block."""

from __future__ import annotations

import numpy as np

from pmr_model import PMRModel
from state import SparseState, add_amplitude


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

    def apply(self, state: SparseState) -> SparseState:
        """Apply |mu>|x> -> |mu>|x XOR mu> to a sparse state."""
        result: SparseState = {}
        for (flag, workspace, move, x), amplitude in state.items():
            mask = self.model.masks[move] if move < self.model.move_count else 0
            add_amplitude(result, (flag, workspace, move, x ^ mask), amplitude)
        return result

    def inverse(self) -> np.ndarray:
        # Every XOR move is its own inverse.
        return self.unitary()
