"""Framework-independent PREP block for the grouped PMR circuit."""

from __future__ import annotations

import numpy as np

from pmr_model import PMRModel
from state import SparseState, add_amplitude


class PrepBlock:
    """Prepare amplitudes sqrt(b_mu / alpha) in the move register."""

    def __init__(self, model: PMRModel) -> None:
        self.model = model

    def target_state(self) -> np.ndarray:
        state = np.zeros(self.model.move_dim, dtype=np.complex128)
        state[: self.model.move_count] = np.sqrt(self.model.bounds / self.model.alpha)
        return state

    def unitary(self) -> np.ndarray:
        """Return a Householder unitary whose first column is the PREP state."""
        target = self.target_state()
        initial = np.zeros_like(target)
        initial[0] = 1.0

        if np.allclose(target, initial):
            return np.eye(self.model.move_dim, dtype=np.complex128)

        direction = initial - target
        direction /= np.linalg.norm(direction)
        return np.eye(self.model.move_dim) - 2.0 * np.outer(
            direction, np.conjugate(direction)
        )

    def inverse(self) -> np.ndarray:
        return np.conjugate(self.unitary().T)

    def _apply_gate(self, state: SparseState, gate: np.ndarray) -> SparseState:
        """Apply a move-register gate to a sparse (f,b,m,x) state."""
        result: SparseState = {}
        for (flag, workspace, move, x), amplitude in state.items():
            for output_move in range(self.model.move_dim):
                factor = gate[output_move, move]
                if abs(factor) > 1e-15:
                    add_amplitude(
                        result,
                        (flag, workspace, output_move, x),
                        factor * amplitude,
                    )
        return result

    def apply(self, state: SparseState) -> SparseState:
        """Apply PREP to the move register."""
        return self._apply_gate(state, self.unitary())

    def apply_inverse(self, state: SparseState) -> SparseState:
        """Apply PREP dagger to the move register."""
        return self._apply_gate(state, self.inverse())
