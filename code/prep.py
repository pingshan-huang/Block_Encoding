"""Qrisp PREP block for the grouped PMR circuit."""

from __future__ import annotations

import numpy as np
from qrisp import QuantumVariable, invert, prepare

from pmr_model import PMRModel


class PrepBlock:
    """Prepare amplitudes sqrt(b_mu / alpha) in the move register."""

    def __init__(self, model: PMRModel) -> None:
        self.model = model

    def target_state(self) -> np.ndarray:
        state = np.zeros(self.model.move_dim, dtype=np.complex128)
        state[: self.model.move_count] = np.sqrt(self.model.bounds / self.model.alpha)
        return state

    def apply(self, move: QuantumVariable) -> None:
        """Apply PREP to a Qrisp move register."""
        if move.size != self.model.move_qubits:
            raise ValueError(
                f"PREP expected {self.model.move_qubits} move qubits, got {move.size}"
            )
        prepare(move, self.target_state())

    def apply_inverse(self, move: QuantumVariable) -> None:
        """Apply PREP dagger using Qrisp's inversion environment."""
        with invert():
            self.apply(move)

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
