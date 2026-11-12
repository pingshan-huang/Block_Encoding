"""PREP and PREP-dagger oracle implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation

from model import PMRModel


@dataclass(frozen=True)
class PrepOracle:
    """Prepare move amplitudes sqrt(bound_mu / alpha)."""

    model: PMRModel

    def target_state(self) -> np.ndarray:
        """Return the desired move-register statevector."""
        amplitudes = np.zeros(self.model.move_dim, dtype=np.complex128)
        amplitudes[: self.model.move_count] = np.sqrt(
            self.model.bounds / self.model.alpha
        )
        return amplitudes

    def append(
        self,
        circuit: QuantumCircuit,
        move: QuantumRegister,
        *,
        inverse: bool = False,
    ) -> None:
        """Append PREP when inverse=False, otherwise append PREP dagger."""
        gate = StatePreparation(self.target_state(), label="PREP")
        if inverse:
            gate = gate.inverse()
            gate.label = "PREP_dg"
        circuit.append(gate, list(move))
