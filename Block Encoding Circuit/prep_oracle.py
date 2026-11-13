"""PREP facade adapted from Antonio's modular oracle (9b091ff)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from model import PMRModel
from qiskit_oracles import append_prep


@dataclass(frozen=True)
class PrepOracle:
    """Prepare ``sqrt(b_mu / alpha)`` amplitudes in the supplied move-row order.

    Zero-weight and padded rows have zero amplitude. A zero coefficient row
    with a positive bound still participates in PREP and total normalization.
    The state-preparation gate has a unitary extension on the full register.
    """

    model: PMRModel

    def target_state(self) -> np.ndarray:
        """Return the padded move state as a new array."""
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
        """Append shared PREP, or its exact circuit inverse when requested."""
        append_prep(circuit, move, self.model, inverse=inverse)
