"""Move-controlled system permutation oracle."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate

from circuit_utils import (
    append_controlled_gate,
    close_control_pattern,
    open_control_pattern,
)
from model import PMRModel


class MoveOracle:
    """Apply |mu>|x> -> |mu>|x XOR masks[mu]>."""

    def __init__(self, model: PMRModel) -> None:
        self.model = model

    def append(
        self,
        circuit: QuantumCircuit,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Append the move-controlled XOR permutation."""
        controls = list(move)

        for move_index, mask in enumerate(self.model.masks):
            if mask == 0:
                continue

            opened = open_control_pattern(circuit, controls, move_index)
            for bit, target in enumerate(system):
                if (mask >> bit) & 1:
                    append_controlled_gate(circuit, XGate(), controls, [target])
            close_control_pattern(circuit, opened)
