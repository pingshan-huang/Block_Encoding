"""MOVE facade adapted from Antonio's modular oracle (9b091ff)."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

from model import PMRModel
from qiskit_oracles import append_move


class MoveOracle:
    """Apply each row's XOR mask to the source system state.

    The move register stores row positions, not mask values. Thus the operation
    is ``|mu>|x> -> |mu>|x XOR masks[mu]>`` even for unsorted masks. Padded move
    states act as identity, and applying MOVE twice restores every basis state.
    """

    def __init__(self, model: PMRModel) -> None:
        """Keep the validated model that defines move-row positions and masks."""
        self.model = model

    def append(
        self,
        circuit: QuantumCircuit,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Append the shared move-controlled XOR permutation."""
        append_move(circuit, move, system, self.model)
