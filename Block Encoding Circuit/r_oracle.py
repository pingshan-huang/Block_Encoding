"""Reversible lookup facade adapted from Antonio's modular oracle (9b091ff)."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

from angle_phase_codec import RotationData
from qiskit_oracles import QiskitRTable, append_r


class LowercaseROracle(QiskitRTable):
    """XOR-load the shared angle/phase lookup while preserving the original API.

    Move rows use the model's supplied order. Padding rows have word zero, so
    they leave the workspace unchanged. The operation is its own inverse for
    every initial workspace word, not just an initially empty register.
    """

    def word(self, move_index: int, x: int) -> int:
        """Return the packed word, including zero for a padded move row."""
        return self.lookup_word(move_index, x)

    def rotation(self, move_index: int, x: int) -> RotationData:
        """Decode one lookup word for inspection or numerical comparison."""
        return self.decoded_rotation(move_index, x)

    def append(
        self,
        circuit: QuantumCircuit,
        workspace: QuantumRegister,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Append ``|b,mu,x> -> |b XOR word(mu,x),mu,x>`` to the circuit."""
        append_r(circuit, workspace, move, system, self)

    def append_inverse(
        self,
        circuit: QuantumCircuit,
        workspace: QuantumRegister,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Undo the lookup using the same XOR operation."""
        self.append(circuit, workspace, move, system)
