"""Reversible lowercase-r lookup oracle."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate

from angle_phase_codec import AnglePhaseCodec, RotationData
from circuit_utils import (
    append_controlled_gate,
    close_control_pattern,
    open_control_pattern,
)
from value_oracle import ValueOracle


class LowercaseROracle:
    """XOR-load fixed-point angle and phase data into register b."""

    def __init__(self, value_oracle: ValueOracle, codec: AnglePhaseCodec) -> None:
        self.value_oracle = value_oracle
        self.model = value_oracle.model
        self.codec = codec
        self._words = np.zeros(
            (self.model.move_dim, self.model.system_dim),
            dtype=np.uint64,
        )

        for move_index in range(self.model.move_count):
            for x in range(self.model.system_dim):
                self._words[move_index, x] = codec.encode_value(
                    value_oracle.value(move_index, x)
                )

        self._words.flags.writeable = False

    @property
    def workspace_bits(self) -> int:
        return self.codec.workspace_bits

    def word(self, move_index: int, x: int) -> int:
        """Return the packed workspace word for one branch."""
        if not 0 <= move_index < self.model.move_dim:
            raise IndexError("move_index is outside the padded move register")
        if not 0 <= x < self.model.system_dim:
            raise IndexError("x is outside the system register")
        return int(self._words[move_index, x])

    def rotation(self, move_index: int, x: int) -> RotationData:
        """Return the decoded rotation for inspection or testing."""
        return self.codec.decode_word(self.word(move_index, x))

    def active_words(self) -> set[int]:
        """Return words produced by valid move and system branches."""
        return {
            self.word(move_index, x)
            for move_index in range(self.model.move_count)
            for x in range(self.model.system_dim)
        }

    def append(
        self,
        circuit: QuantumCircuit,
        workspace: QuantumRegister,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Append r: |b,mu,x> -> |b XOR word(mu,x),mu,x>."""
        if len(workspace) != self.workspace_bits:
            raise ValueError("workspace register has the wrong width")

        controls = [*move, *system]
        move_width = len(move)

        for move_index in range(self.model.move_count):
            for x in range(self.model.system_dim):
                word = self.word(move_index, x)
                if word == 0:
                    continue

                pattern = move_index | (x << move_width)
                opened = open_control_pattern(circuit, controls, pattern)
                for bit, target in enumerate(workspace):
                    if (word >> bit) & 1:
                        append_controlled_gate(circuit, XGate(), controls, [target])
                close_control_pattern(circuit, opened)

    def append_inverse(
        self,
        circuit: QuantumCircuit,
        workspace: QuantumRegister,
        move: QuantumRegister,
        system: QuantumRegister,
    ) -> None:
        """Append r dagger; XOR loading makes it identical to r."""
        self.append(circuit, workspace, move, system)
