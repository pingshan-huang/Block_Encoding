"""Uppercase-R flag-rotation oracle."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PhaseGate, RYGate

from circuit_utils import (
    append_controlled_gate,
    close_control_pattern,
    open_control_pattern,
)
from r_oracle import LowercaseROracle


class FlagRotationOracle:
    """Write each normalized complex coefficient into flag-zero amplitude."""

    def __init__(self, r_oracle: LowercaseROracle) -> None:
        self.r_oracle = r_oracle

    def append(
        self,
        circuit: QuantumCircuit,
        flag: QuantumRegister,
        workspace: QuantumRegister,
    ) -> None:
        """Append the workspace-controlled Ry and success-state phase."""
        controls = list(workspace)

        for word in sorted(self.r_oracle.active_words()):
            data = self.r_oracle.codec.decode_word(word)
            if np.isclose(data.theta, 0.0) and np.isclose(data.phase, 0.0):
                continue

            opened = open_control_pattern(circuit, controls, word)

            if not np.isclose(data.theta, 0.0):
                append_controlled_gate(
                    circuit,
                    RYGate(data.theta),
                    controls,
                    [flag[0]],
                )

            if not np.isclose(data.phase, 0.0):
                # X-Phase-X places exp(i*phase) on the original flag-zero state.
                circuit.x(flag[0])
                append_controlled_gate(
                    circuit,
                    PhaseGate(data.phase),
                    controls,
                    [flag[0]],
                )
                circuit.x(flag[0])

            close_control_pattern(circuit, opened)
