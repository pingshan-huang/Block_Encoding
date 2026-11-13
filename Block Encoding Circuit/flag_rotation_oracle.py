"""Flag-rotation facade adapted from Antonio's modular oracle (9b091ff)."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister

from r_oracle import LowercaseROracle
from qiskit_oracles import append_flag_rotation


class FlagRotationOracle:
    """Place the decoded complex VALUE in the flag-zero amplitude.

    For a loaded word, apply ``diag(exp(i*phase), 1) @ RY(theta)``. Acting on
    flag zero gives ``cos(theta/2)*exp(i*phase)`` in the success branch. This is
    the quantized VALUE, not generally the exact unquantized coefficient.
    Workspace words absent from the lookup act as identity.
    """

    def __init__(self, r_oracle: LowercaseROracle) -> None:
        """Use the same encoded lookup as the preceding lowercase-r oracle."""
        self.r_oracle = r_oracle

    def append(
        self,
        circuit: QuantumCircuit,
        flag: QuantumRegister,
        workspace: QuantumRegister,
    ) -> None:
        """Append the shared workspace-controlled magnitude and phase gates."""
        append_flag_rotation(circuit, flag, workspace, self.r_oracle)
