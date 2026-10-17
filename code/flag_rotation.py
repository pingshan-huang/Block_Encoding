"""Qrisp implementation of uppercase R on the success flag."""

from __future__ import annotations

import numpy as np
from qrisp import QuantumVariable, control, p, ry, x

from r_compute import RComputeBlock, RotationData


class FlagRotationBlock:
    """Multiplexed Ry plus phase operation controlled by register b."""

    def __init__(self, r_compute: RComputeBlock) -> None:
        self.r_compute = r_compute
        self.model = r_compute.model

    @staticmethod
    def single_qubit_unitary(data: RotationData) -> np.ndarray:
        """Return Phase_0(phi) @ Ry(theta).

        The upper-left entry is exactly the normalized complex coefficient.
        """
        rho = data.magnitude
        sine = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        phase = np.exp(1j * data.phase)
        return np.asarray(
            [[phase * rho, -phase * sine], [sine, rho]],
            dtype=np.complex128,
        )

    def apply(self, flag: QuantumVariable, b: QuantumVariable) -> None:
        """Apply the b-controlled variable Ry and phase gates in Qrisp."""
        if flag.size != 1:
            raise ValueError("R requires a one-qubit flag register")
        if b.size != self.r_compute.b_qubits:
            raise ValueError(
                f"R expected {self.r_compute.b_qubits} b qubits, got {b.size}"
            )

        for label, data in self.r_compute.data_by_label.items():
            with control(b.reg, ctrl_state=label):
                ry(data.theta, flag[0])
                if not np.isclose(data.phase, 0.0):
                    # Qrisp p(phi) phases |1>.  Conjugating it by X phases
                    # |0>, which is the success component in Appendix A.
                    x(flag[0])
                    p(data.phase, flag[0])
                    x(flag[0])

    @property
    def full_dim(self) -> int:
        return 2 * self.r_compute.local_dim

    def unitary(self) -> np.ndarray:
        """Return uppercase R on register order (f, b, m, s)."""
        b_dim = self.r_compute.b_dim
        move_dim = self.model.move_dim
        system_dim = self.model.system_dim
        rest_dim = b_dim * move_dim * system_dim
        result = np.zeros((2 * rest_dim, 2 * rest_dim), dtype=np.complex128)

        def index(flag: int, b: int, move: int, x: int) -> int:
            return ((flag * b_dim + b) * move_dim + move) * system_dim + x

        identity = np.eye(2, dtype=np.complex128)
        for b in range(b_dim):
            data = self.r_compute.data_by_label.get(b)
            flag_gate = self.single_qubit_unitary(data) if data else identity
            for move in range(move_dim):
                for x in range(system_dim):
                    for flag_in in range(2):
                        for flag_out in range(2):
                            result[
                                index(flag_out, b, move, x),
                                index(flag_in, b, move, x),
                            ] = flag_gate[flag_out, flag_in]
        return result
