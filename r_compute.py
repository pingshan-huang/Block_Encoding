"""Qrisp implementation of lowercase r, the reversible data computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qrisp import QuantumVariable, control, invert, x as qx

from pmr_model import next_power_of_two
from value import ValueOracle


@dataclass(frozen=True)
class RotationData:
    normalized_value: complex
    magnitude: float
    theta: float
    phase: float


class RComputeBlock:
    """Ideal reversible lookup implementation of the paper's lowercase r.

    The finite ``b`` register stores a symbolic label for precomputed
    ``(theta, phase)`` data.  XOR-loading makes the lookup self-inverse and
    explicitly models the compute/uncompute structure in Fig. 5.
    """

    def __init__(self, value_oracle: ValueOracle) -> None:
        self.value_oracle = value_oracle
        self.model = value_oracle.model

        entry_count = self.model.move_count * self.model.system_dim
        self.b_dim = next_power_of_two(entry_count + 1)
        self.labels = np.zeros(
            (self.model.move_dim, self.model.system_dim), dtype=np.int64
        )
        self.data_by_label: dict[int, RotationData] = {}

        label = 1
        for move_index in range(self.model.move_count):
            for x in range(self.model.system_dim):
                z = self.value_oracle.normalized_value(move_index, x)
                magnitude = min(1.0, abs(z))
                phase = float(np.angle(z)) if magnitude > 0.0 else 0.0
                theta = 2.0 * float(np.arccos(magnitude))
                self.labels[move_index, x] = label
                self.data_by_label[label] = RotationData(
                    normalized_value=z,
                    magnitude=magnitude,
                    theta=theta,
                    phase=phase,
                )
                label += 1

    @property
    def local_dim(self) -> int:
        return self.b_dim * self.model.move_dim * self.model.system_dim

    @property
    def b_qubits(self) -> int:
        return (self.b_dim - 1).bit_length()

    def apply(
        self,
        b: QuantumVariable,
        move: QuantumVariable,
        system: QuantumVariable,
    ) -> None:
        """Compute the symbolic ``(theta, phi)`` label into ``b``.

        The implementation uses Qrisp control environments and controlled-X
        operations.  Each lookup is an XOR, so it is a reversible VALUE plus
        angle/phase-arithmetic oracle at the abstraction level of Fig. 5.
        """
        if b.size != self.b_qubits:
            raise ValueError(f"r expected {self.b_qubits} b qubits, got {b.size}")
        if move.size != self.model.move_qubits:
            raise ValueError(
                f"r expected {self.model.move_qubits} move qubits, got {move.size}"
            )
        if system.size != self.model.n_qubits:
            raise ValueError(
                f"r expected {self.model.n_qubits} system qubits, got {system.size}"
            )

        for move_index in range(self.model.move_count):
            with control(move.reg, ctrl_state=move_index):
                for system_state in range(self.model.system_dim):
                    label = int(self.labels[move_index, system_state])
                    with control(system.reg, ctrl_state=system_state):
                        for bit in range(self.b_qubits):
                            if (label >> bit) & 1:
                                qx(b[bit])

    def apply_inverse(
        self,
        b: QuantumVariable,
        move: QuantumVariable,
        system: QuantumVariable,
    ) -> None:
        """Apply r dagger using Qrisp's inversion environment."""
        with invert():
            self.apply(b, move, system)

    def unitary(self) -> np.ndarray:
        """Return r on the register order (b, m, s)."""
        move_dim = self.model.move_dim
        system_dim = self.model.system_dim
        result = np.zeros((self.local_dim, self.local_dim), dtype=np.complex128)

        def index(b: int, move: int, x: int) -> int:
            return (b * move_dim + move) * system_dim + x

        for b in range(self.b_dim):
            for move in range(move_dim):
                for x in range(system_dim):
                    loaded = b ^ int(self.labels[move, x])
                    result[index(loaded, move, x), index(b, move, x)] = 1.0
        return result

    def inverse(self) -> np.ndarray:
        # XOR loading is an involution. Keep this method explicit so a future
        # arithmetic implementation can supply a different adjoint.
        return self.unitary()
