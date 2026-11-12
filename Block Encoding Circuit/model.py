"""Validated input model for a grouped PMR Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to value."""
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class PMRModel:
    """Store the move-amplitude representation of a Hermitian Hamiltonian.

    ``coefficients[mu, x]`` contributes
    ``coefficients[mu, x] |x XOR masks[mu]><x|`` to the Hamiltonian.
    """

    n_qubits: int
    masks: tuple[int, ...]
    coefficients: np.ndarray
    centered_coefficients: np.ndarray
    bounds: np.ndarray
    shift: float

    @classmethod
    def from_coefficients(
        cls,
        n_qubits: int,
        masks: Sequence[int],
        coefficients: Iterable[Iterable[complex]],
        *,
        bounds: Sequence[float] | None = None,
        shift: float | None = None,
        atol: float = 1e-12,
    ) -> "PMRModel":
        """Validate raw PMR data and calculate centering and normalization data."""
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")

        masks_tuple = tuple(int(mask) for mask in masks)
        if not masks_tuple or 0 not in masks_tuple:
            raise ValueError("the move list must include the diagonal mask 0")
        if len(set(masks_tuple)) != len(masks_tuple):
            raise ValueError("move masks must be unique")

        system_dim = 1 << n_qubits
        if any(mask < 0 or mask >= system_dim for mask in masks_tuple):
            raise ValueError("each move mask must fit in the system register")

        coefficient_array = np.asarray(coefficients, dtype=np.complex128)
        expected_shape = (len(masks_tuple), system_dim)
        if coefficient_array.shape != expected_shape:
            raise ValueError(
                f"coefficients must have shape {expected_shape}, "
                f"got {coefficient_array.shape}"
            )

        diagonal_index = masks_tuple.index(0)
        diagonal = coefficient_array[diagonal_index]
        if not np.allclose(diagonal.imag, 0.0, atol=atol):
            raise ValueError("diagonal coefficients must be real")

        if shift is None:
            shift = 0.5 * (float(np.max(diagonal.real)) + float(np.min(diagonal.real)))
        shift = float(shift)

        centered = coefficient_array.copy()
        centered[diagonal_index] -= shift
        required_bounds = np.max(np.abs(centered), axis=1)

        if bounds is None:
            bound_array = required_bounds
        else:
            bound_array = np.asarray(bounds, dtype=float)
            if bound_array.shape != (len(masks_tuple),):
                raise ValueError("bounds must contain one entry per move")
            if np.any(bound_array + atol < required_bounds):
                raise ValueError("a supplied bound is too small")

        if np.any(bound_array <= 0.0):
            raise ValueError("zero-amplitude moves must be removed")

        model = cls(
            n_qubits=n_qubits,
            masks=masks_tuple,
            coefficients=coefficient_array,
            centered_coefficients=centered,
            bounds=np.asarray(bound_array, dtype=float),
            shift=shift,
        )
        model.validate_hermiticity(atol=atol)
        return model

    @property
    def system_dim(self) -> int:
        return 1 << self.n_qubits

    @property
    def move_count(self) -> int:
        return len(self.masks)

    @property
    def move_dim(self) -> int:
        return max(2, next_power_of_two(self.move_count))

    @property
    def move_qubits(self) -> int:
        return (self.move_dim - 1).bit_length()

    @property
    def alpha(self) -> float:
        return float(np.sum(self.bounds))

    def validate_hermiticity(self, *, atol: float = 1e-12) -> None:
        """Check c_mu(x XOR mu) = conjugate(c_mu(x))."""
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                reverse_x = x ^ mask
                lhs = self.coefficients[move_index, reverse_x]
                rhs = np.conjugate(self.coefficients[move_index, x])
                if not np.allclose(lhs, rhs, atol=atol):
                    raise ValueError(
                        f"Hermiticity failed for mask={mask}, x={x}: "
                        f"{lhs} != {rhs}"
                    )

    def hamiltonian(self) -> np.ndarray:
        """Construct the dense Hamiltonian for verification."""
        matrix = np.zeros((self.system_dim, self.system_dim), dtype=np.complex128)
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                matrix[x ^ mask, x] += self.coefficients[move_index, x]
        return matrix

    def target_block(self) -> np.ndarray:
        """Return the matrix that the all-zero-ancilla block must encode."""
        identity = np.eye(self.system_dim, dtype=np.complex128)
        return (self.hamiltonian() - self.shift * identity) / self.alpha
