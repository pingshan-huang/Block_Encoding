"""Data model and validation for a grouped PMR Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class PMRModel:
    """Move-amplitude representation of a Hermitian Hamiltonian.

    ``coefficients[j, x]`` is the amplitude ``c_mu(x)`` for the move whose
    bit mask is ``masks[j]``.  It contributes

        c_mu(x) |x xor mu><x|.

    The diagonal move ``mu = 0`` must be present.
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
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")

        masks_tuple = tuple(int(mask) for mask in masks)
        if not masks_tuple:
            raise ValueError("at least one move is required")
        if len(set(masks_tuple)) != len(masks_tuple):
            raise ValueError("move masks must be unique")

        system_dim = 1 << n_qubits
        if any(mask < 0 or mask >= system_dim for mask in masks_tuple):
            raise ValueError("each move mask must fit in n_qubits")
        if 0 not in masks_tuple:
            raise ValueError("the diagonal move mu=0 must be included")

        coeff = np.asarray(coefficients, dtype=np.complex128)
        expected_shape = (len(masks_tuple), system_dim)
        if coeff.shape != expected_shape:
            raise ValueError(
                f"coefficients must have shape {expected_shape}, got {coeff.shape}"
            )

        diagonal_index = masks_tuple.index(0)
        diagonal = coeff[diagonal_index]
        if not np.allclose(diagonal.imag, 0.0, atol=atol):
            raise ValueError("diagonal amplitudes must be real")

        if shift is None:
            energies = diagonal.real
            shift = 0.5 * (float(np.max(energies)) + float(np.min(energies)))
        shift = float(shift)

        centered = coeff.copy()
        centered[diagonal_index] -= shift

        required_bounds = np.max(np.abs(centered), axis=1)
        if bounds is None:
            bound_array = required_bounds
        else:
            bound_array = np.asarray(bounds, dtype=float)
            if bound_array.shape != (len(masks_tuple),):
                raise ValueError("bounds must contain one value per move")
            if np.any(bound_array + atol < required_bounds):
                raise ValueError("a supplied bound is smaller than max_x |c_tilde|")

        if np.any(bound_array <= 0.0):
            raise ValueError("zero-amplitude moves must be removed from the move list")

        model = cls(
            n_qubits=n_qubits,
            masks=masks_tuple,
            coefficients=coeff,
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
        """Padded Hilbert-space dimension of the move register."""
        # Keep one physical qubit even for a single-move model.  The unused
        # state has zero PREP amplitude and MOVE acts as the identity there.
        return max(2, next_power_of_two(self.move_count))

    @property
    def move_qubits(self) -> int:
        return (self.move_dim - 1).bit_length()

    @property
    def alpha(self) -> float:
        return float(np.sum(self.bounds))

    def validate_hermiticity(self, *, atol: float = 1e-12) -> None:
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                reverse_x = x ^ mask
                lhs = self.coefficients[move_index, reverse_x]
                rhs = np.conjugate(self.coefficients[move_index, x])
                if not np.allclose(lhs, rhs, atol=atol):
                    raise ValueError(
                        "Hermiticity failed for "
                        f"mask={mask}, x={x}: c(x xor mu)={lhs}, conj(c(x))={rhs}"
                    )

    def hamiltonian(self) -> np.ndarray:
        matrix = np.zeros((self.system_dim, self.system_dim), dtype=np.complex128)
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                matrix[x ^ mask, x] += self.coefficients[move_index, x]
        return matrix

    def shifted_hamiltonian(self) -> np.ndarray:
        return self.hamiltonian() - self.shift * np.eye(self.system_dim)

    def target_block(self) -> np.ndarray:
        return self.shifted_hamiltonian() / self.alpha
