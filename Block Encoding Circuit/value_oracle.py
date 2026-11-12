"""Normalized VALUE oracle data for the grouped PMR circuit."""

from __future__ import annotations

import numpy as np

from model import PMRModel


class ValueOracle:
    """Provide z_mu(x) = centered_coefficient_mu(x) / bound_mu."""

    def __init__(self, model: PMRModel, *, atol: float = 1e-12) -> None:
        self.model = model
        self._table = model.centered_coefficients / model.bounds[:, np.newaxis]
        if np.any(np.abs(self._table) > 1.0 + atol):
            raise ValueError("normalized VALUE output exceeds unit magnitude")

    def value(self, move_index: int, x: int) -> complex:
        """Return one normalized complex coefficient."""
        if not 0 <= move_index < self.model.move_count:
            raise IndexError("move_index is outside the move list")
        if not 0 <= x < self.model.system_dim:
            raise IndexError("x is outside the system register")
        return complex(self._table[move_index, x])

    def table(self) -> np.ndarray:
        """Return a safe copy of the complete lookup table."""
        return self._table.copy()
