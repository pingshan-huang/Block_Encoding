"""VALUE oracle for normalized, centered PMR move amplitudes."""

from __future__ import annotations

import numpy as np

from pmr_model import PMRModel


class ValueOracle:
    """Classical reference for VALUE: c_tilde_mu(x) / b_mu."""

    def __init__(self, model: PMRModel, *, atol: float = 1e-12) -> None:
        self.model = model
        self.atol = atol
        normalized = model.centered_coefficients / model.bounds[:, np.newaxis]
        if np.any(np.abs(normalized) > 1.0 + atol):
            raise ValueError("normalized VALUE output exceeds unit magnitude")
        self._normalized = normalized

    def normalized_value(self, move_index: int, x: int) -> complex:
        if not 0 <= move_index < self.model.move_count:
            raise IndexError("move_index is outside the valid move list")
        if not 0 <= x < self.model.system_dim:
            raise IndexError("x is outside the system register")
        return complex(self._normalized[move_index, x])

    def table(self) -> np.ndarray:
        """Return a copy of the normalized VALUE lookup table."""
        return self._normalized.copy()
