"""VALUE oracle for normalized, centered PMR move amplitudes."""

from __future__ import annotations

import numpy as np

from pmr_model import PMRModel, _finite_real_array


class ValueOracle:
    """Classical VALUE: c_tilde_mu(x) / b_mu, or zero on a zero-bound row.

    The model enforces conservative bounds independently of ``atol``. This
    tolerance only permits roundoff in the computed normalized magnitude;
    it never turns a nonzero row into an inactive row or repairs a bound.
    The model reference and tolerance are read-only, and ``table`` returns an
    independent copy, so public access cannot invalidate the cached values.
    """

    def __init__(self, model: PMRModel, *, atol: float = 1e-12) -> None:
        self._model = model
        tolerance = _finite_real_array(atol, name="VALUE atol")
        if tolerance.ndim != 0 or tolerance < 0:
            raise ValueError("VALUE atol must be a finite nonnegative scalar")
        self._atol = atol = float(tolerance)
        normalized = model.normalized_coefficients()
        if np.any(np.abs(normalized) > 1.0 + atol):
            raise ValueError("normalized VALUE output exceeds unit magnitude")
        self._normalized = normalized

    @property
    def model(self) -> PMRModel:
        """The immutable model used to compute this oracle's cached values."""
        return self._model

    @property
    def atol(self) -> float:
        """Magnitude tolerance used when constructing this oracle."""
        return self._atol

    def normalized_value(self, move_index: int, x: int) -> complex:
        if not 0 <= move_index < self.model.move_count:
            raise IndexError("move_index is outside the valid move list")
        if not 0 <= x < self.model.system_dim:
            raise IndexError("x is outside the system register")
        return complex(self._normalized[move_index, x])

    def table(self) -> np.ndarray:
        """Return a copy of the normalized VALUE lookup table."""
        return self._normalized.copy()
