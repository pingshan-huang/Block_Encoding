"""VALUE oracle for normalized, centered PMR move amplitudes."""

from __future__ import annotations

import numpy as np

from pmr_model import PMRModel, _finite_real_array


class ValueOracle:
    """Classical VALUE: c_tilde_mu(x) / b_mu, or zero on a zero-bound row.

    ``x`` labels the source state and each row follows the model's mask order.
    In exact arithmetic, the preparation weight ``b_mu / alpha`` times VALUE
    gives ``c_tilde_mu(x) / alpha``, the coefficient of the expected block
    ``(H - shift*I) / alpha``. This table doesn't include angle quantization
    or sparse-simulator pruning and isn't a quantum circuit implementation.

    The model checks that every bound covers its centered coefficients.
    ``atol`` only allows roundoff in the normalized magnitude; it doesn't
    treat tiny coefficients as zero or make an insufficient bound valid.
    The model reference and tolerance are read-only. ``table`` returns a copy,
    so callers can inspect or edit it without changing the cached values.
    """

    def __init__(self, model: PMRModel, *, atol: float = 1e-12) -> None:
        """Cache the model's normalized table after checking bounds and range.

        Supply a validated model; this doesn't repeat its Hermiticity check.
        ``atol`` must be finite, real, and nonnegative. It only allows roundoff
        in the unit-magnitude check, independently of the model's Hermiticity
        tolerance and exact bound-coverage checks.
        """
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
        """The model used to calculate the cached VALUE table."""
        return self._model

    @property
    def atol(self) -> float:
        """Tolerance used to check the normalized magnitude at construction."""
        return self._atol

    def normalized_value(self, move_index: int, x: int) -> complex:
        """Return one complex VALUE entry for a mask row and source state.

        ``move_index`` is the row's position, not the XOR mask itself. A
        zero-bound row returns zero. Indices outside the model's row or system
        range raise IndexError; padded move-register states aren't table rows.
        """
        if not 0 <= move_index < self.model.move_count:
            raise IndexError("move_index is outside the valid move list")
        if not 0 <= x < self.model.system_dim:
            raise IndexError("x is outside the system register")
        return complex(self._normalized[move_index, x])

    def table(self) -> np.ndarray:
        """Return a separate ``(move_count, system_dim)`` VALUE table.

        Rows follow the supplied masks, without padded move states. Changing
        the returned array won't affect the cache or later scalar lookups.
        """
        return self._normalized.copy()
