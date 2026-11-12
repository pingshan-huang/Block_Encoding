"""Data model and validation for a grouped PMR Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _integer(value: object, *, name: str) -> int:
    """Require a Python/NumPy integer without accepting floats or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer (not a boolean)")
    return int(value)


def _finite_real_array(values: object, *, name: str) -> np.ndarray:
    """Read finite real values without silently dropping an imaginary part."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        if np.any(raw.imag != 0):
            raise ValueError(f"{name} must be real-valued")
        raw = raw.real
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real values")
    return result


def _hermiticity_tolerance(atol: object) -> float:
    """Read a finite, nonnegative scalar tolerance without dropping imaginary parts."""
    tolerance = _finite_real_array(atol, name="Hermiticity atol")
    if tolerance.ndim != 0 or tolerance < 0:
        raise ValueError("Hermiticity atol must be a finite nonnegative real scalar")
    return float(tolerance)


def _validate_hermiticity(
    coefficients: np.ndarray, masks: tuple[int, ...], *, atol: float
) -> None:
    """Check each conjugate pair with absolute tolerance only (rtol=0)."""
    for row, mask in enumerate(masks):
        for x in range(coefficients.shape[1]):
            lhs = coefficients[row, x ^ mask]
            rhs = np.conjugate(coefficients[row, x])
            # If the residual overflows, no finite tolerance can accept it.
            # Report a failed Hermiticity check instead of a NumPy warning.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                residual = float(np.abs(lhs - rhs))
            if not np.isfinite(residual) or residual > atol:
                raise ValueError(
                    "Hermiticity failed for "
                    f"mask={mask}, x={x}: |c(x xor mu)-conj(c(x))|="
                    f"{residual} exceeds atol={atol} (rtol=0)"
                )


def _bound_covers(value: complex, bound: float) -> bool:
    """Check |value| <= bound exactly for the floats we actually store.

    A rounded complex magnitude can make a slightly too-small bound look valid.
    Compare squared integer ratios to avoid that rounding, overflow, and
    underflow. This only checks coverage; it doesn't change the complex128
    coefficients or switch the rest of the calculation to exact arithmetic.
    """
    if value.imag == 0:
        return abs(value.real) <= bound
    if value.real == 0:
        return abs(value.imag) <= bound
    real, real_den = float(value.real).as_integer_ratio()
    imag, imag_den = float(value.imag).as_integer_ratio()
    limit, limit_den = float(bound).as_integer_ratio()
    denominator = max(real_den, imag_den, limit_den)
    return (
        (real * (denominator // real_den)) ** 2
        + (imag * (denominator // imag_den)) ** 2
        <= (limit * (denominator // limit_den)) ** 2
    )


def _check_preserved_components(
    original: np.ndarray, result: np.ndarray, *, context: str
) -> None:
    """Reject nonfinite results or a nonzero real/imaginary part lost to zero."""
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} produced nonfinite coefficients")
    if np.any((original.real != 0) & (result.real == 0)) or np.any(
        (original.imag != 0) & (result.imag == 0)
    ):
        raise ValueError(f"{context}: nonzero coefficient component underflowed to zero")


def _divide_coefficients(
    coefficients: np.ndarray, divisors: object, *, context: str
) -> np.ndarray:
    """Divide by nonnegative real scales, checking for lost components.

    Divide the real and imaginary parts separately. Complex division can
    overflow internally for a tiny denominator even when the answer fits in
    a float. A zero coefficient with a zero scale returns zero without division.
    Reject nonfinite answers or a nonzero component that underflows to zero;
    silently losing it would change the operator we claim to represent.
    """
    scales = np.asarray(divisors, dtype=np.float64)
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(scales)):
        raise ValueError(f"{context} requires finite coefficients and scales")
    if np.any(scales < 0) or np.any((scales == 0) & (coefficients != 0)):
        raise ValueError(f"{context}: zero scales require exactly zero coefficients")
    result = np.zeros_like(coefficients, dtype=np.complex128)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            np.divide(coefficients.real, scales, out=result.real, where=scales > 0)
            np.divide(coefficients.imag, scales, out=result.imag, where=scales > 0)
    except FloatingPointError as exc:
        raise ValueError(f"{context}: unsupported numerical range") from exc
    _check_preserved_components(coefficients, result, context=context)
    return result


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


class _OwnedArray:
    """Keep our own array and return a separate copy whenever it is read.

    This keeps the public field names working with dataclass construction,
    ``replace``, and ``asdict``. A read-only flag alone isn't enough: callers
    could still change an array's shape or dtype. Returning a copy protects
    those too.
    """

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype

    def __set_name__(self, owner: type, name: str) -> None:
        self.storage_name = "_" + name

    def __get__(self, instance: object, owner: type | None = None) -> np.ndarray:
        if instance is None:
            # This tells dataclasses that callers must supply the field.
            raise AttributeError("array fields require an instance")
        return getattr(instance, self.storage_name).copy()

    def __set__(self, instance: object, values: object) -> None:
        owned = np.array(values, dtype=self.dtype, copy=True)
        owned.setflags(write=False)
        object.__setattr__(instance, self.storage_name, owned)


@dataclass(frozen=True)
class PMRModel:
    """Store grouped XOR coefficients for a numerically Hermitian Hamiltonian.

    ``coefficients[j, x]`` is the amplitude ``c_mu(x)`` for the move whose
    bit mask is ``masks[j]``. Here ``x`` is the source state, so the term is

        c_mu(x) |x xor mu><x|.

    Qubit counts and masks must be Python/NumPy integers, not booleans. Include
    the diagonal move ``mu = 0`` and use each mask only once. Duplicate masks
    aren't supported, even if their summed Hamiltonian would be Hermitian.
    Rows stay in input order, including rows with zero coefficients.

    Hermiticity requires ``abs(c_mu(x xor mu) - conjugate(c_mu(x))) <= atol``
    at every entry, with ``rtol=0``. For the diagonal row, this difference is
    ``2 * abs(imag(c_0(x)))``. The finite, nonnegative scalar ``atol`` defaults
    to 1e-12. Differences within tolerance are kept; we don't alter the input
    to make it exactly Hermitian.

    Bounds must cover the centered complex128 coefficients. A zero bound is
    allowed only when every centered coefficient in that row is exactly zero.
    A zero row may also have a positive bound, which still counts toward alpha.
    Automatic bounds start at the computed maximum magnitude and round up
    only as needed to cover the stored floats. Supplied bounds face the same
    coverage check, but we reject insufficient ones instead of enlarging them.
    An all-zero centered target needs explicit bounds with a positive sum.
    The target is ``(H - shift*I) / alpha``, where ``alpha = sum(bounds)``.

    VALUE and target normalization reject overflow or the loss of a nonzero
    real/imaginary component to zero. These checks are separate from the sparse
    simulator's pruning and fixed-point approximation.

    Use ``from_coefficients`` to validate inputs. The direct initializer expects
    tables that are already checked and consistently centered. Both keep their
    own arrays without changing the caller's array flags. Public array fields
    return copies, so changing their values, flags, shape, or dtype won't change
    the model. Create a new model to change its Hamiltonian or normalization.
    """

    n_qubits: int
    masks: tuple[int, ...]
    coefficients: np.ndarray = _OwnedArray(np.complex128)
    centered_coefficients: np.ndarray = _OwnedArray(np.complex128)
    bounds: np.ndarray = _OwnedArray(np.float64)
    shift: float

    def __post_init__(self) -> None:
        """Copy scalar and mask inputs too; use the factory for validation."""
        object.__setattr__(self, "n_qubits", _integer(self.n_qubits, name="n_qubits"))
        object.__setattr__(
            self, "masks", tuple(_integer(mask, name="move mask") for mask in self.masks)
        )
        object.__setattr__(self, "shift", float(self.shift))

    def __eq__(self, other: object) -> bool:
        """Keep the original dataclass comparison, including its NumPy limitation.

        Self-comparison works. Comparing distinct models can raise ValueError
        because a multi-element array has no single truth value.
        """
        # Compare stored fields as the original dataclass did. Reading the
        # public copies here would make even model == model fail in NumPy.
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (
            self.n_qubits, self.masks, self._coefficients,
            self._centered_coefficients, self._bounds, self.shift,
        ) == (
            other.n_qubits, other.masks, other._coefficients,
            other._centered_coefficients, other._bounds, other.shift,
        )

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
        """Build an owned model after checking its coefficients and normalization.

        Supply one row per unique mask and one column per source basis state:
        the table shape is ``(len(masks), 2**n_qubits)``. Include mask zero for
        the diagonal. Rows keep their supplied order, including inactive rows.

        With no supplied shift, use the midpoint of the smallest and largest
        real diagonal entries. Subtract it only from the mask-zero row. This
        removes a scalar identity term and can reduce that row's bound; the
        encoded target is then ``(H - shift*I) / alpha`` rather than ``H / alpha``.

        Bounds apply after centering. Automatic bounds start at each row's
        largest magnitude and round upward as needed to cover the stored
        floats. Supplied bounds must already cover the row; they aren't raised
        to make them valid. A zero bound needs an exactly zero row. If all
        centered rows are zero, supply explicit bounds with a positive sum.

        ``atol`` controls the absolute conjugate-pair check described on the
        class, with no relative tolerance. Passing means Hermiticity within
        that tolerance, not an exact Hermitian projection. The returned model
        owns copies of its arrays. Unsupported normalization ranges raise
        ValueError instead of silently losing nonzero components.
        """
        n_qubits = _integer(n_qubits, name="n_qubits")
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")

        atol = _hermiticity_tolerance(atol)
        try:
            mask_iterator = iter(masks)
        except TypeError as exc:
            raise ValueError("masks must be an iterable of integer move masks") from exc
        masks_tuple = tuple(_integer(mask, name="move mask") for mask in mask_iterator)
        if not masks_tuple:
            raise ValueError("at least one move is required")
        if len(set(masks_tuple)) != len(masks_tuple):
            raise ValueError(
                "duplicate move masks are an unsupported input representation; "
                "move masks must be unique"
            )

        system_dim = 1 << n_qubits
        if any(mask < 0 or mask >= system_dim for mask in masks_tuple):
            raise ValueError("each move mask must fit in n_qubits")
        if 0 not in masks_tuple:
            raise ValueError("the diagonal move mu=0 must be included")

        try:
            with np.errstate(over="raise", invalid="raise"):
                coeff = np.asarray(coefficients, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            raise ValueError("coefficients must contain finite complex128 values") from exc
        expected_shape = (len(masks_tuple), system_dim)
        if coeff.shape != expected_shape:
            raise ValueError(
                f"coefficients must have shape {expected_shape}, got {coeff.shape}"
            )
        if not np.all(np.isfinite(coeff)):
            raise ValueError("coefficients must be finite")

        _validate_hermiticity(coeff, masks_tuple, atol=atol)
        diagonal_index = masks_tuple.index(0)
        diagonal = coeff[diagonal_index]

        if shift is None:
            energies = diagonal.real
            low, high = float(np.min(energies)), float(np.max(energies))
            total = low + high
            # Add before halving so equal tiny endpoints keep their midpoint.
            # If the sum overflows, halve the endpoints before adding instead.
            shift = total * 0.5 if np.isfinite(total) else low * 0.5 + high * 0.5
        shift_array = _finite_real_array(shift, name="shift")
        if shift_array.ndim != 0:
            raise ValueError("shift must be a finite real scalar")
        shift = float(shift_array)

        centered = coeff.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                centered[diagonal_index] -= shift
        except FloatingPointError as exc:
            raise ValueError("centering exceeds the supported numerical range") from exc
        if not np.all(np.isfinite(centered)):
            raise ValueError("centered coefficients must be finite")

        if bounds is None:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    bound_array = np.max(np.abs(centered), axis=1)
                if not np.all(np.isfinite(bound_array)):
                    raise ValueError("no finite bound covers a centered row")
                for row, bound in enumerate(bound_array):
                    while not all(_bound_covers(z, bound) for z in centered[row]):
                        with np.errstate(over="raise"):
                            bound = np.nextafter(bound, np.inf)
                        if not np.isfinite(bound):
                            raise ValueError("no finite bound covers a centered row")
                    bound_array[row] = bound
            except FloatingPointError as exc:
                raise ValueError("no finite bound covers a centered row") from exc
            if np.all(centered == 0):
                raise ValueError(
                    "zero centered target has zero automatic normalization; "
                    "supply explicit bounds with a positive sum"
                )
        else:
            bound_array = _finite_real_array(bounds, name="bounds")
            if bound_array.shape != (len(masks_tuple),):
                raise ValueError("bounds must contain one value per move")

        model = cls(
            n_qubits=n_qubits,
            masks=masks_tuple,
            coefficients=coeff,
            centered_coefficients=centered,
            bounds=np.asarray(bound_array, dtype=float),
            shift=shift,
        )
        model.normalized_coefficients()
        return model

    @property
    def system_dim(self) -> int:
        """Number of computational-basis states in the system register."""
        return 1 << self.n_qubits

    @property
    def move_count(self) -> int:
        """Number of supplied mask rows, including rows with zero weight."""
        return len(self.masks)

    @property
    def move_dim(self) -> int:
        """Size of the move register after padding to a power of two."""
        # Keep one physical qubit even for a single-move model. The unused
        # state has zero PREP amplitude and MOVE acts as the identity there.
        return max(2, next_power_of_two(self.move_count))

    @property
    def move_qubits(self) -> int:
        """Number of qubits needed for the padded move register."""
        return (self.move_dim - 1).bit_length()

    @property
    def alpha(self) -> float:
        """Return the finite, positive sum of row bounds used to scale the target.

        A positive bound still counts even when its centered row is zero.
        Reject a nonpositive or overflowing total instead of using epsilon.
        """
        try:
            with np.errstate(over="raise", invalid="raise"):
                alpha = float(np.sum(self._bounds))
        except FloatingPointError as exc:
            raise ValueError("total normalization alpha must be finite and positive") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                "total normalization alpha must be finite and positive; "
                "a zero centered target requires explicit positive bounds"
            )
        return alpha

    def normalized_coefficients(self) -> np.ndarray:
        """Return the VALUE table ``v_mu(x) = c_tilde_mu(x) / b_mu``.

        Zero-bound rows return zero without division. Hermiticity tolerances
        don't relax the bound checks. A positive bound on a zero row still
        contributes to alpha and preparation weights. Reject normalization
        that loses a nonzero real or imaginary component to zero.

        In exact arithmetic, weighting each row by ``b_mu / alpha`` gives the
        source coefficients of ``(H - shift*I) / alpha``. This method also checks
        the rounded weight-times-VALUE calculation for lost components. It
        returns a separate array; changing it won't affect the model.
        """
        centered = self._centered_coefficients
        bounds = self._bounds
        if not np.all(np.isfinite(centered)) or not np.all(np.isfinite(bounds)):
            raise ValueError("centered coefficients and bounds must be finite")
        if np.any(bounds < 0):
            raise ValueError("bounds must be nonnegative")
        zero_rows = np.all(centered == 0, axis=1)
        if np.any((bounds == 0) & ~zero_rows):
            raise ValueError("a zero bound requires an exactly zero centered row")
        for row, bound in zip(centered, bounds):
            if not all(_bound_covers(z, bound) for z in row):
                raise ValueError("a bound is smaller than a centered coefficient magnitude")

        alpha = self.alpha
        with np.errstate(under="ignore"):
            weights = bounds / alpha
        if not np.all(np.isfinite(weights)) or np.any((bounds > 0) & (weights == 0)):
            raise ValueError("preparation weight normalization underflowed or is nonfinite")
        normalized = _divide_coefficients(
            centered, bounds[:, np.newaxis], context="VALUE normalization"
        )
        _divide_coefficients(centered, alpha, context="target normalization")
        # Dividing by a row bound and then multiplying by its weight can lose
        # a tiny component even when direct division by alpha preserves it.
        # Check that two-step result before any sparse pruning or quantization.
        with np.errstate(under="ignore"):
            weighted = normalized * weights[:, np.newaxis]
        _check_preserved_components(
            centered, weighted, context="weighted VALUE reconstruction"
        )
        return normalized

    def validate_hermiticity(self, *, atol: float = 1e-12) -> None:
        """Raise ValueError if a source-indexed conjugate pair exceeds ``atol``.

        For unique XOR masks, check
        ``abs(c_mu(x xor mu) - conjugate(c_mu(x))) <= atol`` at every entry,
        with ``rtol=0``. For mask zero the difference is twice the imaginary
        magnitude. ``atol`` must be a finite, nonnegative real scalar.

        Returning normally means the stored coefficients pass this numerical
        check; it doesn't make them exactly Hermitian or modify any entries.
        This tolerance doesn't decide whether a row is zero and doesn't allow
        a bound to be smaller than a coefficient's magnitude.
        """
        _validate_hermiticity(
            self._coefficients, self.masks, atol=_hermiticity_tolerance(atol)
        )

    def hamiltonian(self) -> np.ndarray:
        """Reconstruct a new dense matrix from source-indexed coefficients.

        ``c_mu(x)`` contributes to matrix entry ``[x xor mu, x]``: the row is
        the destination and the column is the source. No centering or scaling
        is applied, and any within-tolerance Hermiticity differences remain.
        The matrix has ``2**n_qubits`` rows and columns, so this is a small-system
        reference calculation, not a scalable circuit construction.
        """
        matrix = np.zeros((self.system_dim, self.system_dim), dtype=np.complex128)
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                matrix[x ^ mask, x] += self._coefficients[move_index, x]
        return matrix

    def shifted_hamiltonian(self) -> np.ndarray:
        """Return a new dense ``H - shift*I`` matrix in the system basis.

        Subtracting this energy offset changes only diagonal entries. The
        automatic shift centers their real range; a caller-supplied shift is
        used as given and need not center that range around zero.
        """
        return self.hamiltonian() - self.shift * np.eye(self.system_dim)

    def target_block(self) -> np.ndarray:
        """Return the expected dense block ``(H - shift*I) / alpha``.

        This is the mathematical reference for projecting all ancillas onto
        zero. It doesn't simulate the circuit or include fixed-point encoding
        and sparse pruning. Division rejects nonfinite results or a nonzero
        component lost to zero. As with reconstruction, memory grows as 4**n.
        """
        return _divide_coefficients(
            self.shifted_hamiltonian(), self.alpha, context="target normalization"
        )
