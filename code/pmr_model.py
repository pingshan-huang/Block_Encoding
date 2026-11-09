"""Data model and validation for a grouped PMR Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _integer(value: object, *, name: str) -> int:
    """Accept Python/NumPy integers without truncation or boolean coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer (not a boolean)")
    return int(value)


def _finite_real_array(values: object, *, name: str) -> np.ndarray:
    """Convert real-valued data without discarding an imaginary component."""
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
    tolerance = _finite_real_array(atol, name="Hermiticity atol")
    if tolerance.ndim != 0 or tolerance < 0:
        raise ValueError("Hermiticity atol must be a finite nonnegative real scalar")
    return float(tolerance)


def _validate_hermiticity(
    coefficients: np.ndarray, masks: tuple[int, ...], *, atol: float
) -> None:
    """Check every source-state residual with absolute tolerance, rtol=0."""
    for row, mask in enumerate(masks):
        for x in range(coefficients.shape[1]):
            lhs = coefficients[row, x ^ mask]
            rhs = np.conjugate(coefficients[row, x])
            # An overflowing residual cannot fit any finite tolerance. Treat
            # it as a failed check without leaking a floating-point warning.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                residual = float(np.abs(lhs - rhs))
            if not np.isfinite(residual) or residual > atol:
                raise ValueError(
                    "Hermiticity failed for "
                    f"mask={mask}, x={x}: |c(x xor mu)-conj(c(x))|="
                    f"{residual} exceeds atol={atol} (rtol=0)"
                )


def _bound_covers(value: complex, bound: float) -> bool:
    """Compare |value| <= bound exactly for the represented binary floats.

    Squared integer ratios avoid overflow, underflow, and a rounded complex
    magnitude accepting a slightly insufficient bound. This is a validation
    check; coefficients and all numerical operations remain complex128.
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
    """Reject nonfinite results and loss of a represented nonzero component."""
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{context} produced nonfinite coefficients")
    if np.any((original.real != 0) & (result.real == 0)) or np.any(
        (original.imag != 0) & (result.imag == 0)
    ):
        raise ValueError(f"{context}: nonzero coefficient component underflowed to zero")


def _divide_coefficients(
    coefficients: np.ndarray, divisors: object, *, context: str
) -> np.ndarray:
    """Divide by real nonnegative scales without losing nonzero components.

    Real and imaginary parts are divided separately: complex division can
    overflow internally even when division by a subnormal real scale has a
    representable result. Zero divided by an inactive zero scale is defined
    as zero. Other nonfinite results and component underflow to zero are
    unsupported numerical ranges, rather than silently accepted truncation.
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
    """A dataclass field that stores owned data and exposes independent copies.

    Keeping the public field names preserves dataclass construction, ``replace``
    and ``asdict``. Separate array headers also protect stored shape and dtype;
    making the same publicly exposed ndarray read-only would not do that.
    """

    def __init__(self, dtype: type) -> None:
        self.dtype = dtype

    def __set_name__(self, owner: type, name: str) -> None:
        self.storage_name = "_" + name

    def __get__(self, instance: object, owner: type | None = None) -> np.ndarray:
        if instance is None:
            # Dataclasses interprets AttributeError here as a required field.
            raise AttributeError("array fields require an instance")
        return getattr(instance, self.storage_name).copy()

    def __set__(self, instance: object, values: object) -> None:
        owned = np.array(values, dtype=self.dtype, copy=True)
        owned.setflags(write=False)
        object.__setattr__(instance, self.storage_name, owned)


@dataclass(frozen=True)
class PMRModel:
    """Move-amplitude representation of a Hermitian Hamiltonian.

    ``coefficients[j, x]`` is the amplitude ``c_mu(x)`` for the move whose
    bit mask is ``masks[j]``.  It contributes

        c_mu(x) |x xor mu><x|.

    Qubit counts and masks must be Python/NumPy integers, excluding booleans.
    The diagonal move ``mu = 0`` must be present. Masks must be unique;
    duplicate masks are an unsupported representation, not evidence that the
    summed Hamiltonian is non-Hermitian. Rows retain their input order,
    including inactive rows. Hermiticity uses the entrywise criterion
    ``abs(c_mu(x xor mu) - conjugate(c_mu(x))) <= atol`` with ``rtol=0``.
    The same criterion applies to diagonal rows, where the residual is
    ``2 * abs(imag(c_0(x)))``. The finite, nonnegative scalar ``atol`` defaults
    to 1e-12; accepted residuals are not silently projected away.

    Bounds cover the centered complex128 coefficients, not the unshifted
    input. A zero bound is permitted only
    for an exactly zero centered row; zero rows may also have positive bounds.

    Automatic bounds start at the computed maximum magnitude and round up
    only as needed for conservative coverage of the represented coefficients.
    Supplied bounds must satisfy the same coverage check and are never enlarged.
    The all-zero centered target requires explicit bounds with a positive sum.
    Numerical ranges that overflow or erase a nonzero component during VALUE
    or target normalization are rejected. The separate sparse simulator's
    pruning and fixed-point approximation are outside this contract.

    Use ``from_coefficients`` for validated construction. The low-level
    initializer accepts already validated, consistently centered tables.
    Both entry points take ownership without changing caller array flags.
    Public array fields return independent copies; editing them, including
    their flags or metadata, cannot change the model. Construct a new model
    to change its Hamiltonian or normalization.
    """

    n_qubits: int
    masks: tuple[int, ...]
    coefficients: np.ndarray = _OwnedArray(np.complex128)
    centered_coefficients: np.ndarray = _OwnedArray(np.complex128)
    bounds: np.ndarray = _OwnedArray(np.float64)
    shift: float

    def __post_init__(self) -> None:
        """Own scalar/sequence inputs too; the factory performs validation."""
        object.__setattr__(self, "n_qubits", _integer(self.n_qubits, name="n_qubits"))
        object.__setattr__(
            self, "masks", tuple(_integer(mask, name="move mask") for mask in self.masks)
        )
        object.__setattr__(self, "shift", float(self.shift))

    def __eq__(self, other: object) -> bool:
        # Preserve the original dataclass comparison of stored fields.
        # Comparing public array copies would even break model == model.
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
            # Halve first only if addition overflows. Otherwise sum first to
            # preserve equal subnormal endpoints when computing the midpoint.
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
        """Return validated VALUE coefficients; inactive zero-weight rows are zero.

        Coverage is independent of Hermiticity tolerances. Positive bounds on
        zero rows remain in alpha and preparation weights. Both row and total
        normalization must preserve every nonzero real/imaginary component.
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
        # Two separately rounded operations can erase a component even when
        # direct division by alpha does not. Check the actual weights*VALUE
        # contract as well, independently of sparse simulation or quantization.
        with np.errstate(under="ignore"):
            weighted = normalized * weights[:, np.newaxis]
        _check_preserved_components(
            centered, weighted, context="weighted VALUE reconstruction"
        )
        return normalized

    def validate_hermiticity(self, *, atol: float = 1e-12) -> None:
        """Check the documented entrywise residual with ``rtol=0``.

        This tolerance never determines whether a centered row is inactive
        and never relaxes the conservative coefficient-bound requirement.
        """
        _validate_hermiticity(
            self._coefficients, self.masks, atol=_hermiticity_tolerance(atol)
        )

    def hamiltonian(self) -> np.ndarray:
        matrix = np.zeros((self.system_dim, self.system_dim), dtype=np.complex128)
        for move_index, mask in enumerate(self.masks):
            for x in range(self.system_dim):
                matrix[x ^ mask, x] += self._coefficients[move_index, x]
        return matrix

    def shifted_hamiltonian(self) -> np.ndarray:
        return self.hamiltonian() - self.shift * np.eye(self.system_dim)

    def target_block(self) -> np.ndarray:
        return _divide_coefficients(
            self.shifted_hamiltonian(), self.alpha, context="target normalization"
        )
