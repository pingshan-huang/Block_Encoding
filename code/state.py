"""Sparse state representation shared by the block-encoding primitives."""

from __future__ import annotations  # Enable modern type annotations.

BasisState = tuple[int, int, int, int]  # Registers are (flag, b, move, system).
SparseState = dict[BasisState, complex]  # Store only nonzero basis amplitudes.


def add_amplitude(  # Add and combine an amplitude in a sparse state.
    state: SparseState,  # Receive the destination state.
    basis: BasisState,  # Receive the destination basis key.
    amplitude: complex,  # Receive the amplitude to add.
    *,  # Require a named tolerance.
    atol: float = 1e-15,  # Drop negligible numerical noise.
) -> None:  # Update the state in place.
    value = state.get(basis, 0.0j) + amplitude  # Combine duplicate paths.
    if abs(value) <= atol:  # Detect a numerical zero.
        state.pop(basis, None)  # Remove the negligible entry.
    else:  # Keep a physically relevant amplitude.
        state[basis] = value  # Store the combined result.
