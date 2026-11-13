"""Shared Qiskit primitives for Antonio's grouped PMR circuit designs.

Adapted from the two implementations in Antonio commit 9b091ff. The local
review keeps their unitary extension on unused states, shares the validated
mathematical core, and cancels redundant control-pattern X gates. VALUE is
classical construction data; the reversible lookup is implemented by r.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import PhaseGate, RYGate, StatePreparation, XGate
from qiskit.quantum_info import Statevector

from pmr_model import PMRModel
from rotation_codec import AnglePhaseCodec, RotationData
from value import ValueOracle


@dataclass(frozen=True)
class QiskitAnglePhaseCodec(AnglePhaseCodec):
    """Use the shared angle convention within its supported numerical range.

    Each field needs 1--52 bits and their sum must fit a uint64 lookup word.
    More theta bits can round the endpoint code past its field and corrupt
    phase bits in float64 arithmetic. Reject that range instead of silently
    accepting it. This bound does not promise useful extra accuracy at every
    precision: angles and phases are still calculated with ordinary floats.
    At 52 theta bits, even the zero-value endpoint can round one code low;
    trigonometric evaluation also leaves residuals near machine precision.
    """

    def __post_init__(self) -> None:
        """Reject widths that cannot be safely represented by this float/word codec."""
        for name in ("theta_bits", "phase_bits"):
            width = getattr(self, name)
            if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
                raise ValueError(f"{name} must be an integer between 1 and 52")
            if not 1 <= int(width) <= 52:
                raise ValueError(f"{name} must be between 1 and 52 for float64 encoding")
            object.__setattr__(self, name, int(width))
        if self.workspace_bits > 64:
            raise ValueError("the packed lookup supports at most 64 workspace bits")


@dataclass(frozen=True)
class QiskitRTable:
    """Own the classical packed words used to synthesize reversible r.

    This table is not a quantum memory oracle: construction enumerates every
    supplied move and source state. Padded move rows contain word zero. Public
    array access returns a copy so lookup data cannot be changed by a view.
    """

    value_oracle: ValueOracle
    codec: QiskitAnglePhaseCodec
    _words: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Encode validated VALUE once and take ownership of the padded lookup."""
        QiskitAnglePhaseCodec(self.codec.theta_bits, self.codec.phase_bits)
        words = np.zeros((self.model.move_dim, self.model.system_dim), dtype=np.uint64)
        values = self.value_oracle.table()
        for move in range(self.model.move_count):
            for x in range(self.model.system_dim):
                words[move, x] = self.codec.encode_value(complex(values[move, x]))
        words.setflags(write=False)
        object.__setattr__(self, "_words", words)

    @property
    def model(self) -> PMRModel:
        """The validated mathematical model supplying VALUE."""
        return self.value_oracle.model

    @property
    def workspace_bits(self) -> int:
        """Number of angle and phase bits in each packed word."""
        return self.codec.workspace_bits

    @property
    def words(self) -> np.ndarray:
        """Return a copy of the padded move-by-source lookup table."""
        return self._words.copy()

    def lookup_word(self, move: int, x: int) -> int:
        """Return a packed word, including zero for padded move rows."""
        if not 0 <= move < self.model.move_dim or not 0 <= x < self.model.system_dim:
            raise IndexError("lookup index is outside the move or system register")
        return int(self._words[move, x])

    def decoded_rotation(self, move: int, x: int) -> RotationData:
        """Decode a lookup word for numerical inspection."""
        return self.codec.decode_word(self.lookup_word(move, x))

    def active_words(self) -> set[int]:
        """Words reached by real move rows, including zero-coefficient rows."""
        return {int(word) for word in self._words[: self.model.move_count].flat}


def _check_registers(circuit: QuantumCircuit, *registers: tuple[Sequence[Qubit], int]) -> None:
    """Reject wrong widths, overlapping registers, or qubits absent from a circuit."""
    seen: set[Qubit] = set()
    for register, width in registers:
        if len(register) != width:
            raise ValueError(f"register has width {len(register)}; expected {width}")
        for qubit in register:
            if qubit in seen or qubit not in circuit.qubits:
                raise ValueError("registers must use distinct qubits present in the circuit")
            seen.add(qubit)


def _gray_rank(pattern: int) -> int:
    """Locate a pattern on a reflected Gray path without enumerating that path."""
    rank = 0
    while pattern:
        rank ^= pattern
        pattern >>= 1
    return rank


def _pattern_walk(
    circuit: QuantumCircuit, controls: Sequence[Qubit], patterns: Sequence[int]
) -> Iterator[int]:
    """Visit exact control patterns while retaining useful X inversions.

    Each pattern denotes a disjoint control subspace, so its block commutes
    with every other pattern's block. Only flip bits that change between
    patterns, then restore the controls. The X masks telescope on every input,
    including superpositions and unused states. Sorting costs O(K log K),
    not O(2**len(controls)); K is the number of actual table patterns.
    """
    full_mask = (1 << len(controls)) - 1
    if any(pattern < 0 or pattern > full_mask for pattern in patterns):
        raise ValueError("control pattern does not fit the register")
    patterns = sorted(set(patterns), key=_gray_rank)
    inverted = 0
    for pattern in patterns:
        wanted = full_mask ^ pattern
        changed = inverted ^ wanted
        for bit, qubit in enumerate(controls):
            if (changed >> bit) & 1:
                circuit.x(qubit)
        inverted = wanted
        yield pattern
    for bit, qubit in enumerate(controls):
        if (inverted >> bit) & 1:
            circuit.x(qubit)


def _append_controlled_gate(
    circuit: QuantumCircuit, gate: Gate, controls: Sequence[Qubit], targets: Sequence[Qubit]
) -> None:
    """Keep controlled operations symbolic until a backend decomposes them."""
    operation = gate.control(len(controls), annotated=True) if controls else gate
    circuit.append(operation, [*controls, *targets])


def append_prep(
    circuit: QuantumCircuit, move: QuantumRegister, model: PMRModel, *, inverse: bool = False
) -> None:
    """Append the unitary PREP or its inverse on the move register.

    PREP's zero-input column is sqrt(b_mu/alpha), padded with zeros. Use
    StatePreparation rather than reset-based initialization so PREP dagger
    remains a true inverse on the complete move-register Hilbert space.
    """
    _check_registers(circuit, (move, model.move_qubits))
    amplitudes = np.zeros(model.move_dim, dtype=np.complex128)
    amplitudes[: model.move_count] = np.sqrt(model.bounds / model.alpha)
    gate = StatePreparation(amplitudes, label="PREP")
    if inverse:
        gate = gate.inverse()
        gate.label = "PREP_dg"
    circuit.append(gate, list(move))


def append_r(
    circuit: QuantumCircuit, workspace: QuantumRegister, move: QuantumRegister,
    system: QuantumRegister, r_table: QiskitRTable,
) -> None:
    """Append |b,mu,x> -> |b XOR word(mu,x),mu,x> for arbitrary workspace b.

    The row index is stored in low control bits and the source state in high
    control bits. Zero words and padded moves act as identity. Every target
    operation is X, so applying this same oracle twice uncomputes the word.
    """
    _check_registers(circuit, (workspace, r_table.workspace_bits),
                     (move, r_table.model.move_qubits), (system, r_table.model.n_qubits))
    controls = [*move, *system]
    words = {}
    for row in range(r_table.model.move_count):
        for x in range(r_table.model.system_dim):
            word = r_table.lookup_word(row, x)
            if word != 0:
                words[row | (x << len(move))] = word
    for pattern in _pattern_walk(circuit, controls, list(words)):
        word = words[pattern]
        for bit, qubit in enumerate(workspace):
            if (word >> bit) & 1:
                _append_controlled_gate(circuit, XGate(), controls, [qubit])


def append_flag_rotation(
    circuit: QuantumCircuit, flag: QuantumRegister, workspace: QuantumRegister,
    r_table: QiskitRTable,
) -> None:
    """Append R = diag(exp(i*phi),1) Ry(theta) for each loaded word.

    The flag-zero amplitude is the quantized value cos(theta/2)*exp(i*phi),
    not the unquantized VALUE in general. Other workspace words act as identity.
    Only an exactly zero encoded field is omitted: a numerical closeness test
    would erase valid small phases at high precision. No workspace bit changes.
    """
    _check_registers(circuit, (flag, 1), (workspace, r_table.workspace_bits))
    rotations = {word: r_table.codec.decode_word(word) for word in r_table.active_words()}
    rotations = {word: data for word, data in rotations.items()
                 if data.theta_code != 0 or data.phase_code != 0}
    for word in _pattern_walk(circuit, list(workspace), list(rotations)):
        data = rotations[word]
        if data.theta_code != 0:
            _append_controlled_gate(circuit, RYGate(data.theta), list(workspace), [flag[0]])
        if data.phase_code != 0:
            # Conjugating Phase by X puts the phase on flag zero, preserving
            # its complex sign. It is not a global phase of the whole circuit.
            circuit.x(flag[0])
            _append_controlled_gate(circuit, PhaseGate(data.phase), list(workspace), [flag[0]])
            circuit.x(flag[0])


def append_move(
    circuit: QuantumCircuit, move: QuantumRegister, system: QuantumRegister, model: PMRModel
) -> None:
    """Apply the supplied XOR mask controlled by its row index.

    Padded rows and mask zero act as identity. Controls are never targets, so
    MOVE is an involution on the full space, not just on the prepared move state.
    """
    _check_registers(circuit, (move, model.move_qubits), (system, model.n_qubits))
    masks = {row: mask for row, mask in enumerate(model.masks) if mask != 0}
    for row in _pattern_walk(circuit, list(move), list(masks)):
        for bit, qubit in enumerate(system):
            if (masks[row] >> bit) & 1:
                _append_controlled_gate(circuit, XGate(), list(move), [qubit])


def extract_encoded_block(
    circuit: QuantumCircuit, system: Sequence[Qubit], *, max_qubits: int = 16
) -> np.ndarray:
    """Extract the zero-ancilla block with bounded statevector calculations.

    Parameters
    ----------
    circuit : QuantumCircuit
        The unitary circuit to check. No ancilla postselection normalization
        or column phase corrections are applied.
    system : sequence of Qubit
        System bits in little-endian logical order. Their physical positions
        come from circuit.find_bit, so interleaved registers also work.
    max_qubits : int
        Upper limit checked before allocation or transpilation (default 16).
        The output block is additionally limited to 64 MiB.

    Returns
    -------
    numpy.ndarray
        The dense block <0_ancilla,y|U|0_ancilla,x>. This is a small-system
        verifier; statevectors and the number of columns still grow exponentially.
    """
    if isinstance(max_qubits, (bool, np.bool_)) or not isinstance(max_qubits, (int, np.integer)) or max_qubits < 1:
        raise ValueError("max_qubits must be a positive integer")
    if circuit.num_qubits > max_qubits:
        raise ValueError(f"circuit has {circuit.num_qubits} qubits; limit is {max_qubits}")
    _check_registers(circuit, (system, len(system)))
    if not system:
        raise ValueError("at least one system qubit is required")
    dimension = 1 << len(system)
    if dimension * dimension * np.dtype(np.complex128).itemsize > 64 * 1024 * 1024:
        raise ValueError("encoded block would exceed the 64 MiB verification limit")
    positions = [circuit.find_bit(qubit).index for qubit in system]
    indices = np.array([sum(((x >> bit) & 1) << pos for bit, pos in enumerate(positions))
                        for x in range(dimension)], dtype=np.intp)
    # Lower once, avoiding both repeated synthesis per column and dense matrices
    # for large annotated controlled gates inside Statevector.evolve.
    lowered = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=7)
    encoded = np.empty((dimension, dimension), dtype=np.complex128)
    for x, index in enumerate(indices):
        initial = Statevector.from_int(int(index), dims=1 << circuit.num_qubits)
        encoded[:, x] = initial.evolve(lowered).data[indices]
    return encoded
