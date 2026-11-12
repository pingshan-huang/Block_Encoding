"""Build the grouped PMR Fig. 5 block-encoding circuit with Qiskit.

This is a separate Qiskit implementation of the circuit used by ``main.py``:

    PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger

The existing NumPy sparse simulator remains unchanged.  This file reuses its
validated PMR input model and fixed-point angle/phase codec, but constructs an
actual ``qiskit.QuantumCircuit``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Gate, Qubit
    from qiskit.circuit.library import PhaseGate, RYGate, StatePreparation, XGate
    from qiskit.quantum_info import Statevector
except ModuleNotFoundError as exc:  # Give a useful message when Qiskit is absent.
    raise ModuleNotFoundError(
        "Qiskit is required for qiskit_fig5.py. Install code/requirements-qiskit.txt."
    ) from exc

from main import example_model
from pmr_model import PMRModel
from r_block import RBlock
from rotation_codec import AnglePhaseCodec
from value import ValueOracle


@dataclass(frozen=True)
class QiskitFig5Build:
    """Return the circuit together with its named quantum registers."""

    circuit: QuantumCircuit
    model: PMRModel
    codec: AnglePhaseCodec
    r_table: RBlock
    flag: QuantumRegister
    workspace: QuantumRegister
    move: QuantumRegister
    system: QuantumRegister


def _bits(value: int, width: int) -> list[int]:
    """Return little-endian bits matching Qiskit's qubit-index convention."""
    return [(value >> bit) & 1 for bit in range(width)]


def _open_pattern(
    circuit: QuantumCircuit,
    controls: list[Qubit],
    pattern: int,
) -> list[Qubit]:
    """Turn an arbitrary control pattern into an all-ones control condition.

    Qubits whose requested pattern bit is zero receive X before the controlled
    operation.  The returned list tells ``_close_pattern`` which X gates to undo.
    """
    opened = [qubit for index, qubit in enumerate(controls) if not (pattern >> index) & 1]
    for qubit in opened:
        circuit.x(qubit)
    return opened


def _close_pattern(circuit: QuantumCircuit, opened: list[Qubit]) -> None:
    """Undo the temporary X gates inserted by ``_open_pattern``."""
    for qubit in reversed(opened):
        circuit.x(qubit)


def _append_controlled_gate(
    circuit: QuantumCircuit,
    gate: Gate,
    controls: list[Qubit],
    targets: list[Qubit],
) -> None:
    """Append ``gate`` controlled by every qubit in ``controls``."""
    if controls:
        # annotated=True keeps large controlled rotations symbolic until transpilation.
        controlled_gate = gate.control(len(controls), annotated=True)
        circuit.append(controlled_gate, [*controls, *targets])
    else:
        circuit.append(gate, targets)


def append_prep(
    circuit: QuantumCircuit,
    move: QuantumRegister,
    model: PMRModel,
    *,
    inverse: bool = False,
) -> None:
    """Append PREP or PREP dagger to the move register.

    PREP maps |0> to sum_mu sqrt(b_mu / alpha) |mu>.
    """
    amplitudes = np.zeros(model.move_dim, dtype=np.complex128)
    amplitudes[: model.move_count] = np.sqrt(model.bounds / model.alpha)
    gate = StatePreparation(amplitudes, label="PREP")
    if inverse:
        gate = gate.inverse()
        gate.label = "PREP_dg"
    circuit.append(gate, list(move))


def append_r(
    circuit: QuantumCircuit,
    workspace: QuantumRegister,
    move: QuantumRegister,
    system: QuantumRegister,
    r_table: RBlock,
) -> None:
    """Append lowercase r, which XOR-loads angle/phase data into workspace.

    For each |mu>|x> branch, this performs

        |0>_b -> |encoded(theta_mu(x), phase_mu(x))>_b.

    The same function is used for r dagger because XOR loading is self-inverse.
    """
    controls = [*move, *system]
    move_width = len(move)

    for move_index in range(r_table.model.move_count):
        for x in range(r_table.model.system_dim):
            word = r_table.lookup_word(move_index, x)
            if word == 0:
                continue

            pattern = move_index | (x << move_width)
            opened = _open_pattern(circuit, controls, pattern)
            for bit, qubit in enumerate(workspace):
                if (word >> bit) & 1:
                    _append_controlled_gate(circuit, XGate(), controls, [qubit])
            _close_pattern(circuit, opened)


def append_flag_rotation(
    circuit: QuantumCircuit,
    flag: QuantumRegister,
    workspace: QuantumRegister,
    r_table: RBlock,
) -> None:
    """Append uppercase R controlled by the encoded workspace word.

    On an input flag |0>, the success amplitude becomes

        cos(theta/2) exp(i phase) = c_tilde_mu(x) / b_mu.

    Workspace words that are not produced by lowercase r act as identity.
    """
    controls = list(workspace)
    active_words = {
        r_table.lookup_word(move_index, x)
        for move_index in range(r_table.model.move_count)
        for x in range(r_table.model.system_dim)
    }

    for word in sorted(active_words):
        data = r_table.codec.decode_word(word)
        if np.isclose(data.theta, 0.0) and np.isclose(data.phase, 0.0):
            continue

        opened = _open_pattern(circuit, controls, word)

        if not np.isclose(data.theta, 0.0):
            _append_controlled_gate(circuit, RYGate(data.theta), controls, [flag[0]])

        if not np.isclose(data.phase, 0.0):
            # X-Phase-X applies exp(i*phase) to flag |0>, not flag |1>.
            circuit.x(flag[0])
            _append_controlled_gate(circuit, PhaseGate(data.phase), controls, [flag[0]])
            circuit.x(flag[0])

        _close_pattern(circuit, opened)


def append_move(
    circuit: QuantumCircuit,
    move: QuantumRegister,
    system: QuantumRegister,
    model: PMRModel,
) -> None:
    """Append MOVE: |mu>|x> -> |mu>|x XOR masks[mu]>."""
    controls = list(move)

    for move_index, mask in enumerate(model.masks):
        if mask == 0:
            continue

        opened = _open_pattern(circuit, controls, move_index)
        for bit, target in enumerate(system):
            if (mask >> bit) & 1:
                _append_controlled_gate(circuit, XGate(), controls, [target])
        _close_pattern(circuit, opened)


def build_qiskit_fig5_circuit(
    model: PMRModel,
    *,
    theta_bits: int = 12,
    phase_bits: int = 16,
    barriers: bool = True,
) -> QiskitFig5Build:
    """Build and return the complete Qiskit Fig. 5 circuit.

    The input is a validated ``PMRModel``.  The output contains the Qiskit
    circuit, all registers, and the fixed-point lookup table used by lowercase r.
    """
    codec = AnglePhaseCodec(theta_bits=theta_bits, phase_bits=phase_bits)
    value = ValueOracle(model)
    r_table = RBlock(value, codec)

    flag = QuantumRegister(1, "flag")
    workspace = QuantumRegister(codec.workspace_bits, "b")
    move = QuantumRegister(model.move_qubits, "move")
    system = QuantumRegister(model.n_qubits, "system")
    circuit = QuantumCircuit(flag, workspace, move, system, name="PMR_Fig5")

    append_prep(circuit, move, model)
    if barriers:
        circuit.barrier(label="after PREP")

    append_r(circuit, workspace, move, system, r_table)
    if barriers:
        circuit.barrier(label="after r")

    append_flag_rotation(circuit, flag, workspace, r_table)
    if barriers:
        circuit.barrier(label="after R")

    append_r(circuit, workspace, move, system, r_table)
    if barriers:
        circuit.barrier(label="after r_dagger")

    append_move(circuit, move, system, model)
    if barriers:
        circuit.barrier(label="after MOVE")

    append_prep(circuit, move, model, inverse=True)

    return QiskitFig5Build(
        circuit=circuit,
        model=model,
        codec=codec,
        r_table=r_table,
        flag=flag,
        workspace=workspace,
        move=move,
        system=system,
    )


def encoded_block_from_statevector(
    build: QiskitFig5Build,
    *,
    max_qubits: int = 16,
) -> np.ndarray:
    """Extract the all-zero-ancilla block using Qiskit's Statevector.

    Statevector memory grows as 2**num_qubits.  The default 12+16 bit workspace
    is intentionally rejected here; use smaller precision for local verification.
    """
    circuit = build.circuit
    if circuit.num_qubits > max_qubits:
        raise ValueError(
            f"circuit has {circuit.num_qubits} qubits; statevector verification "
            f"is limited to {max_qubits}. Use fewer theta/phase bits."
        )

    model = build.model
    encoded = np.zeros((model.system_dim, model.system_dim), dtype=np.complex128)
    system_offset = len(build.flag) + len(build.workspace) + len(build.move)

    for x in range(model.system_dim):
        column_circuit = QuantumCircuit(circuit.num_qubits)
        for bit, value in enumerate(_bits(x, model.n_qubits)):
            if value:
                column_circuit.x(system_offset + bit)
        column_circuit.compose(circuit, inplace=True)
        final_state = Statevector.from_instruction(column_circuit)

        for y in range(model.system_dim):
            # Ancillas are zero, so only the system bits contribute to the index.
            encoded[y, x] = final_state.data[y << system_offset]

    return encoded


def main() -> None:
    """Build the example circuit, optionally draw it or verify a small version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-bits", type=int, default=12)
    parser.add_argument("--phase-bits", type=int, default=16)
    parser.add_argument("--draw", action="store_true", help="print a text circuit")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="extract the encoded block (use a small workspace, such as 3+4 bits)",
    )
    args = parser.parse_args()

    build = build_qiskit_fig5_circuit(
        example_model(),
        theta_bits=args.theta_bits,
        phase_bits=args.phase_bits,
    )

    print("Qiskit grouped PMR Fig. 5 circuit")
    print(f"qubits: {build.circuit.num_qubits}")
    print(f"workspace bits: {build.codec.workspace_bits}")
    print(f"top-level operations: {build.circuit.size()}")

    if args.draw:
        print(build.circuit.draw(output="text", fold=120))

    if args.simulate:
        encoded = encoded_block_from_statevector(build)
        target = build.model.target_block()
        error = np.linalg.norm(encoded - target, ord=2)
        np.set_printoptions(precision=8, suppress=True)
        print("\nEncoded all-zero-ancilla block:")
        print(encoded)
        print("\nTarget (H-cI)/alpha_PMR:")
        print(target)
        print(f"\nfixed-point block error: {error:.3e}")


if __name__ == "__main__":
    main()
