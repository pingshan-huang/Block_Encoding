"""Build the grouped PMR Fig. 5 block-encoding circuit with Qiskit.

Originally added on Antonio at commit 9b091ff; this reviewed entry point uses
shared primitives also used by the modular oracle interface. Its sequence is:

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
except ModuleNotFoundError as exc:  # Give a useful message when Qiskit is absent.
    raise ModuleNotFoundError(
        "Qiskit is required for qiskit_fig5.py. Install code/requirements-qiskit.txt."
    ) from exc

from main import example_model
from pmr_model import PMRModel
from qiskit_oracles import (
    QiskitAnglePhaseCodec as AnglePhaseCodec,
    QiskitRTable as RBlock,
    append_prep, append_r, append_flag_rotation, append_move,
    extract_encoded_block,
)
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


def build_qiskit_fig5_circuit(
    model: PMRModel,
    *,
    theta_bits: int = 12,
    phase_bits: int = 16,
    barriers: bool = True,
) -> QiskitFig5Build:
    """Build PREP -> r -> R -> r dagger -> MOVE -> PREP dagger.

    Parameters
    ----------
    model : PMRModel
        A model from the shared validated mathematical core. Bounds and shift
        determine the target (H - shift*I)/alpha, including zero-weight rows.
    theta_bits, phase_bits : int
        Fixed-point field widths, each 1--52 with a total no greater than 64.
        Circuit construction does not allocate a full unitary or statevector.
    barriers : bool
        Add visual boundaries between the six logical stages.

    Returns
    -------
    QiskitFig5Build
        The circuit, named registers, codec, and classical lookup data. Qubit
        order is flag, workspace, move, system (least-significant first).
        R encodes quantized VALUE; a general finite-precision block need not
        be exactly Hermitian even when the input Hamiltonian is Hermitian.
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
    """Return the zero-ancilla block with bounded, little-endian verification.

    The default 31-qubit circuit is intentionally refused by the default
    16-qubit limit. Use small angle/phase fields for numerical checks. Actual
    system-qubit positions are read from the circuit, not inferred from offsets.
    No postselection renormalization or column-phase adjustment is applied.
    """
    return extract_encoded_block(build.circuit, build.system, max_qubits=max_qubits)


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
