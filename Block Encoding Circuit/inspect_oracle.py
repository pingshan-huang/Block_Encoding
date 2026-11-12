"""Draw and inspect one oracle from the modular Qiskit implementation."""

from __future__ import annotations

import argparse

from qiskit import QuantumCircuit, QuantumRegister

from angle_phase_codec import AnglePhaseCodec
from flag_rotation_oracle import FlagRotationOracle
from main import example_model
from move_oracle import MoveOracle
from prep_oracle import PrepOracle
from r_oracle import LowercaseROracle
from value_oracle import ValueOracle


ORACLE_NAMES = (
    "value",
    "prep",
    "prep-dagger",
    "r",
    "r-dagger",
    "R",
    "move",
    "all",
)


def build_oracle_circuit(
    name: str,
    *,
    theta_bits: int,
    phase_bits: int,
) -> QuantumCircuit | None:
    """Build a standalone circuit containing only the requested oracle."""
    model = example_model()
    codec = AnglePhaseCodec(theta_bits=theta_bits, phase_bits=phase_bits)
    value = ValueOracle(model)
    r_oracle = LowercaseROracle(value, codec)

    if name == "value":
        return None

    if name in {"prep", "prep-dagger"}:
        move = QuantumRegister(model.move_qubits, "move")
        circuit = QuantumCircuit(move, name=name)
        PrepOracle(model).append(circuit, move, inverse=name == "prep-dagger")
        return circuit

    if name in {"r", "r-dagger"}:
        workspace = QuantumRegister(codec.workspace_bits, "b")
        move = QuantumRegister(model.move_qubits, "move")
        system = QuantumRegister(model.n_qubits, "system")
        circuit = QuantumCircuit(workspace, move, system, name=name)
        if name == "r":
            r_oracle.append(circuit, workspace, move, system)
        else:
            r_oracle.append_inverse(circuit, workspace, move, system)
        return circuit

    if name == "R":
        flag = QuantumRegister(1, "flag")
        workspace = QuantumRegister(codec.workspace_bits, "b")
        circuit = QuantumCircuit(flag, workspace, name="R")
        FlagRotationOracle(r_oracle).append(circuit, flag, workspace)
        return circuit

    if name == "move":
        move = QuantumRegister(model.move_qubits, "move")
        system = QuantumRegister(model.n_qubits, "system")
        circuit = QuantumCircuit(move, system, name="MOVE")
        MoveOracle(model).append(circuit, move, system)
        return circuit

    raise ValueError(f"unknown oracle: {name}")


def print_value_oracle() -> None:
    """Print VALUE's normalized lookup data; VALUE is used to construct r."""
    model = example_model()
    value = ValueOracle(model)
    print("VALUE oracle lookup table z_mu(x) = c_tilde_mu(x) / b_mu")
    print("Rows are move indices; columns are system states x.")
    print(value.table())
    print("\nVALUE is classical construction data in this design.")
    print("Its reversible quantum action is implemented by lowercase r.")


def print_oracle(
    name: str,
    *,
    theta_bits: int,
    phase_bits: int,
    decompose: int,
    fold: int,
) -> None:
    """Print one oracle's register count, operation count, and text diagram."""
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    if name == "value":
        print_value_oracle()
        return

    circuit = build_oracle_circuit(
        name,
        theta_bits=theta_bits,
        phase_bits=phase_bits,
    )
    if circuit is None:
        return

    displayed = circuit.decompose(reps=decompose) if decompose else circuit
    print(f"qubits: {circuit.num_qubits}")
    print(f"top-level operations: {circuit.size()}")
    if decompose:
        print(f"display decomposition levels: {decompose}")
    print(displayed.draw(output="text", fold=fold))


def main() -> None:
    """Parse the requested oracle and display its standalone Qiskit circuit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", choices=ORACLE_NAMES)
    parser.add_argument("--theta-bits", type=int, default=3)
    parser.add_argument("--phase-bits", type=int, default=4)
    parser.add_argument(
        "--decompose",
        type=int,
        default=0,
        metavar="LEVELS",
        help="decompose the circuit this many levels before drawing",
    )
    parser.add_argument("--fold", type=int, default=160)
    args = parser.parse_args()

    if args.decompose < 0:
        parser.error("--decompose must be zero or positive")

    names = ORACLE_NAMES[:-1] if args.oracle == "all" else (args.oracle,)
    for name in names:
        print_oracle(
            name,
            theta_bits=args.theta_bits,
            phase_bits=args.phase_bits,
            decompose=args.decompose,
            fold=args.fold,
        )


if __name__ == "__main__":
    main()
