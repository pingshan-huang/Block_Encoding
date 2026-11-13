"""Wire Antonio's modular Qiskit interface through shared PMR circuit helpers.

The register layout and oracle interface originate in Antonio's commit 9b091ff.
Local integration reuses the validated mathematical core and the same circuit
primitives as ``code/qiskit_fig5.py`` so their numerical rules cannot drift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from angle_phase_codec import AnglePhaseCodec
from flag_rotation_oracle import FlagRotationOracle
from model import PMRModel
from move_oracle import MoveOracle
from prep_oracle import PrepOracle
from r_oracle import LowercaseROracle
from value_oracle import ValueOracle
from qiskit_oracles import extract_encoded_block


@dataclass(frozen=True)
class CircuitRegisters:
    """Keep named access to every register in the assembled circuit."""

    flag: QuantumRegister
    workspace: QuantumRegister
    move: QuantumRegister
    system: QuantumRegister


@dataclass(frozen=True)
class CircuitOracles:
    """Keep the independently constructed oracle objects."""

    prep: PrepOracle
    value: ValueOracle
    r: LowercaseROracle
    rotation: FlagRotationOracle
    move: MoveOracle


@dataclass(frozen=True)
class CircuitBuild:
    """Return everything needed to inspect or execute the circuit."""

    circuit: QuantumCircuit
    model: PMRModel
    codec: AnglePhaseCodec
    registers: CircuitRegisters
    oracles: CircuitOracles


def example_model() -> PMRModel:
    """Return the one-qubit complex Hamiltonian used by the project."""
    coefficients = np.asarray(
        [
            [1.5, -0.5],
            [0.3 + 0.4j, 0.3 - 0.4j],
        ],
        dtype=np.complex128,
    )
    return PMRModel.from_coefficients(
        n_qubits=1,
        masks=(0b0, 0b1),
        coefficients=coefficients,
    )


def create_registers(model: PMRModel, codec: AnglePhaseCodec) -> CircuitRegisters:
    """Create flag, workspace, move, and system registers in that qubit order.

    Qiskit assigns low integer-index bits to the first qubits, so the system
    register occupies the highest positions in this standard layout. Each
    register's element zero is its least-significant bit.
    """
    return CircuitRegisters(
        flag=QuantumRegister(1, "flag"),
        workspace=QuantumRegister(codec.workspace_bits, "b"),
        move=QuantumRegister(model.move_qubits, "move"),
        system=QuantumRegister(model.n_qubits, "system"),
    )


def create_oracles(model: PMRModel, codec: AnglePhaseCodec) -> CircuitOracles:
    """Share one validated model and encoded lookup among all six circuit steps.

    VALUE is classical data used to build lowercase r; it is not another gate
    in the quantum sequence. The rotation oracle reads that same quantized
    lookup, and r dagger reverses the same XOR operation used by r.
    """
    prep = PrepOracle(model)
    value = ValueOracle(model)
    r = LowercaseROracle(value, codec)
    rotation = FlagRotationOracle(r)
    move = MoveOracle(model)
    return CircuitOracles(
        prep=prep,
        value=value,
        r=r,
        rotation=rotation,
        move=move,
    )


def build_fig5_circuit(
    model: PMRModel,
    *,
    theta_bits: int = 12,
    phase_bits: int = 16,
    barriers: bool = True,
) -> CircuitBuild:
    """Build PREP -> r -> R -> r dagger -> MOVE -> PREP dagger.

    Parameters
    ----------
    model : PMRModel
        Shared, validated source-indexed coefficients and centered row bounds.
    theta_bits, phase_bits : int
        Fixed-point angle widths accepted by the shared circuit codec. These
        bits are real workspace qubits, not just classical precision settings.
    barriers : bool
        Separate the six steps visually. Disable for resource comparisons.

    Returns
    -------
    CircuitBuild
        The unitary circuit, registers, and independently inspectable oracles.
        Its all-zero-ancilla block approximates ``(H - shift*I) / alpha`` through
        angle quantization. Construction does not allocate a statevector; full
        simulation of the default workspace is intentionally not supported by
        the bounded extraction helper.
    """
    codec = AnglePhaseCodec(theta_bits=theta_bits, phase_bits=phase_bits)
    registers = create_registers(model, codec)
    oracles = create_oracles(model, codec)
    circuit = QuantumCircuit(
        registers.flag,
        registers.workspace,
        registers.move,
        registers.system,
        name="Grouped_PMR_Block_Encoding",
    )

    oracles.prep.append(circuit, registers.move)
    if barriers:
        circuit.barrier(label="PREP")

    oracles.r.append(
        circuit,
        registers.workspace,
        registers.move,
        registers.system,
    )
    if barriers:
        circuit.barrier(label="r")

    oracles.rotation.append(circuit, registers.flag, registers.workspace)
    if barriers:
        circuit.barrier(label="R")

    oracles.r.append_inverse(
        circuit,
        registers.workspace,
        registers.move,
        registers.system,
    )
    if barriers:
        circuit.barrier(label="r_dagger")

    oracles.move.append(circuit, registers.move, registers.system)
    if barriers:
        circuit.barrier(label="MOVE")

    oracles.prep.append(circuit, registers.move, inverse=True)

    return CircuitBuild(
        circuit=circuit,
        model=model,
        codec=codec,
        registers=registers,
        oracles=oracles,
    )


def _little_endian_bits(value: int, width: int) -> list[int]:
    """Return integer bits in Qiskit's qubit-index order."""
    return [(value >> bit) & 1 for bit in range(width)]


def encoded_block_from_statevector(
    build: CircuitBuild,
    *,
    max_qubits: int = 16,
) -> np.ndarray:
    """Return ``<0_ancilla,y|U|0_ancilla,x>`` for a bounded small circuit.

    Parameters
    ----------
    build : CircuitBuild
        Circuit and the system qubits whose order defines matrix row/columns.
    max_qubits : int
        Maximum full statevector width, checked before simulation. Use small
        theta/phase widths rather than increasing this limit for default builds.

    Returns
    -------
    numpy.ndarray
        Unrenormalized complex block, using the actual Qiskit qubit positions.
        No column-phase alignment is performed. All non-system qubits start
        and end at zero, including any ancillary registers in the circuit.
    """
    return extract_encoded_block(
        build.circuit, build.registers.system, max_qubits=max_qubits
    )


def main() -> None:
    """Build, display, or simulate the modular circuit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-bits", type=int, default=12)
    parser.add_argument("--phase-bits", type=int, default=16)
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    build = build_fig5_circuit(
        example_model(),
        theta_bits=args.theta_bits,
        phase_bits=args.phase_bits,
    )

    print("Modular Qiskit grouped PMR block encoding")
    print("flow: PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger")
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
