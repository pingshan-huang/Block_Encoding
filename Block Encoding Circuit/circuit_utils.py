"""Original single-pattern helpers from Antonio's modular circuit (9b091ff).

These remain available for callers that build one controlled block at a time.
The integrated oracles use the shared pattern traversal in ``qiskit_oracles``
to avoid reopening and closing all control bits between adjacent blocks.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Qubit


def open_control_pattern(
    circuit: QuantumCircuit,
    controls: list[Qubit],
    pattern: int,
) -> list[Qubit]:
    """Convert an arbitrary little-endian pattern into all-one controls."""
    opened = [
        qubit
        for index, qubit in enumerate(controls)
        if not (pattern >> index) & 1
    ]
    for qubit in opened:
        circuit.x(qubit)
    return opened


def close_control_pattern(circuit: QuantumCircuit, opened: list[Qubit]) -> None:
    """Undo X gates introduced by ``open_control_pattern``."""
    for qubit in reversed(opened):
        circuit.x(qubit)


def append_controlled_gate(
    circuit: QuantumCircuit,
    gate: Gate,
    controls: list[Qubit],
    targets: list[Qubit],
) -> None:
    """Append a gate controlled by every qubit in ``controls``."""
    if controls:
        controlled_gate = gate.control(len(controls), annotated=True)
        circuit.append(controlled_gate, [*controls, *targets])
    else:
        circuit.append(gate, targets)
