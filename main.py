"""Assemble and verify the Qrisp grouped block encoding from Fig. 5."""

from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np

try:
    from qrisp import QuantumVariable, conjugate, x as qx
except ModuleNotFoundError as exc:
    if exc.name != "qrisp":
        raise
    raise SystemExit(
        "Qrisp is not installed in the Python interpreter running this file.\n"
        "From the project directory, run:\n"
        r"  .\.venv\Scripts\python.exe main.py"
        "\nOr install the dependencies with:\n"
        r"  python -m pip install -r requirements.txt"
    ) from None

from flag_rotation import FlagRotationBlock
from move import MoveBlock
from pmr_model import PMRModel
from prep import PrepBlock
from r_compute import RComputeBlock
from value import ValueOracle


@dataclass(frozen=True)
class CircuitBuild:
    model: PMRModel
    session: object
    encoded_block: np.ndarray
    flag: QuantumVariable
    b: QuantumVariable
    move_register: QuantumVariable
    system: QuantumVariable
    prep: PrepBlock
    value: ValueOracle
    r_compute: RComputeBlock
    flag_rotation: FlagRotationBlock
    move: MoveBlock


def _assemble_qrisp_session(
    model: PMRModel, *, system_input: int | None = None
) -> tuple[
    object,
    QuantumVariable,
    QuantumVariable,
    QuantumVariable,
    QuantumVariable,
    PrepBlock,
    ValueOracle,
    RComputeBlock,
    FlagRotationBlock,
    MoveBlock,
]:
    """Create one Qrisp session, optionally initialized to a basis state.

    Builds PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger.
    Qrisp variables are allocated in register order ``(f, b, m, s)``.
    """
    prep = PrepBlock(model)
    value = ValueOracle(model)
    r_compute = RComputeBlock(value)
    flag_rotation = FlagRotationBlock(r_compute)
    move = MoveBlock(model)

    flag = QuantumVariable(1, name="f")
    b = QuantumVariable(r_compute.b_qubits, qs=flag.qs, name="b")
    move_register = QuantumVariable(
        model.move_qubits, qs=flag.qs, name="m"
    )
    system = QuantumVariable(model.n_qubits, qs=flag.qs, name="s")

    if system_input is not None:
        if not 0 <= system_input < model.system_dim:
            raise ValueError("system_input is outside the system register")
        for bit in range(model.n_qubits):
            if (system_input >> bit) & 1:
                qx(system[bit])

    # Qrisp's conjugation environment emits PREP, the body, and PREP dagger.
    with conjugate(prep.apply)(move_register):
        r_compute.apply(b, move_register, system)
        flag_rotation.apply(flag, b)
        r_compute.apply_inverse(b, move_register, system)
        move.apply(move_register, system)

    return (
        flag.qs,
        flag,
        b,
        move_register,
        system,
        prep,
        value,
        r_compute,
        flag_rotation,
        move,
    )


def build_fig5_circuit(model: PMRModel) -> CircuitBuild:
    """Build the Qrisp circuit and simulate its encoded system block."""
    (
        session,
        flag,
        b,
        move_register,
        system,
        prep,
        value,
        r_compute,
        flag_rotation,
        move,
    ) = _assemble_qrisp_session(model)

    encoded = simulate_zero_ancilla_block(model)
    return CircuitBuild(
        model=model,
        session=session,
        encoded_block=encoded,
        flag=flag,
        b=b,
        move_register=move_register,
        system=system,
        prep=prep,
        value=value,
        r_compute=r_compute,
        flag_rotation=flag_rotation,
        move=move,
    )


def _basis_index(session: object, system: QuantumVariable, value: int) -> int:
    """Return Qrisp's statevector-array index with all ancillas zero.

    Qrisp lists qubits in allocation order but ``statevector_array`` places
    the first listed qubit on the most-significant array axis.
    """
    basis_index = 0
    qubit_count = len(session.qubits)
    for bit, qubit in enumerate(system.reg):
        if (value >> bit) & 1:
            position = session.qubits.index(qubit)
            basis_index |= 1 << (qubit_count - 1 - position)
    return basis_index


def simulate_zero_ancilla_block(model: PMRModel) -> np.ndarray:
    """Simulate only the columns needed for the all-zero ancilla block."""
    encoded = np.zeros((model.system_dim, model.system_dim), dtype=np.complex128)
    for x_value in range(model.system_dim):
        session, _, _, _, system, *_ = _assemble_qrisp_session(
            model, system_input=x_value
        )
        statevector = np.asarray(session.statevector_array(), dtype=np.complex128)
        for y_value in range(model.system_dim):
            encoded[y_value, x_value] = statevector[
                _basis_index(session, system, y_value)
            ]
    return encoded


def example_model() -> PMRModel:
    """One-qubit example containing diagonal and complex off-diagonal moves."""
    # c_1(1) = conj(c_1(0)) enforces Hermiticity for the XOR-1 move.
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


def main() -> None:
    # Qrisp's text circuit renderer uses Unicode box-drawing characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model = example_model()
    build = build_fig5_circuit(model)
    expected = model.target_block()
    error = np.linalg.norm(build.encoded_block - expected, ord=2)

    np.set_printoptions(precision=6, suppress=True)
    print("Qrisp grouped PMR Fig. 5 circuit")
    print(f"move masks: {model.masks}")
    print(f"bounds: {model.bounds}")
    print(f"shift c: {model.shift:.6g}")
    print(f"alpha_PMR: {model.alpha:.6g}")
    print(
        "register qubits (f,b,m,s): "
        f"(1,{build.r_compute.b_qubits},{model.move_qubits},{model.n_qubits})"
    )
    print(f"Qrisp instructions: {len(build.session.data)}")
    print(
        "Live registers: "
        + ", ".join(qv.name for qv in build.session.qv_list)
    )
    print("\nEncoded all-zero ancilla block:")
    print(build.encoded_block)
    print("\nExpected (H-cI)/alpha_PMR:")
    print(expected)
    print(f"\nblock spectral-norm error: {error:.3e}")

    if error > 2e-6:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
