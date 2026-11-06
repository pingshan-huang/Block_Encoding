"""Assemble and verify the framework-independent grouped block encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flag_rotation import FlagRotationBlock
from move import MoveBlock
from pmr_model import PMRModel
from prep import PrepBlock
from r_block import RBlock
from rotation_codec import AnglePhaseCodec
from state import SparseState
from value import ValueOracle


@dataclass(frozen=True)
class CircuitBuild:
    """Collect the assembled primitives and simulated encoded block."""

    model: PMRModel
    encoded_block: np.ndarray
    prep: PrepBlock
    value: ValueOracle
    r: RBlock
    flag_rotation: FlagRotationBlock
    move: MoveBlock


def run_column(
    system_input: int,
    prep: PrepBlock,
    r: RBlock,
    flag_rotation: FlagRotationBlock,
    move: MoveBlock,
) -> SparseState:
    """Run PREP -> r -> R -> r_dagger -> MOVE -> PREP_dagger."""
    state: SparseState = {(0, 0, 0, system_input): 1.0 + 0.0j}
    state = prep.apply(state)
    state = r.apply(state)
    state = flag_rotation.apply(state)
    state = r.apply_inverse(state)
    state = move.apply(state)
    state = prep.apply_inverse(state)
    return state


def build_fig5_circuit(
    model: PMRModel,
    *,
    theta_bits: int = 12,
    phase_bits: int = 16,
) -> CircuitBuild:
    """Build all primitives and simulate the all-zero-ancilla block."""
    prep = PrepBlock(model)
    value = ValueOracle(model)
    codec = AnglePhaseCodec(theta_bits=theta_bits, phase_bits=phase_bits)
    r = RBlock(value, codec)
    flag_rotation = FlagRotationBlock(r)
    move = MoveBlock(model)
    encoded = np.zeros((model.system_dim, model.system_dim), dtype=np.complex128)

    for x in range(model.system_dim):
        final_state = run_column(x, prep, r, flag_rotation, move)
        for y in range(model.system_dim):
            encoded[y, x] = final_state.get((0, 0, 0, y), 0.0j)

    return CircuitBuild(
        model=model,
        encoded_block=encoded,
        prep=prep,
        value=value,
        r=r,
        flag_rotation=flag_rotation,
        move=move,
    )


def example_model() -> PMRModel:
    """Return the one-qubit complex example used throughout the project."""
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
    model = example_model()
    build = build_fig5_circuit(model)
    expected = model.target_block()
    error = np.linalg.norm(build.encoded_block - expected, ord=2)

    np.set_printoptions(precision=8, suppress=True)
    print("Framework-independent grouped PMR block encoding")
    print(f"move masks: {model.masks}")
    print(f"bounds: {model.bounds}")
    print(f"shift c: {model.shift:.8g}")
    print(f"alpha_PMR: {model.alpha:.8g}")
    print(f"b-register bits: {build.r.workspace_bits}")
    print("\nEncoded all-zero-ancilla block:")
    print(build.encoded_block)
    print("\nExpected (H-cI)/alpha_PMR:")
    print(expected)
    print(f"\nfixed-point block error: {error:.3e}")

    if error > 5e-5:
        raise SystemExit("verification failed: increase theta_bits or phase_bits")


if __name__ == "__main__":
    main()
