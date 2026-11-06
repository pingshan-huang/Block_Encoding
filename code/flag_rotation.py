"""Framework-independent uppercase R acting on the success flag."""

from __future__ import annotations

import numpy as np

from r_block import RBlock
from rotation_codec import RotationData
from state import SparseState, add_amplitude


class FlagRotationBlock:
    """Multiplexed Ry plus phase operation controlled by register b."""

    def __init__(self, r_block: RBlock) -> None:
        self.r_block = r_block
        self.model = r_block.model

    @staticmethod
    def single_qubit_unitary(data: RotationData) -> np.ndarray:
        """Return Phase_0(phi) @ Ry(theta).

        The upper-left entry is exactly the normalized complex coefficient.
        """
        rho = data.magnitude
        sine = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        phase = np.exp(1j * data.phase)
        return np.asarray(
            [[phase * rho, -phase * sine], [sine, rho]],
            dtype=np.complex128,
        )

    def apply(self, state: SparseState) -> SparseState:
        """Apply the b-controlled flag rotation to a sparse state."""
        result: SparseState = {}
        for (flag, workspace, move, x), amplitude in state.items():
            data = self.r_block.codec.decode_word(workspace)
            gate = self.single_qubit_unitary(data)
            for output_flag in range(2):
                factor = gate[output_flag, flag]
                if abs(factor) > 1e-15:
                    add_amplitude(
                        result,
                        (output_flag, workspace, move, x),
                        factor * amplitude,
                    )
        return result
