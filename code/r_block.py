"""Framework-independent reversible implementation of lowercase r."""

from __future__ import annotations  # Enable modern type annotations.

import numpy as np  # Store packed lookup words.

from rotation_codec import AnglePhaseCodec, RotationData  # Import b encoding.
from state import SparseState, add_amplitude  # Import sparse-state helpers.
from value import ValueOracle  # Import normalized VALUE.


class RBlock:  # Implement reversible fixed-point angle and phase loading.
    def __init__(self, value_oracle: ValueOracle, codec: AnglePhaseCodec) -> None:  # Initialize r.
        self.value_oracle = value_oracle  # Save VALUE.
        self.model = value_oracle.model  # Save the PMR model.
        self.codec = codec  # Save fixed-point precision.
        self.words = np.zeros(  # Allocate one word per (move,x) branch.
            (self.model.move_dim, self.model.system_dim),  # Set table shape.
            dtype=np.uint64,  # Store packed unsigned integers.
        )  # Finish allocation.
        for move in range(self.model.move_count):  # Visit valid moves.
            for x in range(self.model.system_dim):  # Visit system states.
                z = value_oracle.normalized_value(move, x)  # Read z_mu(x).
                self.words[move, x] = codec.encode_value(z)  # Store theta and phase.

    @property  # Expose b width.
    def workspace_bits(self) -> int:  # Return b bits.
        return self.codec.workspace_bits  # Delegate to the codec.

    def lookup_word(self, move: int, x: int) -> int:  # Read one branch word.
        if not 0 <= move < self.model.move_dim:  # Validate move.
            raise IndexError("move is out of range")  # Reject invalid move.
        if not 0 <= x < self.model.system_dim:  # Validate x.
            raise IndexError("system state is out of range")  # Reject invalid x.
        return int(self.words[move, x])  # Return the packed word.

    def decoded_rotation(self, move: int, x: int) -> RotationData:  # Inspect data.
        return self.codec.decode_word(self.lookup_word(move, x))  # Decode the word.

    def apply_basis(  # Apply r to one basis state.
        self, flag: int, workspace: int, move: int, x: int  # Receive f,b,m,x.
    ) -> tuple[int, int, int, int]:  # Return f,b,m,x.
        return flag, workspace ^ self.lookup_word(move, x), move, x  # XOR-load b.

    def apply(self, state: SparseState) -> SparseState:  # Apply r to a sparse state.
        result: SparseState = {}  # Allocate output.
        for basis, amplitude in state.items():  # Visit active basis states.
            add_amplitude(result, self.apply_basis(*basis), amplitude)  # Permute.
        return result  # Return transformed state.

    def apply_inverse(self, state: SparseState) -> SparseState:  # Apply r dagger.
        return self.apply(state)  # XOR loading is self-inverse.
