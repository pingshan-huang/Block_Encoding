"""Fixed-point angle and phase encoding for lowercase r."""

from __future__ import annotations  # Enable modern type annotations.

from dataclasses import dataclass  # Provide immutable data containers.

import numpy as np  # Provide trigonometric and complex operations.


@dataclass(frozen=True)  # Keep decoded rotation data immutable.
class RotationData:  # Store a decoded b-register word.
    theta: float  # Store the Ry angle.
    phase: float  # Store the success-state phase.
    theta_code: int  # Store the fixed-point theta integer.
    phase_code: int  # Store the fixed-point phase integer.

    @property  # Expose magnitude as a calculated value.
    def magnitude(self) -> float:  # Recover rho from theta.
        return float(np.cos(self.theta / 2.0))  # Use rho=cos(theta/2).

    @property  # Expose the reconstructed coefficient.
    def normalized_value(self) -> complex:  # Reconstruct quantized z.
        return self.magnitude * np.exp(1j * self.phase)  # Return rho*exp(i phi).


@dataclass(frozen=True)  # Keep precision settings immutable.
class AnglePhaseCodec:  # Encode theta and phase in one integer word.
    theta_bits: int = 12  # Use twelve bits for theta by default.
    phase_bits: int = 16  # Use sixteen bits for phase by default.

    def __post_init__(self) -> None:  # Validate the precision.
        if self.theta_bits < 1:  # Require at least one theta bit.
            raise ValueError("theta_bits must be positive")  # Reject invalid input.
        if self.phase_bits < 1:  # Require at least one phase bit.
            raise ValueError("phase_bits must be positive")  # Reject invalid input.

    @property  # Expose the b-register width.
    def workspace_bits(self) -> int:  # Calculate total workspace bits.
        return self.theta_bits + self.phase_bits  # Join both field widths.

    @property  # Expose the conceptual b dimension.
    def workspace_dim(self) -> int:  # Calculate the number of b states.
        return 1 << self.workspace_bits  # Return 2**workspace_bits.

    @property  # Expose the maximum theta code.
    def theta_levels(self) -> int:  # Calculate theta's integer range.
        return (1 << self.theta_bits) - 1  # Use all theta bit patterns.

    @property  # Expose the phase modulus.
    def phase_levels(self) -> int:  # Calculate phase's integer range.
        return 1 << self.phase_bits  # Return 2**phase_bits.

    def encode_theta(self, theta: float) -> int:  # Quantize theta.
        bounded = float(np.clip(theta, 0.0, np.pi))  # Restrict to [0,pi].
        return int(round(bounded * self.theta_levels / np.pi))  # Encode theta.

    def decode_theta(self, code: int) -> float:  # Decode theta.
        if not 0 <= code <= self.theta_levels:  # Validate the code.
            raise ValueError("theta code is out of range")  # Reject invalid input.
        return float(code * np.pi / self.theta_levels)  # Recover radians.

    def encode_phase(self, phase: float) -> int:  # Quantize phase.
        wrapped = float(phase % (2.0 * np.pi))  # Wrap into [0,2*pi).
        scaled = wrapped * self.phase_levels / (2.0 * np.pi)  # Scale to codes.
        return int(round(scaled)) % self.phase_levels  # Encode and wrap.

    def decode_phase(self, code: int) -> float:  # Decode phase.
        if not 0 <= code < self.phase_levels:  # Validate the code.
            raise ValueError("phase code is out of range")  # Reject invalid input.
        phase = float(code * 2.0 * np.pi / self.phase_levels)  # Recover radians.
        return phase - 2.0 * np.pi if phase >= np.pi else phase  # Sign phase.

    def pack(self, theta_code: int, phase_code: int) -> int:  # Join fields.
        return theta_code | (phase_code << self.theta_bits)  # Put phase above theta.

    def unpack(self, word: int) -> tuple[int, int]:  # Separate fields.
        theta_mask = (1 << self.theta_bits) - 1  # Select low theta bits.
        return word & theta_mask, word >> self.theta_bits  # Return both codes.

    def encode_value(self, value: complex) -> int:  # Encode normalized z.
        magnitude = float(np.clip(abs(value), 0.0, 1.0))  # Calculate rho.
        theta = 2.0 * float(np.arccos(magnitude))  # Calculate Ry angle.
        phase = float(np.angle(value)) if magnitude > 0.0 else 0.0  # Find phi.
        return self.pack(self.encode_theta(theta), self.encode_phase(phase))  # Pack.

    def decode_word(self, word: int) -> RotationData:  # Decode b data.
        theta_code, phase_code = self.unpack(word)  # Separate fixed-point fields.
        return RotationData(  # Return readable rotation data.
            theta=self.decode_theta(theta_code),  # Decode theta.
            phase=self.decode_phase(phase_code),  # Decode phase.
            theta_code=theta_code,  # Preserve theta's integer.
            phase_code=phase_code,  # Preserve phase's integer.
        )  # Finish the result.
