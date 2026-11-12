"""Fixed-point encoding used by the lowercase-r oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationData:
    """Decoded flag-rotation parameters."""

    theta: float
    phase: float
    theta_code: int
    phase_code: int

    @property
    def magnitude(self) -> float:
        return float(np.cos(self.theta / 2.0))

    @property
    def normalized_value(self) -> complex:
        return self.magnitude * np.exp(1j * self.phase)


@dataclass(frozen=True)
class AnglePhaseCodec:
    """Quantize theta and phase and pack them into one integer word."""

    theta_bits: int = 12
    phase_bits: int = 16

    def __post_init__(self) -> None:
        if self.theta_bits < 1 or self.phase_bits < 1:
            raise ValueError("theta_bits and phase_bits must be positive")
        if self.workspace_bits > 64:
            raise ValueError("this implementation supports at most 64 workspace bits")

    @property
    def workspace_bits(self) -> int:
        return self.theta_bits + self.phase_bits

    @property
    def theta_levels(self) -> int:
        return (1 << self.theta_bits) - 1

    @property
    def phase_levels(self) -> int:
        return 1 << self.phase_bits

    def encode_theta(self, theta: float) -> int:
        bounded = float(np.clip(theta, 0.0, np.pi))
        return int(round(bounded * self.theta_levels / np.pi))

    def decode_theta(self, code: int) -> float:
        if not 0 <= code <= self.theta_levels:
            raise ValueError("theta code is out of range")
        return float(code * np.pi / self.theta_levels)

    def encode_phase(self, phase: float) -> int:
        wrapped = float(phase % (2.0 * np.pi))
        return int(round(wrapped * self.phase_levels / (2.0 * np.pi))) % self.phase_levels

    def decode_phase(self, code: int) -> float:
        if not 0 <= code < self.phase_levels:
            raise ValueError("phase code is out of range")
        phase = float(code * 2.0 * np.pi / self.phase_levels)
        return phase - 2.0 * np.pi if phase >= np.pi else phase

    def pack(self, theta_code: int, phase_code: int) -> int:
        return theta_code | (phase_code << self.theta_bits)

    def unpack(self, word: int) -> tuple[int, int]:
        theta_mask = (1 << self.theta_bits) - 1
        return word & theta_mask, word >> self.theta_bits

    def encode_value(self, value: complex) -> int:
        magnitude = float(np.clip(abs(value), 0.0, 1.0))
        theta = 2.0 * float(np.arccos(magnitude))
        phase = float(np.angle(value)) if magnitude > 0.0 else 0.0
        return self.pack(self.encode_theta(theta), self.encode_phase(phase))

    def decode_word(self, word: int) -> RotationData:
        theta_code, phase_code = self.unpack(word)
        return RotationData(
            theta=self.decode_theta(theta_code),
            phase=self.decode_phase(phase_code),
            theta_code=theta_code,
            phase_code=phase_code,
        )
