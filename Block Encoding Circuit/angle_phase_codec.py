"""Shared circuit codec for Antonio's modular interface (9b091ff).

Both Qiskit entry points use the same supported precision rules and fixed-point
packing. The NumPy sparse simulator keeps its existing codec implementation.
"""

import _shared_core  # Make the sibling core importable from standalone scripts.
from qiskit_oracles import QiskitAnglePhaseCodec as AnglePhaseCodec
from rotation_codec import RotationData


__all__ = ["AnglePhaseCodec", "RotationData"]
