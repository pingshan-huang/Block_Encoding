"""Compatibility imports for Antonio's modular circuit interface (9b091ff).

The model now comes from the shared mathematical core. This avoids maintaining
a second set of centering, Hermiticity, bound, and array-ownership rules. Use
``PMRModel.from_coefficients`` for validated construction, including exact zero
rows and explicit positive bounds for an all-zero centered target.
"""

from _shared_core import PMRModel, next_power_of_two


__all__ = ["PMRModel", "next_power_of_two"]
