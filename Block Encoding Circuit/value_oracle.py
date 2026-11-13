"""Shared VALUE calculations behind Antonio's modular interface (9b091ff)."""

from __future__ import annotations

from _shared_core import CoreValueOracle


class ValueOracle(CoreValueOracle):
    """Expose the original ``value`` spelling for the validated VALUE table.

    This is classical construction data: each entry is the centered source
    coefficient divided by its row bound. A zero-bound row returns zero without
    division. The inherited table-copy and model-ownership protections are the
    same ones used by the shared mathematical core; lowercase r implements the
    reversible lookup on qubits.
    """

    def value(self, move_index: int, x: int) -> complex:
        """Return one entry by move-row position and source basis-state index."""
        return self.normalized_value(move_index, x)
