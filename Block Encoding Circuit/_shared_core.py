"""Locate the shared PMR core for Antonio's standalone oracle entry points.

The original modular interface comes from Antonio's commit 9b091ff. The local
integration keeps that interface while using the validated mathematical core
in ``code``. Appending its directory preserves this folder's ``main.py`` when
the original command-line tools and tests import it by name.
"""

from __future__ import annotations

from pathlib import Path
import sys


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.append(str(CODE_DIRECTORY))

from pmr_model import PMRModel, next_power_of_two
from value import ValueOracle as CoreValueOracle


__all__ = ["PMRModel", "next_power_of_two", "CoreValueOracle"]
