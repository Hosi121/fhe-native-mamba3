"""Environment workarounds. Import before transformers.

The .venv ships torchvision 0.25 against torch 2.11 — a C++ op-registration
mismatch that makes ``import torchvision`` raise RuntimeError. transformers'
lazy loader then fails to import any modeling class. Nothing in this project
uses torchvision, so we block the import outright: a ``None`` entry in
sys.modules makes ``import torchvision`` raise ImportError immediately, which
transformers handles as "torchvision not installed".

Remove once the environment pins a matching torchvision (or drops it).
"""

from __future__ import annotations

import sys


def block_broken_torchvision() -> None:
    if "torchvision" not in sys.modules:
        sys.modules["torchvision"] = None  # type: ignore[assignment]
