import re
from pathlib import Path

import fhe_native_mamba3

ROOT = Path(__file__).resolve().parents[1]


def test_source_tree_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None
    assert fhe_native_mamba3.__version__ == match.group(1)
