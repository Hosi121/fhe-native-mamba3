"""Shared support helpers for CLI output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_payload_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _write_json_text(text: str, output_json: str | Path) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def write_json_payload(payload: dict[str, Any], output_json: str | Path) -> None:
    _write_json_text(json_payload_text(payload), output_json)


def emit_json_payload(payload: dict[str, Any], *, output_json: str | Path | None = "") -> None:
    text = json_payload_text(payload)
    if output_json:
        _write_json_text(text, output_json)
    print(text)
