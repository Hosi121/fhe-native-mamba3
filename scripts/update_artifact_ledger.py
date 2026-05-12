#!/usr/bin/env python3
"""Append collected artifact ledger rows to docs/artifact_ledger.md."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class LedgerRow:
    """One six-column artifact ledger Markdown row."""

    pbi_id: str
    job_id: str
    artifact_path: str
    commit_tag: str
    status: str
    result_memo: str

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize_key_cell(self.job_id), _normalize_key_cell(self.artifact_path))

    def to_markdown(self) -> str:
        return (
            f"| {self.pbi_id} | {self.job_id} | {self.artifact_path} | "
            f"{self.commit_tag} | {self.status} | {self.result_memo} |"
        )


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.cli_support import emit_json_payload

    args = _parse_args()
    ledger_path = Path(args.ledger)
    ledger_text = ledger_path.read_text(encoding="utf-8")
    candidate_rows = _candidate_rows(args.from_json)
    result = update_artifact_ledger_text(ledger_text, candidate_rows)
    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "artifact-ledger-update",
        "passed": not result["conflicts"],
        "ledger": str(ledger_path),
        "sources": list(args.from_json),
        "dry_run": not args.write,
        **{key: value for key, value in result.items() if key != "ledger_text"},
        "measurement_scope": {
            "claim": "local artifact-ledger update planning/apply result",
            "devex_only": True,
            "network_access": False,
            "github_project_sync": False,
        },
    }
    if args.write and not result["conflicts"]:
        ledger_path.write_text(str(result["ledger_text"]), encoding="utf-8")
    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def update_artifact_ledger_text(
    ledger_text: str,
    candidate_rows: tuple[LedgerRow, ...],
) -> dict[str, Any]:
    """Return updated ledger text and a deterministic append/skip/conflict summary."""

    existing_rows = _existing_rows(ledger_text)
    existing_by_key = {row.key: row for row in existing_rows}
    additions: list[LedgerRow] = []
    skipped: list[LedgerRow] = []
    conflicts: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        if row.key in seen_candidate_keys:
            skipped.append(row)
            continue
        seen_candidate_keys.add(row.key)
        existing = existing_by_key.get(row.key)
        if existing is None:
            additions.append(row)
        elif existing.to_markdown() == row.to_markdown():
            skipped.append(row)
        else:
            conflicts.append(
                {
                    "key": list(row.key),
                    "existing": existing.to_markdown(),
                    "candidate": row.to_markdown(),
                }
            )
    updated_text = ledger_text if conflicts else _append_rows(ledger_text, additions)
    return {
        "candidate_count": len(candidate_rows),
        "added_count": len(additions) if not conflicts else 0,
        "skipped_existing_count": len(skipped),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "added_rows": [row.to_markdown() for row in additions] if not conflicts else [],
        "skipped_rows": [row.to_markdown() for row in skipped],
        "ledger_text": updated_text,
    }


def parse_ledger_row(value: str) -> LedgerRow:
    """Parse a six-column artifact ledger Markdown row."""

    raw = value.strip()
    parts = [part.strip() for part in raw.strip("|").split("|")]
    if len(parts) != 6:
        msg = f"ledger row must have 6 columns: {value!r}"
        raise ValueError(msg)
    if parts[0] == "PBI ID" or set(parts[0]) <= {"-", ":"}:
        msg = f"ledger row must be a data row: {value!r}"
        raise ValueError(msg)
    return LedgerRow(
        pbi_id=parts[0],
        job_id=parts[1],
        artifact_path=parts[2],
        commit_tag=parts[3],
        status=parts[4],
        result_memo=parts[5],
    )


def _candidate_rows(paths: list[str]) -> tuple[LedgerRow, ...]:
    rows: list[LedgerRow] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger_rows = payload.get("ledger_rows") if isinstance(payload, dict) else None
        if not isinstance(ledger_rows, list):
            msg = f"{path} must contain a top-level ledger_rows list"
            raise ValueError(msg)
        for item in ledger_rows:
            if not isinstance(item, str):
                msg = f"{path} contains a non-string ledger row"
                raise ValueError(msg)
            rows.append(parse_ledger_row(item))
    return tuple(rows)


def _existing_rows(ledger_text: str) -> tuple[LedgerRow, ...]:
    rows = []
    for line in ledger_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        try:
            rows.append(parse_ledger_row(stripped))
        except ValueError:
            continue
    return tuple(rows)


def _append_rows(ledger_text: str, rows: list[LedgerRow]) -> str:
    if not rows:
        return ledger_text
    prefix = ledger_text if ledger_text.endswith("\n") else f"{ledger_text}\n"
    return prefix + "\n".join(row.to_markdown() for row in rows) + "\n"


def _normalize_key_cell(value: str) -> str:
    return value.strip().strip("`")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="append", required=True)
    parser.add_argument("--ledger", default="docs/artifact_ledger.md")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
