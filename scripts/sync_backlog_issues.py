#!/usr/bin/env python3
"""Generate a GitHub Issue sync plan from docs/backlog.md."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from fhe_native_mamba3 import __version__
    from fhe_native_mamba3.artifact_validation import current_git_commit
    from fhe_native_mamba3.backlog_issues import (
        parse_backlog_pbis,
        parse_existing_issues,
        plan_issue_sync,
        summarize_issue_plan,
    )
    from fhe_native_mamba3.cli_support import emit_json_payload

    args = _parse_args()
    backlog_path = Path(args.backlog)
    pbis = parse_backlog_pbis(backlog_path.read_text(encoding="utf-8"))
    include_statuses = _parse_csv(args.statuses)
    existing_rows = _load_existing_issues(args)
    existing = parse_existing_issues(existing_rows)
    plans = plan_issue_sync(
        pbis,
        existing,
        include_statuses=include_statuses,
        source_path=str(backlog_path),
    )
    if args.body_dir:
        _write_issue_bodies(Path(args.body_dir), plans, source_path=str(backlog_path))

    payload = {
        "version": __version__,
        "repo_commit": current_git_commit(ROOT),
        "stage": "backlog-issue-sync-plan",
        "passed": True,
        "dry_run": not args.apply,
        "backlog": str(backlog_path),
        "repo": args.repo,
        "include_statuses": list(include_statuses),
        "pbi_count": len(pbis),
        "tracked_pbi_count": len(plans),
        "existing_issue_count": len(existing),
        "action_counts": summarize_issue_plan(plans),
        "measurement_scope": {
            "claim": "deterministic GitHub Issue sync plan generated from docs/backlog.md",
            "devex_only": True,
            "github_issue_sync": bool(args.apply),
            "github_project_sync": False,
            "full_model_correctness_claimed": False,
            "network_access": bool(args.repo or args.apply),
        },
        "plans": [plan.to_json_dict(source_path=str(backlog_path)) for plan in plans],
    }

    if args.apply:
        apply_results = _apply_issue_plan(plans, repo=args.repo)
        payload["apply_results"] = apply_results
        payload["passed"] = all(result["returncode"] == 0 for result in apply_results)

    emit_json_payload(payload, output_json=args.output_json)
    return 0 if payload["passed"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backlog", default="docs/backlog.md")
    parser.add_argument("--statuses", default="Open,Blocked")
    parser.add_argument("--existing-json", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--body-dir", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _load_existing_issues(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.existing_json:
        return json.loads(Path(args.existing_json).read_text(encoding="utf-8"))
    if not args.repo:
        return []
    command = [
        "gh",
        "issue",
        "list",
        "--repo",
        args.repo,
        "--state",
        "all",
        "--limit",
        "300",
        "--json",
        "number,title,state,labels,body",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _write_issue_bodies(body_dir: Path, plans: tuple[Any, ...], *, source_path: str) -> None:
    body_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        (body_dir / f"{plan.pbi.pbi_id}.md").write_text(
            plan.pbi.to_issue_body(source_path=source_path),
            encoding="utf-8",
        )


def _apply_issue_plan(plans: tuple[Any, ...], *, repo: str) -> list[dict[str, Any]]:
    if not repo:
        msg = "--apply requires --repo"
        raise ValueError(msg)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="backlog-issues-") as tmp:
        tmp_dir = Path(tmp)
        for plan in plans:
            if plan.action == "noop":
                results.append(
                    {
                        "pbi_id": plan.pbi.pbi_id,
                        "action": plan.action,
                        "issue_number": plan.issue_number,
                        "returncode": 0,
                    }
                )
                continue
            body_path = tmp_dir / f"{plan.pbi.pbi_id}.md"
            body_path.write_text(plan.pbi.to_issue_body(), encoding="utf-8")
            command = _gh_command_for_plan(plan, repo=repo, body_path=body_path)
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            results.append(
                {
                    "pbi_id": plan.pbi.pbi_id,
                    "action": plan.action,
                    "issue_number": plan.issue_number,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "command": command,
                }
            )
    return results


def _gh_command_for_plan(plan: Any, *, repo: str, body_path: Path) -> list[str]:
    labels = ",".join(plan.pbi.labels)
    if plan.action == "create":
        return [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            plan.pbi.title,
            "--body-file",
            str(body_path),
            "--label",
            labels,
        ]
    if plan.action in {"update", "reopen"}:
        command = [
            "gh",
            "issue",
            "edit",
            str(plan.issue_number),
            "--repo",
            repo,
            "--title",
            plan.pbi.title,
            "--body-file",
            str(body_path),
        ]
        if labels:
            command.extend(["--add-label", labels])
        return command
    if plan.action == "close":
        return [
            "gh",
            "issue",
            "close",
            str(plan.issue_number),
            "--repo",
            repo,
            "--comment",
            "Closing because the canonical backlog marks this PBI as done or obsolete.",
        ]
    msg = f"unsupported issue sync action: {plan.action}"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
