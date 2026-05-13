"""Backlog-to-GitHub-Issue export helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_PBI_ID_RE = re.compile(r"\bPBI-[A-Z0-9]+-\d+\b")
_PBI_TITLE_RE = re.compile(r"\b(PBI-[A-Z0-9]+-\d+)\b")


@dataclass(frozen=True)
class BacklogPBI:
    """One product backlog item parsed from the canonical backlog table."""

    pbi_id: str
    stage: str
    status: str
    depends_on: tuple[str, ...]
    acceptance_criteria: str

    @property
    def title(self) -> str:
        summary = _short_summary(self.acceptance_criteria)
        return f"{self.pbi_id}: {summary}"

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    "PBI",
                    _stage_label(self.stage),
                    _status_label(self.status),
                    *_priority_labels(self),
                    *(("dependency",) if self.depends_on else ()),
                )
            )
        )

    def to_issue_body(self, *, source_path: str = "docs/backlog.md") -> str:
        dependencies = ", ".join(self.depends_on) if self.depends_on else "none"
        return "\n".join(
            [
                "<!-- generated-from: docs/backlog.md; do not hand-edit canonical PBI data -->",
                "",
                "## PBI",
                "",
                f"- ID: `{self.pbi_id}`",
                f"- Stage: `{self.stage}`",
                f"- Status: `{self.status}`",
                f"- Depends on: `{dependencies}`",
                f"- Source: `{source_path}`",
                "",
                "## Acceptance Criteria",
                "",
                self.acceptance_criteria,
                "",
                "## Dependency Notes",
                "",
                _dependency_markdown(self.depends_on),
                "",
                "## Sync Notes",
                "",
                "- This body is generated from the canonical backlog table.",
                "- Update `docs/backlog.md` first, then regenerate the issue plan.",
                "",
            ]
        )

    def to_json_dict(self, *, source_path: str = "docs/backlog.md") -> dict[str, Any]:
        return {
            "pbi_id": self.pbi_id,
            "title": self.title,
            "stage": self.stage,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "labels": list(self.labels),
            "body": self.to_issue_body(source_path=source_path),
        }


@dataclass(frozen=True)
class ExistingIssue:
    """Minimal GitHub issue metadata used for dry-run sync planning."""

    number: int
    title: str
    state: str
    labels: tuple[str, ...] = ()
    body: str = ""

    @property
    def pbi_id(self) -> str | None:
        match = _PBI_TITLE_RE.search(self.title)
        return match.group(1) if match else None


@dataclass(frozen=True)
class BacklogIssueSyncPlan:
    """One planned issue operation."""

    pbi: BacklogPBI
    action: str
    issue_number: int | None = None
    missing_labels: tuple[str, ...] = ()
    title_changed: bool = False
    body_changed: bool = False

    def to_json_dict(self, *, source_path: str = "docs/backlog.md") -> dict[str, Any]:
        return {
            **self.pbi.to_json_dict(source_path=source_path),
            "action": self.action,
            "issue_number": self.issue_number,
            "missing_labels": list(self.missing_labels),
            "title_changed": self.title_changed,
            "body_changed": self.body_changed,
        }


def parse_backlog_pbis(backlog_text: str) -> tuple[BacklogPBI, ...]:
    """Parse PBIs from the Markdown table in `docs/backlog.md`."""

    rows: list[BacklogPBI] = []
    for raw_line in backlog_text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("| PBI-"):
            continue
        cells = _split_markdown_row(stripped)
        if len(cells) != 5:
            msg = f"expected 5 PBI table cells, got {len(cells)}: {raw_line!r}"
            raise ValueError(msg)
        pbi_id, stage, status, depends_on, acceptance = cells
        rows.append(
            BacklogPBI(
                pbi_id=pbi_id,
                stage=stage,
                status=status,
                depends_on=_parse_dependencies(depends_on),
                acceptance_criteria=acceptance,
            )
        )
    return tuple(rows)


def parse_existing_issues(values: Iterable[dict[str, Any]]) -> tuple[ExistingIssue, ...]:
    """Parse `gh issue list --json number,title,state,labels,body` style rows."""

    issues: list[ExistingIssue] = []
    for value in values:
        labels = value.get("labels", ())
        if isinstance(labels, list):
            label_names = tuple(
                item["name"] if isinstance(item, dict) and "name" in item else str(item)
                for item in labels
            )
        else:
            label_names = ()
        issues.append(
            ExistingIssue(
                number=int(value["number"]),
                title=str(value["title"]),
                state=str(value.get("state", "OPEN")).upper(),
                labels=label_names,
                body=str(value.get("body", "")),
            )
        )
    return tuple(issues)


def plan_issue_sync(
    pbis: Iterable[BacklogPBI],
    existing_issues: Iterable[ExistingIssue] = (),
    *,
    include_statuses: Iterable[str] = ("Open", "Blocked"),
    source_path: str = "docs/backlog.md",
) -> tuple[BacklogIssueSyncPlan, ...]:
    """Build a deterministic create/update/close plan for backlog PBIs."""

    wanted_statuses = {status.lower() for status in include_statuses}
    issue_by_pbi = {issue.pbi_id: issue for issue in existing_issues if issue.pbi_id is not None}
    plans: list[BacklogIssueSyncPlan] = []
    for pbi in pbis:
        should_track = pbi.status.lower() in wanted_statuses
        issue = issue_by_pbi.get(pbi.pbi_id)
        if issue is None:
            if should_track:
                plans.append(BacklogIssueSyncPlan(pbi=pbi, action="create"))
            continue

        if not should_track:
            action = "close" if issue.state == "OPEN" else "noop"
            plans.append(BacklogIssueSyncPlan(pbi=pbi, action=action, issue_number=issue.number))
            continue

        expected_body = pbi.to_issue_body(source_path=source_path)
        missing_labels = tuple(label for label in pbi.labels if label not in set(issue.labels))
        title_changed = issue.title != pbi.title
        body_changed = issue.body != expected_body
        if issue.state != "OPEN":
            action = "reopen"
        elif title_changed or body_changed or missing_labels:
            action = "update"
        else:
            action = "noop"
        plans.append(
            BacklogIssueSyncPlan(
                pbi=pbi,
                action=action,
                issue_number=issue.number,
                missing_labels=missing_labels,
                title_changed=title_changed,
                body_changed=body_changed,
            )
        )
    return tuple(plans)


def summarize_issue_plan(plans: Iterable[BacklogIssueSyncPlan]) -> dict[str, int]:
    """Return action counts for a sync plan."""

    counts = {"create": 0, "update": 0, "close": 0, "reopen": 0, "noop": 0}
    for plan in plans:
        counts[plan.action] = counts.get(plan.action, 0) + 1
    return counts


def _split_markdown_row(value: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in value.strip().strip("|").split("|"))


def _parse_dependencies(value: str) -> tuple[str, ...]:
    normalized = value.strip()
    if normalized.lower() in {"", "none"}:
        return ()
    return tuple(_PBI_ID_RE.findall(normalized))


def _short_summary(value: str, *, max_chars: int = 92) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)[0]
    summary = re.sub(r"\s+", " ", first_sentence).strip(" .")
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 1].rstrip() + "..."


def _stage_label(stage: str) -> str:
    normalized = stage.lower().replace(" ", "-")
    if normalized.startswith("stage-"):
        return normalized
    if normalized == "devex":
        return "devex"
    return normalized.replace("/", "-")


def _status_label(status: str) -> str:
    return f"status-{status.lower().replace(' ', '-')}"


def _priority_labels(pbi: BacklogPBI) -> tuple[str, ...]:
    if pbi.status != "Open":
        return ()
    if pbi.pbi_id in {"PBI-S1-041"}:
        return ("P0",)
    if pbi.stage in {"Stage 0", "Stage 1"}:
        return ("P1",)
    if pbi.pbi_id in {"PBI-S2-009"}:
        return ("P1", "parallelizable")
    if pbi.stage == "DevEx":
        return ("P2",)
    return ("P2",)


def _dependency_markdown(dependencies: tuple[str, ...]) -> str:
    if not dependencies:
        return "- No upstream PBI dependency recorded."
    return "\n".join(f"- Depends on `{dependency}`." for dependency in dependencies)
