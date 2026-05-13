from __future__ import annotations

from fhe_native_mamba3.backlog_issues import (
    ExistingIssue,
    parse_backlog_pbis,
    parse_existing_issues,
    plan_issue_sync,
    summarize_issue_plan,
)


def test_parse_backlog_pbis_extracts_dependencies_and_labels() -> None:
    pbis = parse_backlog_pbis(_backlog_text())

    assert [pbi.pbi_id for pbi in pbis] == ["PBI-S1-041", "PBI-S2-014"]
    assert pbis[0].depends_on == ("PBI-S1-040",)
    assert "stage-1" in pbis[0].labels
    assert "status-open" in pbis[0].labels
    assert "dependency" in pbis[0].labels
    assert pbis[0].title.startswith("PBI-S1-041: Attempt a bounded")


def test_plan_issue_sync_creates_updates_and_closes() -> None:
    pbis = parse_backlog_pbis(_backlog_text())
    done_body = pbis[1].to_issue_body()
    existing = (
        ExistingIssue(
            number=10,
            title="PBI-S1-041: stale title",
            state="OPEN",
            labels=("PBI",),
            body="stale body",
        ),
        ExistingIssue(
            number=11,
            title=pbis[1].title,
            state="OPEN",
            labels=pbis[1].labels,
            body=done_body,
        ),
    )

    plans = plan_issue_sync(pbis, existing)
    by_id = {plan.pbi.pbi_id: plan for plan in plans}

    assert by_id["PBI-S1-041"].action == "update"
    assert by_id["PBI-S1-041"].issue_number == 10
    assert by_id["PBI-S1-041"].title_changed is True
    assert "stage-1" in by_id["PBI-S1-041"].missing_labels
    assert by_id["PBI-S2-014"].action == "close"
    assert summarize_issue_plan(plans)["update"] == 1


def test_plan_issue_sync_marks_missing_open_pbi_for_create() -> None:
    pbis = parse_backlog_pbis(_backlog_text())

    plans = plan_issue_sync(pbis, ())

    assert len(plans) == 1
    assert plans[0].pbi.pbi_id == "PBI-S1-041"
    assert plans[0].action == "create"


def test_parse_existing_issues_accepts_gh_json_shape() -> None:
    issues = parse_existing_issues(
        [
            {
                "number": 47,
                "title": "PBI-S1-041: Attempt bounded eval",
                "state": "OPEN",
                "labels": [{"name": "PBI"}, {"name": "stage-1"}],
                "body": "body",
            }
        ]
    )

    assert issues[0].pbi_id == "PBI-S1-041"
    assert issues[0].labels == ("PBI", "stage-1")


def _backlog_text() -> str:
    return "\n".join(
        [
            "# Backlog",
            "",
            "| ID | Stage | Status | Depends On | Acceptance Criteria |",
            "| --- | --- | --- | --- | --- |",
            (
                "| PBI-S1-041 | Stage 1 | Open | PBI-S1-040 | "
                "Attempt a bounded Mamba-130M-shape one-layer OpenFHE evaluation. |"
            ),
            (
                "| PBI-S2-014 | Stage 2 | Done | PBI-S2-005, PBI-S2-013 | "
                "Expand learned sketch baselines from a single trace to the matrix. |"
            ),
        ]
    )
