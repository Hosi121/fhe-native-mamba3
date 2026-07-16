#!/usr/bin/env python3
"""Run resumable DGX benchmark campaigns without treating tolerance misses as infra failures."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class CampaignInterruptedError(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"campaign interrupted by signal {signum}")
        self.signum = signum


def _raise_campaign_signal(signum: int, _frame: Any) -> None:
    raise CampaignInterruptedError(signum)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timeout_seconds(value: Any) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout < 0:
        raise ValueError("timeout_seconds must be non-negative")
    return timeout or None


def _artifact_paths(env: dict[str, str]) -> list[Path]:
    results_dir = Path(env.get("RESULTS_DIR", str(Path.home() / "fhemamba" / "results")))
    layers = env.get("LAYERS", "5 8 12 24").split()
    tokens = env.get("TOKENS", "1")
    run_tag = env["RUN_TAG"]
    return [results_dir / f"m2_chain_{run_tag}_l{layer}_t{tokens}.json" for layer in layers]


def _load_artifacts(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    issues: list[str] = []
    for path in paths:
        if not path.is_file():
            issues.append(f"missing artifact: {path}")
            continue
        try:
            payload = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid artifact {path}: {exc}")
            continue
        if "status" not in payload or "passed" not in payload:
            issues.append(f"artifact lacks status/passed: {path}")
            continue
        measurements = payload.get("measurements", {})
        timing = payload.get("timing", {})
        scope = payload.get("measurement_scope", {})
        artifacts.append(
            {
                "path": str(path),
                "version": payload.get("version"),
                "stage": payload.get("stage"),
                "status": payload["status"],
                "passed": payload["passed"],
                "measurements": {
                    key: measurements.get(key)
                    for key in (
                        "max_abs_error",
                        "per_token_max_abs_error",
                        "per_token_decrypt_ok",
                        "executed_bootstrap_count",
                        "peak_rss_gib",
                    )
                    if key in measurements
                },
                "timing": {
                    key: timing.get(key)
                    for key in ("eval_seconds", "total_seconds")
                    if key in timing
                },
                "measurement_scope": {
                    key: scope.get(key)
                    for key in ("zero_intermediate_decrypts", "full_layer_chain")
                    if key in scope
                },
            }
        )
    return artifacts, issues


def _max_error(artifacts: list[dict[str, Any]]) -> float:
    errors = []
    for artifact in artifacts:
        value = artifact.get("measurements", {}).get("max_abs_error")
        if isinstance(value, (int, float)):
            errors.append(float(value))
    return max(errors, default=float("inf"))


def _all_decrypt(artifacts: list[dict[str, Any]]) -> bool:
    if not artifacts:
        return False
    for artifact in artifacts:
        values = artifact.get("measurements", {}).get("per_token_decrypt_ok")
        if not isinstance(values, list) or not values or not all(bool(value) for value in values):
            return False
    return True


def _promotion_satisfied(
    condition: dict[str, Any] | None,
    completed: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    if condition is None:
        return True, "unconditional"
    source_name = condition.get("experiment")
    if not isinstance(source_name, str) or source_name not in completed:
        return False, f"promotion source is unavailable: {source_name!r}"
    source = completed[source_name]
    artifacts = source.get("artifacts", [])
    if source.get("infrastructure_ok") is not True:
        return False, f"promotion source has an infrastructure failure: {source_name}"
    threshold = condition.get("max_abs_error_lte")
    if threshold is not None and _max_error(artifacts) > float(threshold):
        return False, f"{source_name} max error exceeds {threshold}"
    if condition.get("all_tokens_decrypt") and not _all_decrypt(artifacts):
        return False, f"{source_name} does not decrypt every token"
    if condition.get("passed") and not all(
        artifact.get("passed") is True for artifact in artifacts
    ):
        return False, f"{source_name} did not pass"
    return True, "promotion gate passed"


def _repo_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "working-tree"


def _run_runner(
    runner: Path,
    *,
    root: Path,
    env: dict[str, str],
    timeout_seconds: float | None,
) -> tuple[int, bool]:
    process = subprocess.Popen(
        [str(runner)],
        cwd=root,
        env=env,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return 124, True
    except BaseException:
        _terminate_process_group(process)
        raise


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    except ProcessLookupError:
        process.wait()


def _gpu_processes(nvidia_smi: str) -> tuple[list[dict[str, int]], str | None]:
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"{nvidia_smi} exited {completed.returncode}"
        return [], f"GPU preflight failed: {message}"
    processes = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0]:
            continue
        try:
            processes.append({"pid": int(fields[0]), "used_memory_mib": int(fields[1])})
        except ValueError:
            return [], f"GPU preflight returned an invalid row: {line!r}"
    return processes, None


def _gpu_utilization(nvidia_smi: str) -> tuple[float, str | None]:
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"{nvidia_smi} exited {completed.returncode}"
        return 0.0, f"GPU utilization preflight failed: {message}"
    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            return 0.0, f"GPU utilization preflight returned an invalid row: {line!r}"
    if not values:
        return 0.0, "GPU utilization preflight returned no GPUs"
    return max(values), None


def _mem_available_gib(meminfo_path: str) -> tuple[float, str | None]:
    try:
        fields = Path(meminfo_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return 0.0, f"memory preflight failed: {exc}"
    for line in fields:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024**2, None
                except ValueError:
                    break
    return 0.0, f"memory preflight found no valid MemAvailable in {meminfo_path}"


def _wait_for_idle_gpu(config: dict[str, Any] | None) -> tuple[float, str | None]:
    if not config or not bool(config.get("required")):
        return 0.0, None
    nvidia_smi = str(config.get("nvidia_smi", "nvidia-smi"))
    poll_seconds = float(config.get("poll_seconds", 30))
    timeout_seconds = float(config.get("timeout_seconds", 7200))
    max_utilization = float(config.get("max_utilization_percent", 5))
    min_mem_available_gib = float(config.get("min_mem_available_gib", 0))
    stable_polls = int(config.get("stable_polls", 1))
    meminfo_path = str(config.get("meminfo_path", "/proc/meminfo"))
    if (
        poll_seconds <= 0
        or timeout_seconds < 0
        or max_utilization < 0
        or min_mem_available_gib < 0
        or stable_polls <= 0
    ):
        raise ValueError("GPU preflight thresholds and polling values are invalid")
    started_at = time.monotonic()
    idle_polls = 0
    while True:
        processes, issue = _gpu_processes(nvidia_smi)
        if issue:
            return time.monotonic() - started_at, issue
        utilization, issue = _gpu_utilization(nvidia_smi)
        if issue:
            return time.monotonic() - started_at, issue
        mem_available_gib, issue = _mem_available_gib(meminfo_path)
        if issue:
            return time.monotonic() - started_at, issue
        idle = (
            not processes
            and utilization <= max_utilization
            and mem_available_gib >= min_mem_available_gib
        )
        idle_polls = idle_polls + 1 if idle else 0
        if idle_polls >= stable_polls:
            return time.monotonic() - started_at, None
        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            occupied = ", ".join(
                f"pid={item['pid']} memory={item['used_memory_mib']}MiB" for item in processes
            )
            detail = occupied or "no process rows"
            return elapsed, (
                f"GPU remained occupied after {timeout_seconds}s: {detail}; "
                f"utilization={utilization}% mem_available={mem_available_gib:.1f}GiB"
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _campaign_payload(
    *,
    name: str,
    version: str,
    repo_commit: str,
    started_at: float,
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    infra_failures = sum(item.get("infrastructure_ok") is False for item in experiments)
    executed = sum(item.get("state") in {"executed", "resumed"} for item in experiments)
    skipped = sum(item.get("state") == "skipped" for item in experiments)
    candidate_passes = sum(item.get("candidate_passed") is True for item in experiments)
    completed_ok = infra_failures == 0
    return {
        "version": version,
        "stage": "fhemamba-dgx-campaign-report",
        "repo_commit": repo_commit,
        "backend": "orchestration",
        "encrypted": False,
        "config": {"input_mode": "campaign-orchestration"},
        "status": "passed" if completed_ok else "failed",
        "passed": completed_ok,
        "campaign": name,
        "experiments": experiments,
        "measurements": {
            "experiments_total": len(experiments),
            "experiments_executed_or_resumed": executed,
            "experiments_skipped": skipped,
            "infrastructure_failures": infra_failures,
            "candidate_passes": candidate_passes,
        },
        "timing": {"campaign_seconds": time.monotonic() - started_at},
        "measurement_scope": {
            "campaign_orchestration_only": True,
            "full_model_correctness_claimed": False,
            "candidate_failures_are_not_infrastructure_failures": True,
            "claim": (
                "Campaign execution and artifact collection only; candidate artifacts carry "
                "their own encrypted-correctness claims."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("fhemamba/experiments/run_dgx_layer_ladder.sh"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGHUP, _raise_campaign_signal)
    signal.signal(signal.SIGTERM, _raise_campaign_signal)
    root = Path(__file__).resolve().parents[2]
    manifest = _read_object(args.manifest)
    name = str(manifest.get("name", args.manifest.stem))
    version = str(manifest.get("version", "0.4.5"))
    default_timeout = _timeout_seconds(manifest.get("timeout_seconds", 0))
    gpu_preflight = manifest.get("gpu_preflight")
    if gpu_preflight is not None and not isinstance(gpu_preflight, dict):
        raise ValueError("manifest gpu_preflight must be an object")
    defaults = manifest.get("defaults", {})
    experiments_spec = manifest.get("experiments", [])
    if not isinstance(defaults, dict) or not isinstance(experiments_spec, list):
        raise ValueError("manifest defaults must be an object and experiments must be a list")

    started_at = time.monotonic()
    records: list[dict[str, Any]] = []
    completed_by_name: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for spec in experiments_spec:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError("each experiment must be an object with a string name")
        experiment_name = spec["name"]
        if experiment_name in seen:
            raise ValueError(f"duplicate experiment name: {experiment_name}")
        seen.add(experiment_name)

        condition = spec.get("when")
        if condition is not None and not isinstance(condition, dict):
            raise ValueError(f"experiment {experiment_name} when must be an object")
        if args.dry_run:
            promoted, promotion_reason = True, "dry-run does not evaluate promotion gates"
        else:
            promoted, promotion_reason = _promotion_satisfied(condition, completed_by_name)
        if not promoted:
            record = {
                "name": experiment_name,
                "state": "skipped",
                "reason": promotion_reason,
                "infrastructure_ok": True,
                "candidate_passed": False,
                "artifacts": [],
            }
            records.append(record)
            completed_by_name[experiment_name] = record
            _write_json(
                args.output_json,
                _campaign_payload(
                    name=name,
                    version=version,
                    repo_commit=_repo_commit(root),
                    started_at=started_at,
                    experiments=records,
                ),
            )
            continue

        overrides = spec.get("env", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"experiment {experiment_name} env must be an object")
        campaign_env = {str(key): str(value) for key, value in defaults.items()}
        campaign_env.update({str(key): str(value) for key, value in overrides.items()})
        campaign_env.setdefault("RUN_TAG", experiment_name)
        artifact_paths = _artifact_paths(campaign_env)

        artifacts: list[dict[str, Any]] = []
        issues: list[str] = []
        state = "dry-run" if args.dry_run else "executed"
        returncode: int | None = None
        timed_out = False
        duration = 0.0
        gpu_wait_seconds = 0.0
        if args.resume and not args.dry_run:
            artifacts, issues = _load_artifacts(artifact_paths)
            if not issues and len(artifacts) == len(artifact_paths):
                state = "resumed"
        if state not in {"resumed", "dry-run"}:
            gpu_wait_seconds, preflight_issue = _wait_for_idle_gpu(gpu_preflight)
            if preflight_issue:
                state = "preflight-failed"
                issues.append(preflight_issue)
            else:
                env = os.environ.copy()
                env.update(campaign_env)
                experiment_start = time.monotonic()
                timeout_value = spec.get("timeout_seconds", default_timeout)
                timeout_seconds = _timeout_seconds(timeout_value)
                returncode, timed_out = _run_runner(
                    args.runner,
                    root=root,
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
                duration = time.monotonic() - experiment_start
                artifacts, issues = _load_artifacts(artifact_paths)
                if timed_out:
                    issues.insert(0, f"runner exceeded timeout of {timeout_seconds} seconds")

        infrastructure_ok = not issues
        candidate_passed = bool(artifacts) and all(
            artifact.get("status") == "passed" and artifact.get("passed") is True
            for artifact in artifacts
        )
        max_error = _max_error(artifacts)
        record = {
            "name": experiment_name,
            "state": state,
            "reason": promotion_reason,
            "returncode": returncode,
            "timed_out": timed_out,
            "seconds": duration,
            "gpu_wait_seconds": gpu_wait_seconds,
            "environment": campaign_env,
            "artifact_paths": [str(path) for path in artifact_paths],
            "artifacts": artifacts,
            "issues": issues,
            "infrastructure_ok": infrastructure_ok,
            "candidate_passed": candidate_passed,
            "max_abs_error": max_error if math.isfinite(max_error) else None,
            "all_tokens_decrypt": _all_decrypt(artifacts),
        }
        records.append(record)
        completed_by_name[experiment_name] = record
        _write_json(
            args.output_json,
            _campaign_payload(
                name=name,
                version=version,
                repo_commit=_repo_commit(root),
                started_at=started_at,
                experiments=records,
            ),
        )
        if not infrastructure_ok and not bool(spec.get("continue_on_infrastructure_failure")):
            return 1

    return 0 if all(record.get("infrastructure_ok") is not False for record in records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignInterruptedError as exc:
        print(f"run_dgx_campaign.py: {exc}", file=sys.stderr)
        raise SystemExit(128 + exc.signum) from exc
    except KeyboardInterrupt:
        print("run_dgx_campaign.py: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_dgx_campaign.py: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
