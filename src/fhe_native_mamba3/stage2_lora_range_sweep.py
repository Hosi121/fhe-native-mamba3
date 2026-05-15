"""Sweep LoRA range-tuning settings for rank/gate projection payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.range_finetune import LoRAConfig, RangeLossConfig
from fhe_native_mamba3.stage1_rank_gate_payload import Stage1RankGatePayload
from fhe_native_mamba3.stage2_lora_range_smoke import (
    Stage2LoRARangeMetrics,
    run_lora_range_smoke,
)


@dataclass(frozen=True)
class Stage2LoRARangeSweepRow:
    """One LoRA range-tuning sweep row."""

    row_index: int
    seed: int
    steps: int
    range_weight: float
    learning_rate: float
    lora_rank: int
    lora_alpha: float
    passed: bool
    before: Stage2LoRARangeMetrics
    after: Stage2LoRARangeMetrics
    lora_parameter_count: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["before"] = self.before.to_json_dict()
        payload["after"] = self.after.to_json_dict()
        return payload


@dataclass(frozen=True)
class Stage2LoRARangeSweepResult:
    """Aggregate result for a LoRA range-tuning sweep."""

    passed: bool
    best_row_index: int
    rows: tuple[Stage2LoRARangeSweepRow, ...]
    target_abs: float
    sample_count: int
    noise_scale: float
    device: str
    measurement_scope: dict[str, Any]

    @property
    def best_row(self) -> Stage2LoRARangeSweepRow:
        return self.rows[self.best_row_index]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "best_row_index": self.best_row_index,
            "best_row": self.best_row.to_json_dict(),
            "row_count": len(self.rows),
            "rows": [row.to_json_dict() for row in self.rows],
            "target_abs": self.target_abs,
            "sample_count": self.sample_count,
            "noise_scale": self.noise_scale,
            "device": self.device,
            "measurement_scope": self.measurement_scope,
        }


def run_lora_range_sweep(
    payload: Stage1RankGatePayload,
    *,
    seeds: tuple[int, ...],
    steps_values: tuple[int, ...],
    range_weights: tuple[float, ...],
    learning_rates: tuple[float, ...] = (1e-2,),
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    target_abs: float = 6.0,
    sample_count: int = 256,
    noise_scale: float = 0.01,
    device: str = "cpu",
) -> Stage2LoRARangeSweepResult:
    """Run a Cartesian sweep and pick the lowest post-tuning max excess row."""

    if not seeds:
        msg = "seeds must not be empty"
        raise ValueError(msg)
    if not steps_values:
        msg = "steps_values must not be empty"
        raise ValueError(msg)
    if not range_weights:
        msg = "range_weights must not be empty"
        raise ValueError(msg)
    if not learning_rates:
        msg = "learning_rates must not be empty"
        raise ValueError(msg)
    rows: list[Stage2LoRARangeSweepRow] = []
    for seed in seeds:
        for steps in steps_values:
            for range_weight in range_weights:
                for learning_rate in learning_rates:
                    result = run_lora_range_smoke(
                        payload,
                        sample_count=sample_count,
                        noise_scale=noise_scale,
                        steps=steps,
                        learning_rate=learning_rate,
                        lora_config=LoRAConfig(rank=lora_rank, alpha=lora_alpha),
                        range_loss_config=RangeLossConfig(
                            target_abs=target_abs,
                            weight=range_weight,
                            reduction="mean",
                        ),
                        seed=seed,
                        device=device,
                    )
                    rows.append(
                        Stage2LoRARangeSweepRow(
                            row_index=len(rows),
                            seed=seed,
                            steps=steps,
                            range_weight=range_weight,
                            learning_rate=learning_rate,
                            lora_rank=lora_rank,
                            lora_alpha=lora_alpha,
                            passed=result.passed,
                            before=result.before,
                            after=result.after,
                            lora_parameter_count=result.lora_parameter_count,
                        )
                    )
    best_index = min(
        range(len(rows)),
        key=lambda index: (
            rows[index].after.max_excess,
            rows[index].after.total_loss,
            rows[index].after.task_mse,
        ),
    )
    return Stage2LoRARangeSweepResult(
        passed=any(row.passed for row in rows),
        best_row_index=best_index,
        rows=tuple(rows),
        target_abs=target_abs,
        sample_count=sample_count,
        noise_scale=noise_scale,
        device=device,
        measurement_scope={
            "stage2_lora_range_sweep": True,
            "lora_training_executed": True,
            "rank_gate_projection_only": True,
            "encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Sweeps plaintext LoRA range-tuning settings on a rank/gate "
                "projection payload; it does not execute encrypted inference or "
                "claim full-model quality."
            ),
        },
    )


__all__ = [
    "Stage2LoRARangeSweepResult",
    "Stage2LoRARangeSweepRow",
    "run_lora_range_sweep",
]
