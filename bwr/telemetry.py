"""
Telemetry logger for JSONL records, reproducible run IDs, and statistical metrics aggregation.
"""

from __future__ import annotations
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats
from bwr.models import RoutingState


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown_commit"


class TelemetryLogger:
    """
    Writes immutable JSONL telemetry events with unique run IDs.
    """

    def __init__(self, log_dir: str | Path = "results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.git_commit = get_git_commit()

    def generate_run_id(self, experiment_name: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rand_suffix = os.urandom(3).hex()
        return f"{experiment_name}_{timestamp}_{rand_suffix}"

    def log_task_step(
        self,
        run_id: str,
        experiment_name: str,
        task_id: str,
        domain: str,
        step: int,
        model_id: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_seconds: float,
        cost: float,
        verification_score: float,
        passed: bool,
        failures: List[Dict[str, Any]],
        accumulated_cost: float,
        accumulated_tokens: int,
        is_residual: bool = False,
        residual_tokens: Optional[int] = None,
        context_reduction_ratio: Optional[float] = None,
        self_reported_confidence: Optional[float] = None,
    ) -> None:
        log_file = self.log_dir / f"{run_id}.jsonl"
        entry: Dict[str, Any] = {
            "run_id": run_id,
            "experiment": experiment_name,
            "git_commit": self.git_commit,
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "domain": domain,
            "step": step,
            "model_id": model_id,
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_seconds": latency_seconds,
            "cost": cost,
            "accumulated_cost": accumulated_cost,
            "accumulated_tokens": accumulated_tokens,
            "verification_score": verification_score,
            "passed": passed,
            "failure_count": len(failures),
            "failures": failures,
            "is_residual": is_residual,
            "residual_tokens": residual_tokens,
            "context_reduction_ratio": context_reduction_ratio,
            "self_reported_confidence": self_reported_confidence,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class MetricsAggregator:
    """
    Computes rigorous aggregate statistics, cost reductions, escalation rates,
    and confidence intervals across benchmark runs.
    """

    @staticmethod
    def summarize_states(
        states: List[RoutingState],
        baseline_states: Optional[List[RoutingState]] = None,
        strongest_model_id: str = "model_5",
    ) -> Dict[str, Any]:
        n_tasks = len(states)
        if n_tasks == 0:
            return {}

        successes = sum(1 for s in states if s.success)
        success_rate = successes / n_tasks

        total_costs = [s.accumulated_cost for s in states]
        total_tokens = [s.accumulated_tokens for s in states]
        latencies = [s.accumulated_latency for s in states]

        avg_cost = float(np.mean(total_costs))
        std_cost = float(np.std(total_costs, ddof=1)) if n_tasks > 1 else 0.0

        avg_tokens = float(np.mean(total_tokens))
        std_tokens = float(np.std(total_tokens, ddof=1)) if n_tasks > 1 else 0.0

        avg_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies, ddof=1)) if n_tasks > 1 else 0.0

        # Escalation rate: fraction of tasks that required > 1 step
        escalated = sum(1 for s in states if len(s.history) > 1)
        escalation_rate = float(escalated / n_tasks)

        # Strongest model utilization: fraction of tasks touching model_5
        touched_strongest = sum(
            1 for s in states if any(h.model_id == strongest_model_id for h in s.history)
        )
        strongest_utilization = float(touched_strongest / n_tasks)

        # Context reduction average (for residual steps)
        context_reductions = [
            h.context_reduction_ratio
            for s in states
            for h in s.history
            if h.context_reduction_ratio is not None
        ]
        avg_context_reduction = float(np.mean(context_reductions)) if context_reductions else 0.0

        summary: Dict[str, Any] = {
            "n_tasks": n_tasks,
            "success_rate": success_rate,
            "avg_cost": avg_cost,
            "std_cost": std_cost,
            "avg_tokens": avg_tokens,
            "std_tokens": std_tokens,
            "avg_latency": avg_latency,
            "std_latency": std_latency,
            "escalation_rate": escalation_rate,
            "strongest_model_utilization": strongest_utilization,
            "avg_context_reduction": avg_context_reduction,
        }

        # Comparative metrics if baseline provided
        if baseline_states and len(baseline_states) == n_tasks:
            base_avg_cost = float(np.mean([b.accumulated_cost for b in baseline_states]))
            base_avg_tokens = float(np.mean([b.accumulated_tokens for b in baseline_states]))
            base_success = sum(1 for b in baseline_states if b.success) / n_tasks

            cost_savings = 1.0 - (avg_cost / base_avg_cost) if base_avg_cost > 1e-9 else 0.0
            token_savings = 1.0 - (avg_tokens / base_avg_tokens) if base_avg_tokens > 0 else 0.0

            # Paired t-test on cost
            try:
                t_stat, p_val = stats.ttest_rel(total_costs, [b.accumulated_cost for b in baseline_states])
            except Exception:
                t_stat, p_val = 0.0, 1.0

            summary["baseline_success_rate"] = base_success
            summary["baseline_avg_cost"] = base_avg_cost
            summary["cost_savings_pct"] = float(cost_savings * 100.0)
            summary["token_savings_pct"] = float(token_savings * 100.0)
            summary["paired_t_stat"] = float(t_stat)
            summary["paired_p_value"] = float(p_val)

        return summary
