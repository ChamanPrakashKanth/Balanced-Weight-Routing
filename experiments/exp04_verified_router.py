"""
Experiment 04: Verified Full-Task Escalation ($D \to M_1 \to D \to M_2 \dots$).
Escalates upon external verification failure, but resends the full original task without residual reduction.
"""

from __future__ import annotations
import argparse
import yaml
from tabulate import tabulate
from rich.console import Console

from bwr.models import ModelProfile
from bwr.router import VerifiedFullTaskRouter
from bwr.orchestrator import Orchestrator
from bwr.telemetry import MetricsAggregator
from benchmarks.loader import BenchmarkSuite

console = Console()


def run_exp04(
    models_config_path: str = "configs/models.yaml",
    use_mock: bool = True,
    max_tasks: int = 0,
    seed: int = 42,
    baseline_states=None,
):
    with open(models_config_path, "r", encoding="utf-8") as f:
        m_cfg = yaml.safe_load(f)
    profiles = [ModelProfile(**item) for item in m_cfg.get("ladder", [])]

    suite = BenchmarkSuite()
    tasks = suite.load_all_tasks(max_tasks_per_domain=max_tasks if max_tasks > 0 else None)

    router = VerifiedFullTaskRouter(profiles)
    orchestrator = Orchestrator(profiles=profiles, use_mock=use_mock, seed=seed)

    states = orchestrator.run_benchmark(tasks=tasks, router=router, experiment_name="exp04_verified_full_escalation")
    summary = MetricsAggregator.summarize_states(states, baseline_states=baseline_states)

    console.print("\n[bold magenta]=== EXP04: Verified Full-Task Escalation ===[/bold magenta]")
    table_data = [[
        "Verified Full Task",
        f"{summary.get('success_rate', 0.0)*100:.1f}%",
        f"{summary.get('avg_tokens', 0):.1f}",
        f"{summary.get('avg_cost', 0):.5f}",
        f"{summary.get('cost_savings_pct', 0.0):+.1f}%" if baseline_states else "N/A",
        f"{summary.get('escalation_rate', 0.0)*100:.1f}%",
        f"{summary.get('strongest_model_utilization', 0.0)*100:.1f}%",
    ]]
    headers = ["Strategy", "Success Rate (Q)", "Avg Tokens", "Avg Cost (K)", "Cost Saving vs M5", "Escalation%", "M5 Util%"]
    console.print(tabulate(table_data, headers=headers, tablefmt="github"))
    return states, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP04: Verified Full-Task Escalation")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_exp04(use_mock=not args.live, max_tasks=args.tasks, seed=args.seed)
