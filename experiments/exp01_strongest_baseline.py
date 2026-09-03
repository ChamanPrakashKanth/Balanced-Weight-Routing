"""
Experiment 01: Strongest-Only Baseline ($D \to M_5$).
Every task is dispatched directly to the largest/strongest model in the ladder.
"""

from __future__ import annotations
import argparse
import yaml
from tabulate import tabulate
from rich.console import Console

from bwr.models import ModelProfile
from bwr.router import StrongestOnlyRouter
from bwr.orchestrator import Orchestrator
from bwr.telemetry import MetricsAggregator
from benchmarks.loader import BenchmarkSuite

console = Console()


def run_exp01(
    models_config_path: str = "configs/models.yaml",
    use_mock: bool = True,
    max_tasks: int = 0,
    seed: int = 42,
):
    with open(models_config_path, "r", encoding="utf-8") as f:
        m_cfg = yaml.safe_load(f)
    profiles = [ModelProfile(**item) for item in m_cfg.get("ladder", [])]

    suite = BenchmarkSuite()
    tasks = suite.load_all_tasks(max_tasks_per_domain=max_tasks if max_tasks > 0 else None)

    router = StrongestOnlyRouter(profiles)
    orchestrator = Orchestrator(profiles=profiles, use_mock=use_mock, seed=seed)

    states = orchestrator.run_benchmark(tasks=tasks, router=router, experiment_name="exp01_strongest_baseline")
    summary = MetricsAggregator.summarize_states(states)

    console.print("\n[bold green]=== EXP01: Strongest-Only Baseline (M_5) ===[/bold green]")
    table_data = [[
        "Strongest Only (M_5)",
        f"{summary.get('success_rate', 0.0)*100:.1f}%",
        f"{summary.get('avg_tokens', 0):.1f}",
        f"{summary.get('avg_cost', 0):.5f}",
        f"{summary.get('avg_latency', 0):.2f}s",
        f"{summary.get('strongest_model_utilization', 0.0)*100:.1f}%",
    ]]
    headers = ["Strategy", "Success Rate (Q)", "Avg Tokens", "Avg Cost (K)", "Avg Latency", "M5 Util%"]
    console.print(tabulate(table_data, headers=headers, tablefmt="github"))
    return states, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP01: Strongest-Only Baseline")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_exp01(use_mock=not args.live, max_tasks=args.tasks, seed=args.seed)
