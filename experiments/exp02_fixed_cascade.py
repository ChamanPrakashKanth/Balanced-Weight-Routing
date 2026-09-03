"""
Experiment 02: Fixed Linear Cascade ($M_1 \to M_2 \to M_3 \to M_4 \to M_5$).
Escalates sequentially through the ladder without capability awareness, resending the full task.
"""

from __future__ import annotations
import argparse
import yaml
from tabulate import tabulate
from rich.console import Console

from bwr.models import ModelProfile
from bwr.router import FixedCascadeRouter
from bwr.orchestrator import Orchestrator
from bwr.telemetry import MetricsAggregator
from benchmarks.loader import BenchmarkSuite

console = Console()


def run_exp02(
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

    router = FixedCascadeRouter(profiles)
    orchestrator = Orchestrator(profiles=profiles, use_mock=use_mock, seed=seed)

    states = orchestrator.run_benchmark(tasks=tasks, router=router, experiment_name="exp02_fixed_cascade")
    summary = MetricsAggregator.summarize_states(states, baseline_states=baseline_states)

    console.print("\n[bold yellow]=== EXP02: Fixed Cascade (M_1 -> M_5) ===[/bold yellow]")
    table_data = [[
        "Fixed Cascade",
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
    parser = argparse.ArgumentParser(description="EXP02: Fixed Cascade Baseline")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_exp02(use_mock=not args.live, max_tasks=args.tasks, seed=args.seed)
