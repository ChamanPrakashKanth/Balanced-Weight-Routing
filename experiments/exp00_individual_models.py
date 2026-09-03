"""
Experiment 00: Individual Model Capability Benchmarking.
Evaluates every model individually to measure empirical capability vectors w_i across domains
and construct the empirical capability matrix.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
import yaml
from tabulate import tabulate
from rich.console import Console

from bwr.models import ModelProfile, Task
from bwr.capability import EmpiricalCapabilityMatrix
from bwr.router import SmallestOnlyRouter, BaseRouter, RoutingDecision
from bwr.orchestrator import Orchestrator
from bwr.telemetry import MetricsAggregator
from benchmarks.loader import BenchmarkSuite

console = Console()


class SingleModelRouter(BaseRouter):
    def __init__(self, profiles: List[ModelProfile], target_model_id: str):
        super().__init__(profiles)
        self.target_model_id = target_model_id

    def route_step(self, state, last_response=None, last_verification=None):
        return RoutingDecision(
            selected_model_id=self.target_model_id,
            reason=f"evaluate_{self.target_model_id}",
            is_residual=False,
        )

    def should_terminate(self, state, last_verification, last_response):
        if state.current_step >= 1:
            passed = last_verification.passed if last_verification else False
            return True, "verified_success" if passed else "single_step_completed"
        return False, "continue"


def run_exp00(
    models_config_path: str = "configs/models.yaml",
    use_mock: bool = True,
    max_tasks: int = 0,
    seed: int = 42,
    output_matrix_path: str = "results/empirical_capability_matrix.json",
) -> EmpiricalCapabilityMatrix:
    with open(models_config_path, "r", encoding="utf-8") as f:
        m_cfg = yaml.safe_load(f)
    profiles = [ModelProfile(**item) for item in m_cfg.get("ladder", [])]

    suite = BenchmarkSuite()
    tasks = suite.load_all_tasks(max_tasks_per_domain=max_tasks if max_tasks > 0 else None)

    matrix = EmpiricalCapabilityMatrix(model_ids=[p.id for p in profiles])
    orchestrator = Orchestrator(profiles=profiles, use_mock=use_mock, seed=seed)

    table_data = []

    for p in profiles:
        router = SingleModelRouter(profiles, target_model_id=p.id)
        states = orchestrator.run_benchmark(
            tasks=tasks,
            router=router,
            experiment_name=f"exp00_{p.id}",
            capability_matrix=matrix,
        )
        summary = MetricsAggregator.summarize_states(states)
        table_data.append([
            p.id,
            p.name,
            p.ollama_name,
            f"{summary.get('success_rate', 0.0)*100:.1f}%",
            f"{summary.get('avg_tokens', 0):.0f}",
            f"{summary.get('avg_cost', 0):.5f}",
            f"{summary.get('avg_latency', 0):.2f}s",
        ])

    console.print("\n[bold cyan]=== EXP00: Individual Model Capability Benchmarks ===[/bold cyan]")
    headers = ["Model ID", "Name", "Ollama Tag", "Success Rate", "Avg Tokens", "Avg Cost", "Avg Latency"]
    console.print(tabulate(table_data, headers=headers, tablefmt="github"))

    matrix.save_json(output_matrix_path)
    console.print(f"[green]Saved empirical capability matrix to {output_matrix_path}[/green]\n")
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP00: Individual Model Benchmarks")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock model engine")
    parser.add_argument("--live", action="store_true", help="Use live Ollama server")
    parser.add_argument("--tasks", type=int, default=0, help="Max tasks per domain (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_exp00(use_mock=not args.live, max_tasks=args.tasks, seed=args.seed)
