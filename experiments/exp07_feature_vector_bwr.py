"""
Experiment 07: Feature-Vector Balanced Weight Routing (FV-BWR)
Tests the granular hypothesis:
    routing = f(d_math, d_reasoning, d_code, d_language, d_mechanics, d_planning, d_trap)
with:
    1. 7D Task Demand D and Empirical Model Capability C_i
    2. Imbalance Deficit: R_i = (D - C_i)_+
    3. Balancing Objective: J_i = lambda_R * || W (D - C_i)_+ ||_2^2 + lambda_K * K_i
    4. Closed-Loop Verifier Feedback: D_{t+1} <- R_obs
    5. Train / Unseen Test Generalization Split
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import yaml
from rich.console import Console
from tabulate import tabulate

from bwr.models import Task, ModelProfile, DemandVector, CapabilityVector, FEATURE_DIMENSIONS
from bwr.task import TaskLoader
from bwr.capability import FeatureVectorCapabilityMatrix, EmpiricalCapabilityMatrix
from bwr.router import (
    StrongestOnlyRouter,
    FixedCascadeRouter,
    VerifiedFullTaskRouter,
    VerifiedResidualRouter,
    BalancedWeightRouter,
    FeatureVectorBWRRouter,
)
from bwr.orchestrator import Orchestrator
from bwr.cost import CostTracker
from bwr.verifier import VerifierRegistry

console = Console()


def load_all_tasks(benchmarks_dir: str | Path = "benchmarks") -> List[Task]:
    b_path = Path(benchmarks_dir)
    tasks = []
    for json_file in b_path.glob("*/tasks.json"):
        loaded = TaskLoader.load_from_json(json_file)
        tasks.extend(loaded)
    return tasks


def stratified_train_test_split(
    tasks: List[Task],
    train_ratio: float = 0.60,
    seed: int = 42,
) -> Tuple[List[Task], List[Task]]:
    """
    Splits tasks into Train and Unseen Test sets stratified by domain.
    """
    random.seed(seed)
    by_domain: Dict[str, List[Task]] = {}
    for t in tasks:
        d_key = t.domain.value
        by_domain.setdefault(d_key, []).append(t)

    train_tasks: List[Task] = []
    test_tasks: List[Task] = []

    for d_key, d_tasks in by_domain.items():
        shuffled = list(d_tasks)
        random.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio))
        train_tasks.extend(shuffled[:n_train])
        test_tasks.extend(shuffled[n_train:])

    # Shuffle final sets
    random.shuffle(train_tasks)
    random.shuffle(test_tasks)
    return train_tasks, test_tasks


def calibrate_capabilities_on_train_set(
    train_tasks: List[Task],
    profiles: List[ModelProfile],
    orchestrator: Orchestrator,
) -> FeatureVectorCapabilityMatrix:
    """
    Phase 1: Calibrate 7D model capability vectors C_i on the Training Set.
    """
    matrix = FeatureVectorCapabilityMatrix(model_ids=[p.id for p in profiles])
    console.print(f"\n[bold yellow]Phase 1: Calibrating 7D Capability Vectors on {len(train_tasks)} Training Tasks...[/bold yellow]")

    # Execute each model on each train task to measure empirical load responses
    for task in train_tasks:
        for profile in profiles:
            client = orchestrator.clients[profile.id]
            resp = client.generate(prompt=task.prompt, task=task)
            verif = VerifierRegistry.verify_task(task, resp)
            matrix.record_observation(
                model_id=profile.id,
                demand=task.demand,
                passed=verif.passed,
                score=verif.score,
            )

    # Print calibrated capability vectors
    table_rows = []
    for p in profiles:
        c_vec = matrix.get_capability(p.id)
        table_rows.append([
            p.id,
            p.name,
            f"{c_vec.math:.2f}",
            f"{c_vec.reasoning:.2f}",
            f"{c_vec.code:.2f}",
            f"{c_vec.language:.2f}",
            f"{c_vec.mechanics:.2f}",
            f"{c_vec.planning:.2f}",
            f"{c_vec.trap:.2f}",
        ])

    headers = ["ID", "Model Name", "Math", "Reason", "Code", "Lang", "Mech", "Plan", "Trap"]
    console.print("\n[bold cyan]=== Calibrated Empirical Capability Vectors C_i ===[/bold cyan]")
    console.print(tabulate(table_rows, headers=headers, tablefmt="github"))

    return matrix


def run_fv_bwr_experiment(
    use_mock: bool = True,
    seed: int = 42,
    train_ratio: float = 0.60,
    max_tasks: int = 0,
) -> Dict[str, Any]:
    console.print(
        "[bold cyan]+-----------------------------------------------------------------------------+\n"
        "| Feature-Vector Balanced Weight Routing (FV-BWR) Generalization Experiment   |\n"
        "+-----------------------------------------------------------------------------+[/bold cyan]"
    )

    # Load configurations
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    profiles = [ModelProfile(**item) for item in models_cfg["ladder"]]

    all_tasks = load_all_tasks()
    if max_tasks > 0:
        all_tasks = all_tasks[:max_tasks]
    console.print(f"Total benchmark tasks loaded: {len(all_tasks)}")

    # Stratified Train / Test Split
    train_tasks, test_tasks = stratified_train_test_split(all_tasks, train_ratio=train_ratio, seed=seed)
    console.print(f"Dataset Split: [bold green]{len(train_tasks)} Training Tasks[/bold green] | [bold magenta]{len(test_tasks)} Unseen Test Tasks[/bold magenta]")

    orchestrator = Orchestrator(profiles=profiles, use_mock=use_mock, seed=seed)

    # Phase 1: Calibrate Capability Matrix on Train Set
    fv_matrix = calibrate_capabilities_on_train_set(train_tasks, profiles, orchestrator)
    fv_matrix.save_json("results/calibrated_fv_capability_matrix.json")

    # Also build coarse domain matrix for comparison
    coarse_matrix = EmpiricalCapabilityMatrix(model_ids=[p.id for p in profiles])
    for t in train_tasks:
        for p in profiles:
            client = orchestrator.clients[p.id]
            resp = client.generate(prompt=t.prompt, task=t)
            verif = VerifierRegistry.verify_task(t, resp)
            coarse_matrix.record_attempt(p.id, t.domain, verif.passed, cost=0.01)

    # Phase 2: Evaluate on Unseen Test Tasks
    console.print(f"\n[bold yellow]Phase 2: Evaluating Routing Policies on {len(test_tasks)} UNSEEN Test Tasks...[/bold yellow]")

    policies = [
        ("M5_only", "M5 Only (Baseline)", StrongestOnlyRouter(profiles)),
        ("Fixed_Full", "Policy A: Fixed Cascade (Full Context)", VerifiedFullTaskRouter(profiles)),
        ("Fixed_VRR", "Policy B: VRR (Fixed + Residual)", VerifiedResidualRouter(profiles)),
        ("Coarse_BWR", "Policy C: Coarse BWR (Domain-only)", BalancedWeightRouter(profiles, capability_matrix=coarse_matrix, allow_skipping=True)),
        ("FV_BWR_Full", "Policy E: FV-BWR (Full Context)", FeatureVectorBWRRouter(profiles, feature_matrix=fv_matrix, use_residual=False)),
        ("FV_BWR_VRR", "Policy F: FV-BWR + VRR (Feature Vector + Residual)", FeatureVectorBWRRouter(profiles, feature_matrix=fv_matrix, use_residual=True)),
    ]

    all_results: Dict[str, List[Any]] = {}
    all_summaries: Dict[str, Dict[str, Any]] = {}

    for key, label, router in policies:
        console.print(f"Running [cyan]{label}[/cyan] on {len(test_tasks)} unseen tasks...")
        states = orchestrator.run_benchmark(tasks=test_tasks, router=router, experiment_name=f"exp07_{key}")
        all_results[key] = states

        # Calculate metrics
        n = len(states)
        successes = sum(1 for s in states if s.success)
        succ_rate = (successes / n) * 100.0 if n > 0 else 0.0
        avg_tokens = float(np.mean([s.accumulated_tokens for s in states])) if n > 0 else 0.0
        avg_cost = float(np.mean([s.accumulated_cost for s in states])) if n > 0 else 0.0
        avg_latency = float(np.mean([s.accumulated_latency for s in states])) if n > 0 else 0.0
        m5_util = (sum(1 for s in states if any(h.model_id == "model_5" for h in s.history)) / n) * 100.0 if n > 0 else 0.0

        all_summaries[key] = {
            "label": label,
            "success_rate": succ_rate,
            "avg_tokens": avg_tokens,
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "m5_util": m5_util,
        }

    # Baseline references
    base_cost = all_summaries["M5_only"]["avg_cost"]
    base_tokens = all_summaries["M5_only"]["avg_tokens"]
    base_succ = all_summaries["M5_only"]["success_rate"]

    table_rows = []
    for key, label, _ in policies:
        summ = all_summaries[key]
        c_val = summ["avg_cost"]
        t_val = summ["avg_tokens"]
        s_c = CostTracker.compute_cost_savings(c_val, base_cost) * 100.0
        s_t = CostTracker.compute_token_savings(t_val, base_tokens) * 100.0
        summ["cost_savings_pct"] = s_c
        summ["token_savings_pct"] = s_t

        c_str = f"{s_c:+.1f}%" if key != "M5_only" else "0.0% (Base)"
        t_str = f"{s_t:+.1f}%" if key != "M5_only" else "0.0% (Base)"

        table_rows.append([
            label,
            f"{summ['success_rate']:.1f}%",
            f"{summ['avg_tokens']:.1f}",
            f"{summ['avg_cost']:.5f}",
            c_str,
            t_str,
            f"{summ['m5_util']:.1f}%",
            f"{summ['avg_latency']:.2f}s",
        ])

    headers = [
        "Routing Policy",
        "Test Success (Q)",
        "Avg Tokens",
        "Avg Cost (K)",
        "Cost Saving (S_C)",
        "Token Saving (S_T)",
        "M5 Util %",
        "Latency",
    ]

    console.print("\n[bold green]============================== UNSEEN TEST GENERALIZATION RESULTS ==============================[/bold green]")
    table_md = tabulate(table_rows, headers=headers, tablefmt="github")
    console.print(table_md)

    # Save summary report
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "exp07_feature_vector_bwr_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Feature-Vector Balanced Weight Routing (FV-BWR) Generalization Report\n\n")
        f.write(f"**Dataset**: {len(all_tasks)} tasks total | {len(train_tasks)} Train | {len(test_tasks)} Unseen Test\n\n")
        f.write("### Generalization Comparison on Unseen Tasks\n\n")
        f.write(table_md)
        f.write("\n\n### Scientific Conclusions\n")
        f.write(f"- **M5 Only Success**: {base_succ:.1f}%\n")
        f.write(f"- **FV-BWR Test Success**: {all_summaries['FV_BWR_VRR']['success_rate']:.1f}%\n")
        f.write(f"- **FV-BWR Cost Savings**: {all_summaries['FV_BWR_VRR']['cost_savings_pct']:+.1f}%\n")

    console.print(f"\n[green]Saved report to {report_path}[/green]")
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature-Vector BWR Experiment")
    parser.add_argument("--live", action="store_true", help="Use live Ollama models instead of mock engine")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.60, help="Train/test split ratio")
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional max tasks limit")
    args = parser.parse_args()

    run_fv_bwr_experiment(
        use_mock=not args.live,
        seed=args.seed,
        train_ratio=args.train_ratio,
        max_tasks=args.max_tasks,
    )
