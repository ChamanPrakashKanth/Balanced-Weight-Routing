"""
Empirical capability matrix and Bayesian capability estimation across domains and failure modes.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from bwr.models import TaskDomain, ModelProfile


class EmpiricalCapabilityMatrix:
    """
    Maintains historical empirical verification success rates across models and domains.
    Applies Bayesian smoothing: p_hat = (successes + alpha) / (attempts + alpha + beta)
    """

    def __init__(
        self,
        model_ids: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 2.0,
    ):
        self.model_ids = list(model_ids)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # statistics structure: {model_id: {domain_name: {"attempts": int, "successes": int, "total_cost": float, "residual_reductions": list[float]}}}
        self.stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for mid in self.model_ids:
            self.stats[mid] = {}
            for domain in TaskDomain:
                self.stats[mid][domain.value] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_cost": 0.0,
                    "residual_reductions": [],
                }

    def record_attempt(
        self,
        model_id: str,
        domain: TaskDomain | str,
        passed: bool,
        cost: float = 0.0,
        residual_reduction: float = 0.0,
    ) -> None:
        d_key = domain.value if isinstance(domain, TaskDomain) else str(domain)
        if model_id not in self.stats:
            self.stats[model_id] = {}
        if d_key not in self.stats[model_id]:
            self.stats[model_id][d_key] = {
                "attempts": 0,
                "successes": 0,
                "total_cost": 0.0,
                "residual_reductions": [],
            }

        entry = self.stats[model_id][d_key]
        entry["attempts"] += 1
        if passed:
            entry["successes"] += 1
        entry["total_cost"] += max(0.0, cost)
        entry["residual_reductions"].append(residual_reduction)

    def estimate_success_probability(
        self,
        model_id: str,
        domain: TaskDomain | str,
    ) -> float:
        """
        Bayesian smoothed probability of verified success: (s + alpha) / (n + alpha + beta)
        """
        d_key = domain.value if isinstance(domain, TaskDomain) else str(domain)
        entry = self.stats.get(model_id, {}).get(d_key, {"attempts": 0, "successes": 0})
        s = entry["successes"]
        n = entry["attempts"]
        return float((s + self.prior_alpha) / (n + self.prior_alpha + self.prior_beta))

    def estimate_average_cost(self, model_id: str, domain: TaskDomain | str, default_cost: float = 0.005) -> float:
        d_key = domain.value if isinstance(domain, TaskDomain) else str(domain)
        entry = self.stats.get(model_id, {}).get(d_key, {"attempts": 0, "total_cost": 0.0})
        if entry["attempts"] > 0 and entry["total_cost"] > 0:
            return float(entry["total_cost"] / entry["attempts"])
        return default_cost

    def calculate_efficiency(self, model_id: str, domain: TaskDomain | str, default_cost: float = 0.005) -> float:
        """
        eta_i = P(success) / E[K_i]
        """
        p_succ = self.estimate_success_probability(model_id, domain)
        avg_cost = self.estimate_average_cost(model_id, domain, default_cost=default_cost)
        return float(p_succ / max(1e-9, avg_cost))

    def calculate_marginal_efficiency(
        self,
        current_model_id: str,
        candidate_model_id: str,
        domain: TaskDomain | str,
    ) -> float:
        """
        E_ij = (Q_j - Q_i) / (K_j - K_i)
        """
        q_i = self.estimate_success_probability(current_model_id, domain)
        q_j = self.estimate_success_probability(candidate_model_id, domain)
        delta_q = q_j - q_i

        k_i = self.estimate_average_cost(current_model_id, domain)
        k_j = self.estimate_average_cost(candidate_model_id, domain)
        delta_k = k_j - k_i

        if delta_k <= 1e-9:
            return 1.0 if delta_q > 0 else 0.0
        return float(delta_q / delta_k)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "stats": self.stats,
        }

    def save_json(self, file_path: str | Path) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str | Path) -> EmpiricalCapabilityMatrix:
        p = Path(file_path)
        if not p.exists():
            return cls(model_ids=[])
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = cls(
            model_ids=list(data.get("stats", {}).keys()),
            prior_alpha=float(data.get("prior_alpha", 1.0)),
            prior_beta=float(data.get("prior_beta", 2.0)),
        )
        matrix.stats = data.get("stats", {})
        return matrix
