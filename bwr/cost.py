"""
Cost computation and tracking for local inference and API models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml
from pathlib import Path
from bwr.models import ModelResponse


@dataclass
class CostConfig:
    alpha: float = 0.00010   # Input token weight
    beta: float = 0.00025    # Output token weight
    gamma: float = 0.00100   # Wall-clock latency weight
    delta: float = 0.00000   # Compute / energy weight
    tier_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "model_1": 0.10,
        "model_2": 0.20,
        "model_3": 0.40,
        "model_4": 0.70,
        "model_5": 1.00,
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> CostConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        local = data.get("local_proxy", {})
        tiers = data.get("tier_multipliers", {})
        return cls(
            alpha=float(local.get("alpha", 0.00010)),
            beta=float(local.get("beta", 0.00025)),
            gamma=float(local.get("gamma", 0.00100)),
            delta=float(local.get("delta", 0.00000)),
            tier_multipliers=tiers or {
                "model_1": 0.10,
                "model_2": 0.20,
                "model_3": 0.40,
                "model_4": 0.70,
                "model_5": 1.00,
            }
        )


class CostTracker:
    """
    Computes normalized inference cost proxies and tracks cumulative costs.
    Formula: K_i = (alpha * T_in + beta * T_out + gamma * t_i + delta * E_i) * tier_multiplier
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def calculate_step_cost(
        self,
        response: ModelResponse,
        model_id: Optional[str] = None,
        energy: float = 0.0
    ) -> float:
        mid = model_id or response.model_id
        mult = self.config.tier_multipliers.get(mid, 1.0)
        
        base_cost = (
            self.config.alpha * response.prompt_tokens +
            self.config.beta * response.completion_tokens +
            self.config.gamma * response.latency_seconds +
            self.config.delta * energy
        )
        return float(base_cost * mult)

    @staticmethod
    def compute_cost_savings(router_cost: float, baseline_cost: float) -> float:
        """S_C = 1 - (K_router / K_baseline)"""
        if baseline_cost <= 1e-9:
            return 0.0
        return float(1.0 - (router_cost / baseline_cost))

    @staticmethod
    def compute_token_savings(router_tokens: int | float, baseline_tokens: int | float) -> float:
        """S_T = 1 - (T_router / T_baseline)"""
        if baseline_tokens <= 0:
            return 0.0
        return float(1.0 - (float(router_tokens) / float(baseline_tokens)))

    @staticmethod
    def compute_context_savings(full_tokens: int, residual_tokens: int) -> float:
        """S_context = 1 - (T_residual / T_full)"""
        if full_tokens <= 0:
            return 0.0
        return float(max(0.0, 1.0 - (float(residual_tokens) / float(full_tokens))))
