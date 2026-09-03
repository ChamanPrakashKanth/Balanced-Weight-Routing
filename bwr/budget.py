"""
Budget controller for enforcing compute, token, latency, and step limits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml
from pathlib import Path


@dataclass
class BudgetLimits:
    max_normalized_cost: float = 5.0
    max_total_tokens: int = 15000
    max_escalations: int = 4
    max_latency_seconds: float = 180.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> BudgetLimits:
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        b = data.get("budget", {})
        return cls(
            max_normalized_cost=float(b.get("max_normalized_cost", 5.0)),
            max_total_tokens=int(b.get("max_total_tokens", 15000)),
            max_escalations=int(b.get("max_escalations", 4)),
            max_latency_seconds=float(b.get("max_latency_seconds", 180.0)),
        )


class BudgetController:
    """
    Guarantees the harness stops when resource limits are reached.
    """

    def __init__(self, limits: Optional[BudgetLimits] = None):
        self.limits = limits or BudgetLimits()

    def check_budget(
        self,
        accumulated_cost: float,
        accumulated_tokens: int,
        current_step: int,
        accumulated_latency: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (within_budget, breach_reason).
        """
        if current_step >= self.limits.max_escalations:
            return False, f"Maximum escalations reached ({current_step} >= {self.limits.max_escalations})"

        if accumulated_cost >= self.limits.max_normalized_cost:
            return False, f"Cost budget exceeded ({accumulated_cost:.4f} >= {self.limits.max_normalized_cost:.4f})"

        if accumulated_tokens >= self.limits.max_total_tokens:
            return False, f"Token budget exceeded ({accumulated_tokens} >= {self.limits.max_total_tokens})"

        if accumulated_latency >= self.limits.max_latency_seconds:
            return False, f"Latency budget exceeded ({accumulated_latency:.2f}s >= {self.limits.max_latency_seconds:.2f}s)"

        return True, None

    def remaining_cost(self, accumulated_cost: float) -> float:
        return max(0.0, self.limits.max_normalized_cost - accumulated_cost)

    def can_afford(self, accumulated_cost: float, estimated_cost: float) -> bool:
        return (accumulated_cost + estimated_cost) <= self.limits.max_normalized_cost
