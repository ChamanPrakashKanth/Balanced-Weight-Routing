"""
Task definitions, domain classification, and multidimensional demand representation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from bwr.models import Task, TaskDomain, DemandVector


class TaskLoader:
    """
    Utility for loading and validating benchmark tasks.
    """

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> Task:
        domain_str = data.get("domain", "code").lower()
        try:
            domain = TaskDomain(domain_str)
        except ValueError:
            domain = TaskDomain.CODE

        demand_dict = data.get("demand", {})
        demand = DemandVector(
            code=float(demand_dict.get("code", 0.0)),
            math=float(demand_dict.get("math", 0.0)),
            mechanics=float(demand_dict.get("mechanics", 0.0)),
            structured=float(demand_dict.get("structured", 0.0)),
            trap=float(demand_dict.get("trap", 0.0)),
        )

        return Task(
            id=str(data.get("id", "task_unknown")),
            title=str(data.get("title", "Untitled Task")),
            prompt=str(data.get("prompt", "")),
            domain=domain,
            difficulty=float(data.get("difficulty", 0.5)),
            demand=demand,
            reference_solution=data.get("reference_solution"),
            expected_output=data.get("expected_output"),
            test_cases=data.get("test_cases", []),
            metadata=data.get("metadata", {}),
            is_trap=bool(data.get("is_trap", False)),
        )

    @staticmethod
    def load_from_json(file_path: str | Path) -> List[Task]:
        p = Path(file_path)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return [TaskLoader.load_from_dict(item) for item in raw]
        elif isinstance(raw, dict) and "tasks" in raw:
            return [TaskLoader.load_from_dict(item) for item in raw["tasks"]]
        elif isinstance(raw, dict):
            return [TaskLoader.load_from_dict(raw)]
        return []


def calculate_residual_demand(
    demand: DemandVector,
    allocated_capabilities: List[DemandVector]
) -> DemandVector:
    """
    R = D - sum(a_i * C_i)
    """
    d_arr = demand.to_array()
    c_arr = sum((c.to_array() for c in allocated_capabilities), start=np.zeros(5))
    r_arr = np.maximum(0.0, d_arr - c_arr)
    return DemandVector.from_array(r_arr)
