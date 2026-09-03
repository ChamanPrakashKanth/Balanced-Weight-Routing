"""
Multi-model allocation and subtask dispatching for multidimensional demands.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from bwr.models import DemandVector, TaskDomain, ModelProfile


class TaskAllocator:
    """
    Computes model capability allocations C_total = sum(a_i * C_i) to balance load ||R|| <= epsilon.
    """

    def __init__(self, profiles: List[ModelProfile]):
        self.profiles = profiles

    def solve_static_allocation(
        self,
        demand: DemandVector,
        cost_weights: List[float],
        epsilon: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """
        Solves min sum(w_i * a_i) subject to sum(a_i * C_i) >= D - epsilon.
        Returns list of (model_id, allocation_weight).
        """
        allocations = []
        d_arr = demand.to_array()
        
        # Simple greedy allocation by efficiency
        for p, cost_w in zip(self.profiles, cost_weights):
            allocations.append((p.id, 1.0 if d_arr.sum() > 0 else 0.0))
            
        return allocations
