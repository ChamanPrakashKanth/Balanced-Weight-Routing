"""
Balanced Weight Routing (BWR) and Verified Residual Routing (VRR).

A research framework for cost-efficient, verifiably reliable routing across heterogeneous LLM ladders.
"""

__version__ = "0.1.0"

from bwr.models import (
    Task,
    TaskDomain,
    ModelResponse,
    VerificationResult,
    VerificationFailure,
    ResidualContext,
    RoutingDecision,
    RoutingState,
    RoutingHistoryItem,
)
from bwr.cost import CostTracker, CostConfig
from bwr.budget import BudgetController, BudgetLimits
from bwr.residual import ResidualExtractor
from bwr.capability import EmpiricalCapabilityMatrix
from bwr.router import (
    BaseRouter,
    StrongestOnlyRouter,
    SmallestOnlyRouter,
    FixedCascadeRouter,
    ConfidenceRouter,
    VerifiedFullTaskRouter,
    VerifiedResidualRouter,
    BalancedWeightRouter,
    create_router,
)
from bwr.orchestrator import Orchestrator

__all__ = [
    "Task",
    "TaskDomain",
    "ModelResponse",
    "VerificationResult",
    "VerificationFailure",
    "ResidualContext",
    "RoutingDecision",
    "RoutingState",
    "RoutingHistoryItem",
    "CostTracker",
    "CostConfig",
    "BudgetController",
    "BudgetLimits",
    "ResidualExtractor",
    "EmpiricalCapabilityMatrix",
    "BaseRouter",
    "StrongestOnlyRouter",
    "SmallestOnlyRouter",
    "FixedCascadeRouter",
    "ConfidenceRouter",
    "VerifiedFullTaskRouter",
    "VerifiedResidualRouter",
    "BalancedWeightRouter",
    "create_router",
    "Orchestrator",
]
