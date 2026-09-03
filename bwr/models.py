"""
Data models and type definitions for Balanced Weight Routing (BWR) and VRR.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class TaskDomain(str, Enum):
    CODE = "code"
    MATH = "math"
    MECHANICS = "mechanics"
    STRUCTURED = "structured"
    MIXED = "mixed"
    TRAP = "trap"


@dataclass
class DemandVector:
    """
    Multidimensional task load representation D = [d_code, d_math, d_mech, d_struct, d_trap].
    """
    code: float = 0.0
    math: float = 0.0
    mechanics: float = 0.0
    structured: float = 0.0
    trap: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.code, self.math, self.mechanics, self.structured, self.trap], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> DemandVector:
        return cls(
            code=float(arr[0]),
            math=float(arr[1]),
            mechanics=float(arr[2]),
            structured=float(arr[3]),
            trap=float(arr[4]) if len(arr) > 4 else 0.0,
        )

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))


@dataclass
class Task:
    """
    Task specification for deterministic and verified evaluation.
    """
    id: str
    title: str
    prompt: str
    domain: TaskDomain
    difficulty: float = 0.5  # Scalar difficulty in [0, 1]
    demand: DemandVector = field(default_factory=DemandVector)
    reference_solution: Optional[str] = None
    expected_output: Optional[Any] = None
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_trap: bool = False  # Hallucination stress test flag

    def __post_init__(self):
        # If demand is empty, initialize based on primary domain and difficulty
        if np.all(self.demand.to_array() == 0.0):
            if self.domain == TaskDomain.CODE:
                self.demand.code = self.difficulty
            elif self.domain == TaskDomain.MATH:
                self.demand.math = self.difficulty
            elif self.domain == TaskDomain.MECHANICS:
                self.demand.mechanics = self.difficulty
            elif self.domain == TaskDomain.STRUCTURED:
                self.demand.structured = self.difficulty
            elif self.domain == TaskDomain.TRAP:
                self.demand.trap = self.difficulty


@dataclass
class ModelProfile:
    """
    Metadata and configuration profile for a model in the ladder.
    """
    id: str
    name: str
    ollama_name: str
    cost_tier: int = 1
    context_length: int = 4096
    default_temperature: float = 0.0
    mock_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """
    Standardized response from a local Ollama model or simulated model.
    """
    model_id: str
    model_name: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    self_reported_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class VerificationFailure:
    """
    Explicit representation of an external verification failure.
    """
    category: str
    message: str
    expected: Optional[str] = None
    observed: Optional[str] = None
    severity: float = 1.0  # [0.0, 1.0]
    test_name: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class VerificationResult:
    """
    Outcome from an external verifier.
    """
    passed: bool
    score: float  # [0.0, 1.0], where 1.0 is full success
    failures: List[VerificationFailure] = field(default_factory=list)
    execution_output: Optional[str] = None
    latency_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def residual_error(self) -> float:
        """R_obs = 1 - V(x, y)"""
        return max(0.0, 1.0 - self.score)


@dataclass
class ResidualContext:
    """
    Minimal isolated residual context extracted after a verification failure.
    """
    residual_prompt: str
    context_reduction_ratio: float = 0.0  # S_context = 1 - T_res / T_full
    full_context_tokens: int = 0
    residual_context_tokens: int = 0
    isolated_failures: List[VerificationFailure] = field(default_factory=list)
    attempted_solution: Optional[str] = None


@dataclass
class RoutingDecision:
    """
    Router decision output.
    """
    selected_model_id: str
    reason: str
    is_residual: bool = False
    subtask: Optional[str] = None
    allocated_budget: Optional[float] = None
    expected_efficiency: float = 0.0
    estimated_capability: float = 0.0
    skip_jump: int = 0  # Number of models skipped in ladder (e.g. M1 -> M4 skips 2)


@dataclass
class RoutingHistoryItem:
    """
    Immutable telemetry record for a single model execution step.
    """
    step: int
    model_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    cost: float
    verification_score: float
    passed: bool
    failures: List[Dict[str, Any]]
    is_residual: bool = False
    residual_context_tokens: Optional[int] = None
    context_reduction_ratio: Optional[float] = None
    self_reported_confidence: Optional[float] = None


@dataclass
class RoutingState:
    """
    End-to-end execution state across routing steps.
    """
    task: Task
    current_step: int = 0
    accumulated_cost: float = 0.0
    accumulated_tokens: int = 0
    accumulated_latency: float = 0.0
    history: List[RoutingHistoryItem] = field(default_factory=list)
    current_residual: float = 1.0
    success: bool = False
    final_response: Optional[ModelResponse] = None
    final_verification: Optional[VerificationResult] = None
    termination_reason: str = "in_progress"
