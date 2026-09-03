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


FEATURE_DIMENSIONS = ["math", "reasoning", "code", "language", "mechanics", "planning", "trap"]


@dataclass
class CapabilityVector:
    """
    Empirical capability vector C_i = [c_math, c_reasoning, c_code, c_language, c_mechanics, c_planning, c_trap].
    """
    model_id: str
    math: float = 0.5
    reasoning: float = 0.5
    code: float = 0.5
    language: float = 0.5
    mechanics: float = 0.5
    planning: float = 0.5
    trap: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([
            self.math,
            self.reasoning,
            self.code,
            self.language,
            self.mechanics,
            self.planning,
            self.trap,
        ], dtype=float)

    @classmethod
    def from_array(cls, model_id: str, arr: np.ndarray) -> CapabilityVector:
        return cls(
            model_id=model_id,
            math=float(arr[0]),
            reasoning=float(arr[1]),
            code=float(arr[2]),
            language=float(arr[3]),
            mechanics=float(arr[4]),
            planning=float(arr[5]),
            trap=float(arr[6]) if len(arr) > 6 else 0.5,
        )


@dataclass
class DemandVector:
    """
    Multidimensional task load representation D = [d_math, d_reasoning, d_code, d_language, d_mechanics, d_planning, d_trap].
    """
    math: float = 0.0
    reasoning: float = 0.0
    code: float = 0.0
    language: float = 0.0
    mechanics: float = 0.0
    planning: float = 0.0
    trap: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.math,
            self.reasoning,
            self.code,
            self.language,
            self.mechanics,
            self.planning,
            self.trap,
        ], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> DemandVector:
        if len(arr) < 7:
            padded = np.zeros(7, dtype=float)
            padded[:len(arr)] = arr
            arr = padded
        return cls(
            math=float(arr[0]),
            reasoning=float(arr[1]),
            code=float(arr[2]),
            language=float(arr[3]),
            mechanics=float(arr[4]),
            planning=float(arr[5]),
            trap=float(arr[6]),
        )

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def deficit(self, capability: CapabilityVector | np.ndarray) -> np.ndarray:
        """
        Calculates component-wise remaining imbalance: R_i = max(0, D - C_i).
        """
        c_arr = capability.to_array() if isinstance(capability, CapabilityVector) else np.asarray(capability)
        return np.maximum(0.0, self.to_array() - c_arr)

    def weighted_imbalance_norm(
        self,
        capability: CapabilityVector | np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculates weighted imbalance norm: R_i = || W (D - C_i)_+ ||_2.
        """
        r = self.deficit(capability)
        w = weights if weights is not None else np.ones_like(r)
        return float(np.sqrt(np.sum(w * (r ** 2))))


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
                self.demand = DemandVector(code=self.difficulty, reasoning=max(0.3, self.difficulty * 0.8), language=0.2)
            elif self.domain == TaskDomain.MATH:
                self.demand = DemandVector(math=self.difficulty, reasoning=max(0.4, self.difficulty * 0.9), language=0.2)
            elif self.domain == TaskDomain.MECHANICS:
                self.demand = DemandVector(mechanics=self.difficulty, math=self.difficulty * 0.7, reasoning=self.difficulty * 0.7, language=0.2)
            elif self.domain == TaskDomain.STRUCTURED:
                self.demand = DemandVector(language=self.difficulty, reasoning=self.difficulty * 0.5, code=0.2)
            elif self.domain == TaskDomain.TRAP:
                self.demand = DemandVector(trap=self.difficulty, reasoning=0.9, language=0.7)
            elif self.domain == TaskDomain.MIXED:
                self.demand = DemandVector(math=0.5, code=0.6, reasoning=0.7, mechanics=0.5, language=0.4)


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
    current_demand: DemandVector = field(default_factory=DemandVector)
    observed_residual_vector: Optional[DemandVector] = None
    success: bool = False
    final_response: Optional[ModelResponse] = None
    final_verification: Optional[VerificationResult] = None
    termination_reason: str = "in_progress"

    def __post_init__(self):
        if np.all(self.current_demand.to_array() == 0.0):
            self.current_demand = DemandVector.from_array(self.task.demand.to_array().copy())
