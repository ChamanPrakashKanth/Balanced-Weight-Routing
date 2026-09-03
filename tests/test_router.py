"""
Unit tests for all routing strategies (Strongest, Cascade, Confidence, VRR, BWR).
"""

import pytest
from bwr.models import Task, TaskDomain, ModelProfile, RoutingState, ModelResponse, VerificationResult
from bwr.capability import EmpiricalCapabilityMatrix
from bwr.router import (
    StrongestOnlyRouter,
    SmallestOnlyRouter,
    FixedCascadeRouter,
    ConfidenceRouter,
    VerifiedFullTaskRouter,
    VerifiedResidualRouter,
    BalancedWeightRouter,
)


@pytest.fixture
def sample_profiles():
    return [
        ModelProfile(id="model_1", name="M1", ollama_name="qwen:0.5b", cost_tier=1),
        ModelProfile(id="model_2", name="M2", ollama_name="llama:1b", cost_tier=2),
        ModelProfile(id="model_3", name="M3", ollama_name="qwen:3b", cost_tier=3),
        ModelProfile(id="model_4", name="M4", ollama_name="mistral:7b", cost_tier=4),
        ModelProfile(id="model_5", name="M5", ollama_name="qwen:14b", cost_tier=5),
    ]


@pytest.fixture
def sample_task():
    return Task(
        id="task_01",
        title="Sample Task",
        prompt="Write hello world",
        domain=TaskDomain.CODE,
    )


def test_strongest_router(sample_profiles, sample_task):
    router = StrongestOnlyRouter(sample_profiles)
    state = RoutingState(task=sample_task)

    dec = router.route_step(state)
    assert dec.selected_model_id == "model_5"
    assert dec.is_residual is False


def test_fixed_cascade_router(sample_profiles, sample_task):
    router = FixedCascadeRouter(sample_profiles)
    state = RoutingState(task=sample_task)

    # Step 0 -> M1
    dec0 = router.route_step(state)
    assert dec0.selected_model_id == "model_1"

    # Step 1 -> M2
    state.current_step = 1
    dec1 = router.route_step(state)
    assert dec1.selected_model_id == "model_2"


def test_confidence_router(sample_profiles, sample_task):
    router = ConfidenceRouter(sample_profiles, confidence_threshold=0.85)
    state = RoutingState(task=sample_task)

    # Low confidence response: continues
    resp_low = ModelResponse(model_id="model_1", model_name="M1", text="text", self_reported_confidence=0.50)
    term, reason = router.should_terminate(state, None, resp_low)
    assert term is False

    # High confidence response: terminates even if wrong
    resp_high = ModelResponse(model_id="model_1", model_name="M1", text="text", self_reported_confidence=0.95)
    term, reason = router.should_terminate(state, None, resp_high)
    assert term is True
    assert "accepted_by_confidence" in reason


def test_vrr_router(sample_profiles, sample_task):
    router = VerifiedResidualRouter(sample_profiles)
    state = RoutingState(task=sample_task)

    dec0 = router.route_step(state)
    assert dec0.selected_model_id == "model_1"
    assert dec0.is_residual is False

    state.current_step = 1
    dec1 = router.route_step(state)
    assert dec1.selected_model_id == "model_2"
    assert dec1.is_residual is True


def test_bwr_router_with_matrix(sample_profiles, sample_task):
    matrix = EmpiricalCapabilityMatrix(model_ids=[p.id for p in sample_profiles])
    # Seed historical statistics: model_1 is cheap but low accuracy, model_4 is high accuracy
    matrix.record_attempt("model_1", TaskDomain.CODE, passed=False, cost=0.001)
    matrix.record_attempt("model_4", TaskDomain.CODE, passed=True, cost=0.007)

    router = BalancedWeightRouter(sample_profiles, capability_matrix=matrix, allow_skipping=True)
    state = RoutingState(task=sample_task)

    dec = router.route_step(state)
    assert dec.selected_model_id in ["model_1", "model_4"]


def test_feature_vector_bwr_router(sample_profiles):
    from bwr.capability import FeatureVectorCapabilityMatrix
    from bwr.models import CapabilityVector, DemandVector
    from bwr.router import FeatureVectorBWRRouter

    fv_matrix = FeatureVectorCapabilityMatrix(model_ids=[p.id for p in sample_profiles])
    # Set M1 weak at math but M4 strong at math
    fv_matrix.capabilities["model_1"] = CapabilityVector(model_id="model_1", math=0.2, code=0.8)
    fv_matrix.capabilities["model_4"] = CapabilityVector(model_id="model_4", math=0.9, code=0.8)

    # Math heavy task should route towards M4 over M1 despite M1 being cheaper
    math_task = Task(
        id="math_heavy",
        title="Heavy Math",
        prompt="Solve integral",
        domain=TaskDomain.MATH,
        demand=DemandVector(math=0.9, reasoning=0.8),
    )
    router = FeatureVectorBWRRouter(sample_profiles, feature_matrix=fv_matrix, lambda_r=10.0, lambda_k=1.0)
    state = RoutingState(task=math_task)

    dec = router.route_step(state)
    assert dec.selected_model_id in ["model_4", "model_5"]
