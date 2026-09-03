"""
Unit tests for cost proxy, savings, and tier multipliers.
"""

import pytest
from bwr.models import ModelResponse
from bwr.cost import CostTracker, CostConfig


def test_cost_calculation():
    config = CostConfig(alpha=0.0001, beta=0.0002, gamma=0.001)
    tracker = CostTracker(config)

    resp = ModelResponse(
        model_id="model_1",
        model_name="Model 1",
        text="test output",
        prompt_tokens=100,
        completion_tokens=50,
        latency_seconds=1.0,
    )
    # base = 100*0.0001 + 50*0.0002 + 1.0*0.001 = 0.01 + 0.01 + 0.001 = 0.021
    # model_1 tier multiplier is 0.10 => 0.0021
    cost = tracker.calculate_step_cost(resp)
    assert abs(cost - 0.0021) < 1e-6


def test_savings_metrics():
    # Cost savings: 1 - (70 / 100) = 0.30 (30%)
    cost_sav = CostTracker.compute_cost_savings(70.0, 100.0)
    assert abs(cost_sav - 0.30) < 1e-6

    # Token savings: 1 - (500 / 1000) = 0.50 (50%)
    tok_sav = CostTracker.compute_token_savings(500, 1000)
    assert abs(tok_sav - 0.50) < 1e-6

    # Context savings: 1 - (200 / 800) = 0.75 (75%)
    ctx_sav = CostTracker.compute_context_savings(800, 200)
    assert abs(ctx_sav - 0.75) < 1e-6
