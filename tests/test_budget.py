"""
Unit tests for budget controller limits and violations.
"""

import pytest
from bwr.budget import BudgetController, BudgetLimits


def test_budget_controller():
    limits = BudgetLimits(
        max_normalized_cost=1.0,
        max_total_tokens=1000,
        max_escalations=3,
        max_latency_seconds=30.0,
    )
    controller = BudgetController(limits)

    # Within limits
    ok, reason = controller.check_budget(
        accumulated_cost=0.5,
        accumulated_tokens=400,
        current_step=1,
        accumulated_latency=5.0,
    )
    assert ok is True
    assert reason is None

    # Step limit reached
    ok, reason = controller.check_budget(
        accumulated_cost=0.5,
        accumulated_tokens=400,
        current_step=3,
        accumulated_latency=5.0,
    )
    assert ok is False
    assert "Maximum escalations" in reason

    # Cost limit reached
    ok, reason = controller.check_budget(
        accumulated_cost=1.2,
        accumulated_tokens=400,
        current_step=1,
        accumulated_latency=5.0,
    )
    assert ok is False
    assert "Cost budget" in reason
