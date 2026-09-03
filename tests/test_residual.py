"""
Unit tests for residual context extraction and compression ratio calculation.
"""

import pytest
from bwr.models import Task, TaskDomain, ModelResponse, VerificationResult, VerificationFailure
from bwr.residual import ResidualExtractor


def test_residual_extraction():
    task = Task(
        id="code_test_01",
        title="Compute Factorial",
        prompt="Write a function `factorial(n: int) -> int` that calculates n! recursively or iteratively.",
        domain=TaskDomain.CODE,
    )
    attempt = ModelResponse(
        model_id="model_1",
        model_name="Model 1",
        text="```python\ndef factorial(n):\n    return 0 # buggy\n```",
    )
    verification = VerificationResult(
        passed=False,
        score=0.4,
        failures=[
            VerificationFailure(
                category="unit_test_failure",
                message="factorial(3) returned 0",
                expected="6",
                observed="0",
                test_name="test_case_3",
            )
        ],
    )

    extractor = ResidualExtractor()
    res_ctx = extractor.extract_residual(task, attempt, verification)

    assert "VERIFICATION FAILURES" in res_ctx.residual_prompt
    assert "factorial(3) returned 0" in res_ctx.residual_prompt
    assert "Expected: 6" in res_ctx.residual_prompt
    assert "Observed: 0" in res_ctx.residual_prompt
    assert res_ctx.residual_context_tokens > 0
    assert 0.0 <= res_ctx.context_reduction_ratio <= 1.0
