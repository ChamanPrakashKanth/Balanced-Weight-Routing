"""
Unit tests for domain verifiers: Code, Math, Mechanics, Structured, and Hallucination Traps.
"""

import pytest
from bwr.models import Task, TaskDomain, ModelResponse
from verifiers.code_verifier import CodeVerifier
from verifiers.math_verifier import MathVerifier
from verifiers.mechanics_verifier import MechanicsVerifier
from verifiers.structured_verifier import StructuredVerifier


def test_code_verifier_success():
    verifier = CodeVerifier()
    task = Task(
        id="code_t1",
        title="Multiply",
        prompt="Write a function `multiply(a, b)`.",
        domain=TaskDomain.CODE,
        test_cases=[
            {"function_name": "multiply", "inputs": [3, 4], "expected": 12},
            {"function_name": "multiply", "inputs": [0, 5], "expected": 0},
        ],
    )
    resp = ModelResponse(
        model_id="m1",
        model_name="M1",
        text="```python\ndef multiply(a, b):\n    return a * b\n```",
    )
    result = verifier.verify(task, resp)
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.failures) == 0


def test_code_verifier_syntax_error():
    verifier = CodeVerifier()
    task = Task(
        id="code_t2",
        title="Broken Syntax",
        prompt="Write a function `foo()`.",
        domain=TaskDomain.CODE,
    )
    resp = ModelResponse(
        model_id="m1",
        model_name="M1",
        text="```python\ndef foo(:\n  return 1\n```",
    )
    result = verifier.verify(task, resp)
    assert result.passed is False
    assert result.score == 0.0
    assert result.failures[0].category == "syntax_error"


def test_math_verifier():
    verifier = MathVerifier(numeric_tolerance=1e-4)
    task = Task(
        id="math_t1",
        title="Sum",
        prompt="Calculate 21 / 2 * 102",
        domain=TaskDomain.MATH,
        expected_output=1071.0,
    )

    # Correct response
    resp_ok = ModelResponse(model_id="m1", model_name="M1", text="Calculation gives Answer: 1071.0")
    res_ok = verifier.verify(task, resp_ok)
    assert res_ok.passed is True
    assert res_ok.score == 1.0

    # Wrong response
    resp_bad = ModelResponse(model_id="m1", model_name="M1", text="Answer: 1000.0")
    res_bad = verifier.verify(task, resp_bad)
    assert res_bad.passed is False
    assert res_bad.failures[0].category == "numerical_mismatch"


def test_mechanics_rotating_mass_verifier():
    verifier = MechanicsVerifier(tolerance=1e-3)
    task = Task(
        id="mech_t1",
        title="Rotating Mass",
        prompt="Calculate balancing mass",
        domain=TaskDomain.MECHANICS,
        expected_output=2.4083,
        metadata={"mechanics_type": "rotating_mass_balance"},
    )

    resp_ok = ModelResponse(model_id="m1", model_name="M1", text="Resultant unbalance = 0.60208, Answer: 2.4083")
    res_ok = verifier.verify(task, resp_ok)
    assert res_ok.passed is True

    resp_bad = ModelResponse(model_id="m1", model_name="M1", text="Answer: 3.4000")
    res_bad = verifier.verify(task, resp_bad)
    assert res_bad.passed is False
    assert res_bad.failures[0].category == "dynamic_unbalance"


def test_trap_verifier():
    verifier = StructuredVerifier()
    task = Task(
        id="trap_t1",
        title="Buoyancy Trap",
        prompt="Calculate apparent underwater weight",
        domain=TaskDomain.TRAP,
        is_trap=True,
        expected_output=16.677,
        metadata={
            "trap_type": "buoyancy_omission",
            "forbidden_hallucinations": ["26.487", "weight is 26.5"],
        },
    )

    # Hallucinated answer
    resp_hallucinated = ModelResponse(
        model_id="m1",
        model_name="M1",
        text="Mass is constant so weight = 26.487 N. Answer: 26.487",
    )
    res_hallucinated = verifier.verify(task, resp_hallucinated)
    assert res_hallucinated.passed is False
    assert "hallucination_trap" in res_hallucinated.failures[0].category

    # Correct deduction
    resp_correct = ModelResponse(
        model_id="m5",
        model_name="M5",
        text="Buoyancy is 9.81 N. Apparent weight = 26.487 - 9.81 = 16.677 N. Answer: 16.677",
    )
    res_correct = verifier.verify(task, resp_correct)
    assert res_correct.passed is True
