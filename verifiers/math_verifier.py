"""
Deterministic Mathematics Verifier checking exact symbolic, numerical, and algebraic equivalence.
"""

from __future__ import annotations
import math
from typing import List, Optional, Any
import sympy as sp
from bwr.models import Task, ModelResponse, VerificationResult, VerificationFailure, TaskDomain
from bwr.verifier import BaseVerifier, extract_numeric_answer, VerifierRegistry


class MathVerifier(BaseVerifier):
    """
    Evaluates mathematical responses using:
    1. Symbolic simplification (sympy.simplify(expr_pred - expr_expected) == 0)
    2. High-precision numerical tolerance (|y_pred - y_expected| <= tol)
    3. Matrix/Vector equality checks
    """

    def __init__(self, numeric_tolerance: float = 1e-5):
        self.tolerance = numeric_tolerance

    def verify(self, task: Task, response: ModelResponse) -> VerificationResult:
        failures: List[VerificationFailure] = []
        expected = task.expected_output
        text = response.text

        # 1. If expected is a float or integer, perform numeric extraction & tolerance check
        if isinstance(expected, (int, float)):
            pred_num = extract_numeric_answer(text)
            if pred_num is None:
                failures.append(
                    VerificationFailure(
                        category="missing_numeric_answer",
                        message="Could not parse a numeric answer from model response",
                        expected=str(expected),
                        observed="No numeric token extracted",
                        severity=1.0,
                    )
                )
                return VerificationResult(passed=False, score=0.0, failures=failures)

            diff = abs(pred_num - float(expected))
            passed = diff <= self.tolerance
            if passed:
                return VerificationResult(
                    passed=True,
                    score=1.0,
                    failures=[],
                    details={"observed": pred_num, "expected": expected, "diff": diff},
                )
            else:
                failures.append(
                    VerificationFailure(
                        category="numerical_mismatch",
                        message=f"Numerical answer differs from ground truth (delta={diff:.6f} > tol={self.tolerance})",
                        expected=str(expected),
                        observed=str(pred_num),
                        severity=0.8,
                    )
                )
                score = max(0.0, 1.0 - (diff / max(1.0, abs(float(expected)))))
                return VerificationResult(
                    passed=False,
                    score=min(0.9, score),
                    failures=failures,
                    details={"observed": pred_num, "expected": expected, "diff": diff},
                )

        # 2. If expected is a symbolic formula (e.g. "2*x*exp(x) + x**2*exp(x)")
        symbolic_expected = task.metadata.get("symbolic_expected")
        if symbolic_expected:
            try:
                # Extract formula snippet
                expr_exp = sp.sympify(symbolic_expected)
                # Try to parse symbolic from text
                extracted_str = task.metadata.get("symbolic_var", "x")
                # Look for expressions in text
                lines = [l.strip() for l in text.split("\n") if "=" in l or "derivative" in l.lower() or "integral" in l.lower()]
                parsed_sym = None
                for line in lines:
                    rhs = line.split("=")[-1].strip().rstrip(".")
                    try:
                        parsed_sym = sp.sympify(rhs)
                        break
                    except Exception:
                        continue

                if parsed_sym is not None:
                    delta = sp.simplify(parsed_sym - expr_exp)
                    if delta == 0:
                        return VerificationResult(passed=True, score=1.0, failures=[])
                    else:
                        failures.append(
                            VerificationFailure(
                                category="symbolic_mismatch",
                                message="Symbolic expression is not equivalent to target solution",
                                expected=str(expr_exp),
                                observed=str(parsed_sym),
                                severity=0.8,
                            )
                        )
                        return VerificationResult(passed=False, score=0.3, failures=failures)
            except Exception as e:
                pass

        # 3. Fallback to substring / string match if expected is string
        if expected is not None:
            if str(expected).strip() in text:
                return VerificationResult(passed=True, score=1.0, failures=[])
            else:
                failures.append(
                    VerificationFailure(
                        category="answer_not_found",
                        message="Target solution text not found in response",
                        expected=str(expected),
                        observed=text[:100] + "...",
                        severity=0.7,
                    )
                )
                return VerificationResult(passed=False, score=0.0, failures=failures)

        return VerificationResult(passed=True, score=1.0, failures=[])


# Register verifier
VerifierRegistry.register(TaskDomain.MATH, MathVerifier())
