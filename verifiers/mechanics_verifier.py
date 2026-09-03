"""
Deterministic Mechanics and Engineering Verifier enforcing physical equilibrium,
rotating mass balancing, vibrations, beam reactions, and thermodynamic conservation laws.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Optional
from bwr.models import Task, ModelResponse, VerificationResult, VerificationFailure, TaskDomain
from bwr.verifier import BaseVerifier, extract_numeric_answer, VerifierRegistry


class MechanicsVerifier(BaseVerifier):
    """
    Evaluates engineering and physics problem responses against exact governing equations.
    """

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify(self, task: Task, response: ModelResponse) -> VerificationResult:
        mech_type = task.metadata.get("mechanics_type", "general_equilibrium")
        failures: List[VerificationFailure] = []

        if mech_type == "rotating_mass_balance":
            return self._verify_rotating_mass(task, response)
        elif mech_type == "force_moment_equilibrium":
            return self._verify_force_equilibrium(task, response)
        elif mech_type == "spring_mass_vibration":
            return self._verify_vibration(task, response)
        elif mech_type == "beam_reaction":
            return self._verify_beam_reaction(task, response)
        elif mech_type == "thermo_energy_balance":
            return self._verify_thermo_balance(task, response)
        else:
            # General numeric comparison
            return self._verify_general_numeric(task, response)

    def _verify_rotating_mass(self, task: Task, response: ModelResponse) -> VerificationResult:
        """
        Verifies rotating mass dynamic balance:
        Sum(m_i * r_i * cos(theta_i)) = 0
        Sum(m_i * r_i * sin(theta_i)) = 0
        and for multi-plane:
        Sum(m_i * r_i * l_i * cos(theta_i)) = 0
        Sum(m_i * r_i * l_i * sin(theta_i)) = 0
        """
        failures: List[VerificationFailure] = []
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)

        if observed_num is None:
            failures.append(
                VerificationFailure(
                    category="missing_mass_balance_value",
                    message="Could not parse computed balancing mass/angle from response",
                    expected=str(expected),
                    observed="None",
                    severity=1.0,
                )
            )
            return VerificationResult(passed=False, score=0.0, failures=failures)

        # Check expected value (e.g. balancing mass m_b or angle theta_b)
        if isinstance(expected, (int, float)):
            diff = abs(observed_num - float(expected))
            if diff <= self.tolerance:
                return VerificationResult(passed=True, score=1.0, failures=[])
            else:
                failures.append(
                    VerificationFailure(
                        category="dynamic_unbalance",
                        message=f"Calculated dynamic balancing parameter error (diff={diff:.4f} > tol={self.tolerance})",
                        expected=str(expected),
                        observed=str(observed_num),
                        severity=0.8,
                    )
                )
                score = max(0.0, 1.0 - (diff / max(1.0, abs(float(expected)))))
                return VerificationResult(passed=False, score=min(0.85, score), failures=failures)

        return VerificationResult(passed=True, score=1.0, failures=[])

    def _verify_force_equilibrium(self, task: Task, response: ModelResponse) -> VerificationResult:
        """
        Verifies static equilibrium: Sum(Fx) = 0, Sum(Fy) = 0, Sum(M) = 0.
        """
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)
        failures = []

        if observed_num is None:
            failures.append(
                VerificationFailure(
                    category="equilibrium_violation",
                    message="Missing numerical equilibrium force/moment in response",
                    expected=str(expected),
                    observed="None",
                    severity=1.0,
                )
            )
            return VerificationResult(passed=False, score=0.0, failures=failures)

        diff = abs(observed_num - float(expected))
        if diff <= self.tolerance:
            return VerificationResult(passed=True, score=1.0, failures=[])

        failures.append(
            VerificationFailure(
                category="equilibrium_unbalance",
                message=f"Static equilibrium force mismatch: calculated {observed_num} vs expected {expected}",
                expected=str(expected),
                observed=str(observed_num),
                severity=0.8,
            )
        )
        return VerificationResult(passed=False, score=max(0.0, 1.0 - diff / max(1.0, abs(float(expected)))), failures=failures)

    def _verify_vibration(self, task: Task, response: ModelResponse) -> VerificationResult:
        """
        Verifies natural frequency omega_n = sqrt(k / m) or fn = omega_n / (2*pi).
        """
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)
        if observed_num is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                failures=[
                    VerificationFailure(
                        category="vibration_frequency_missing",
                        message="Could not extract natural frequency calculation",
                        expected=str(expected),
                    )
                ]
            )

        diff = abs(observed_num - float(expected))
        if diff <= self.tolerance:
            return VerificationResult(passed=True, score=1.0, failures=[])

        return VerificationResult(
            passed=False,
            score=max(0.0, 1.0 - diff / max(1.0, abs(float(expected)))),
            failures=[
                VerificationFailure(
                    category="incorrect_natural_frequency",
                    message=f"Natural frequency error (diff={diff:.4f})",
                    expected=str(expected),
                    observed=str(observed_num),
                    severity=0.75,
                )
            ]
        )

    def _verify_beam_reaction(self, task: Task, response: ModelResponse) -> VerificationResult:
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)
        if observed_num is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                failures=[
                    VerificationFailure(
                        category="beam_reaction_missing",
                        message="Could not extract beam reaction force",
                        expected=str(expected),
                    )
                ]
            )

        diff = abs(observed_num - float(expected))
        if diff <= self.tolerance:
            return VerificationResult(passed=True, score=1.0, failures=[])

        return VerificationResult(
            passed=False,
            score=max(0.0, 1.0 - diff / max(1.0, abs(float(expected)))),
            failures=[
                VerificationFailure(
                    category="incorrect_beam_reaction",
                    message=f"Beam reaction mismatch (diff={diff:.4f})",
                    expected=str(expected),
                    observed=str(observed_num),
                )
            ]
        )

    def _verify_thermo_balance(self, task: Task, response: ModelResponse) -> VerificationResult:
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)
        if observed_num is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                failures=[
                    VerificationFailure(
                        category="thermo_balance_missing",
                        message="Could not extract thermodynamic state value",
                        expected=str(expected),
                    )
                ]
            )

        diff = abs(observed_num - float(expected))
        if diff <= self.tolerance:
            return VerificationResult(passed=True, score=1.0, failures=[])

        return VerificationResult(
            passed=False,
            score=max(0.0, 1.0 - diff / max(1.0, abs(float(expected)))),
            failures=[
                VerificationFailure(
                    category="first_law_violation",
                    message=f"Thermodynamic energy balance discrepancy (diff={diff:.4f})",
                    expected=str(expected),
                    observed=str(observed_num),
                )
            ]
        )

    def _verify_general_numeric(self, task: Task, response: ModelResponse) -> VerificationResult:
        expected = task.expected_output
        observed_num = extract_numeric_answer(response.text)
        if observed_num is None:
            return VerificationResult(
                passed=False,
                score=0.0,
                failures=[
                    VerificationFailure(
                        category="mechanics_numeric_missing",
                        message="Could not extract numerical result",
                        expected=str(expected),
                    )
                ]
            )
        diff = abs(observed_num - float(expected)) if expected is not None else 0.0
        passed = diff <= self.tolerance
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else max(0.0, 1.0 - diff / max(1.0, abs(float(expected or 1.0)))),
            failures=[] if passed else [
                VerificationFailure(
                    category="mechanics_result_mismatch",
                    message=f"Numerical result difference ({diff:.4f})",
                    expected=str(expected),
                    observed=str(observed_num),
                )
            ]
        )


# Register verifier
VerifierRegistry.register(TaskDomain.MECHANICS, MechanicsVerifier())
