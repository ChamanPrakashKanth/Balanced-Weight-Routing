"""
Deterministic Structured Data and Hallucination Trap Verifier.
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from bwr.models import Task, ModelResponse, VerificationResult, VerificationFailure, TaskDomain
from bwr.verifier import BaseVerifier, extract_numeric_answer, VerifierRegistry


class StructuredVerifier(BaseVerifier):
    """
    Validates JSON schemas, structured outputs, constraints, and catches adversarial hallucination traps.
    """

    def verify(self, task: Task, response: ModelResponse) -> VerificationResult:
        if task.is_trap or task.domain == TaskDomain.TRAP:
            return self._verify_trap(task, response)
        return self._verify_json_structure(task, response)

    def _verify_json_structure(self, task: Task, response: ModelResponse) -> VerificationResult:
        text = response.text
        failures: List[VerificationFailure] = []

        # Extract JSON substring
        json_str = text
        if "```json" in text:
            m = re.findall(r"```json\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if m:
                json_str = m[-1]
        elif "```" in text:
            m = re.findall(r"```\n?(.*?)```", text, re.DOTALL)
            if m:
                json_str = m[-1]
        else:
            m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if m:
                json_str = m.group(1)

        try:
            parsed = json.loads(json_str.strip())
        except Exception as e:
            failures.append(
                VerificationFailure(
                    category="invalid_json",
                    message=f"Failed to parse valid JSON: {str(e)}",
                    expected="Valid JSON object/array",
                    observed=text[:100],
                    severity=1.0,
                )
            )
            return VerificationResult(passed=False, score=0.0, failures=failures)

        # Check required schema keys or structure
        required_keys = task.metadata.get("required_keys", [])
        for k in required_keys:
            if isinstance(parsed, dict) and k not in parsed:
                failures.append(
                    VerificationFailure(
                        category="missing_json_key",
                        message=f"Missing required key '{k}' in JSON output",
                        expected=f"Key '{k}' present",
                        observed=f"Keys: {list(parsed.keys())}",
                        severity=0.7,
                    )
                )

        # Check exact expected match if provided
        if task.expected_output is not None:
            if parsed == task.expected_output:
                return VerificationResult(passed=True, score=1.0, failures=[])
            else:
                failures.append(
                    VerificationFailure(
                        category="json_content_mismatch",
                        message="Parsed JSON does not match target specification",
                        expected=str(task.expected_output),
                        observed=str(parsed),
                        severity=0.6,
                    )
                )
                return VerificationResult(passed=False, score=0.4 if not failures else 0.2, failures=failures)

        passed = len(failures) == 0
        return VerificationResult(
            passed=passed,
            score=1.0 if passed else 0.3,
            failures=failures,
            details={"parsed_json": parsed},
        )

    def _verify_trap(self, task: Task, response: ModelResponse) -> VerificationResult:
        """
        Hallucination Stress Test Verifier:
        Detects if the model fell for subtle traps, impossible units, sign traps, or extraneous roots.
        """
        text = response.text
        failures: List[VerificationFailure] = []
        trap_type = task.metadata.get("trap_type", "subtle_assumption_error")
        forbidden_substrings = task.metadata.get("forbidden_hallucinations", [])
        expected = task.expected_output
        pred_num = extract_numeric_answer(text)

        # If expected is numeric, verify numeric correctness against expected and forbidden answers
        if isinstance(expected, (int, float)):
            if pred_num is None:
                failures.append(
                    VerificationFailure(
                        category="trap_missing_answer",
                        message="Could not extract numerical answer from response",
                        expected=str(expected),
                        observed="None",
                        severity=1.0,
                    )
                )
                return VerificationResult(passed=False, score=0.0, failures=failures)

            diff = abs(pred_num - float(expected))
            if diff <= 1e-4:
                return VerificationResult(passed=True, score=1.0, failures=[])

            # Check if observed number matches any forbidden numeric hallucination
            matched_forbidden = None
            for fb in forbidden_substrings:
                try:
                    if abs(pred_num - float(fb)) <= 1e-4:
                        matched_forbidden = fb
                        break
                except ValueError:
                    if str(fb).lower() in text.lower():
                        matched_forbidden = fb
                        break

            msg = f"Model fell for hallucination trap '{trap_type}'"
            if matched_forbidden:
                msg += f" with known fallacy value '{matched_forbidden}'"

            failures.append(
                VerificationFailure(
                    category=f"hallucination_trap_{trap_type}",
                    message=msg,
                    expected=str(expected),
                    observed=str(pred_num),
                    severity=1.0,
                )
            )
            return VerificationResult(passed=False, score=0.0, failures=failures)

        # Non-numeric checks
        for forbidden in forbidden_substrings:
            if str(forbidden).lower() in text.lower():
                failures.append(
                    VerificationFailure(
                        category=f"hallucination_trap_{trap_type}",
                        message=f"Model accepted flawed premise or fallacy: '{forbidden}'",
                        expected=str(expected or "Correct physical/mathematical deduction"),
                        observed=f"Hallucinated pattern detected: '{forbidden}'",
                        severity=1.0,
                    )
                )
                return VerificationResult(passed=False, score=0.0, failures=failures)

        return VerificationResult(passed=True, score=1.0, failures=[])


# Register verifiers
VerifierRegistry.register(TaskDomain.STRUCTURED, StructuredVerifier())
VerifierRegistry.register(TaskDomain.TRAP, StructuredVerifier())
VerifierRegistry.register(TaskDomain.MIXED, StructuredVerifier())
