"""
Deterministic Code Verifier executing sandboxed Python code with unit tests and AST checks.
"""

from __future__ import annotations
import ast
import sys
import io
import contextlib
import traceback
from typing import Dict, Any, List
from bwr.models import Task, ModelResponse, VerificationResult, VerificationFailure, TaskDomain
from bwr.verifier import BaseVerifier, extract_code_block, VerifierRegistry


class CodeVerifier(BaseVerifier):
    """
    Evaluates Python code solutions:
    1. Syntax / AST Parse (0.25)
    2. Unit Tests Execution (0.50)
    3. Runtime Error-free (0.15)
    4. Deterministic Output Match (0.10)
    """

    def __init__(self, execution_timeout: float = 3.0):
        self.execution_timeout = execution_timeout

    def verify(self, task: Task, response: ModelResponse) -> VerificationResult:
        code = extract_code_block(response.text, default_lang="python")
        failures: List[VerificationFailure] = []
        score = 0.0

        # Step 1: AST Syntax Validation
        try:
            tree = ast.parse(code)
            score += 0.25
        except SyntaxError as e:
            failures.append(
                VerificationFailure(
                    category="syntax_error",
                    message=f"SyntaxError on line {e.lineno}: {e.msg}",
                    expected="Valid Python syntax",
                    observed=str(e),
                    severity=1.0,
                    snippet=e.text.strip() if e.text else None,
                )
            )
            return VerificationResult(passed=False, score=0.0, failures=failures)

        # Step 2: Sandbox Execution of user code + task unit tests
        globals_dict: Dict[str, Any] = {"__name__": "__main__"}
        stdout_buf = io.StringIO()

        def _execute_sandboxed():
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, globals_dict)
                return True

        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute_sandboxed)
                future.result(timeout=self.execution_timeout)
            score += 0.15  # Runtime definition executed ok
        except TimeoutError:
            failures.append(
                VerificationFailure(
                    category="timeout_error",
                    message=f"Code execution timed out after {self.execution_timeout}s (possible infinite loop)",
                    expected="Fast termination",
                    observed="Timeout",
                    severity=1.0,
                )
            )
            return VerificationResult(passed=False, score=score, failures=failures)
        except Exception as e:
            failures.append(
                VerificationFailure(
                    category="runtime_error",
                    message=f"Execution error: {type(e).__name__}: {str(e)}",
                    expected="Clean execution",
                    observed=traceback.format_exc(),
                    severity=0.8,
                )
            )
            return VerificationResult(passed=False, score=score, failures=failures)

        # Step 3: Run Test Cases with timeout
        test_cases = task.test_cases
        if not test_cases:
            if task.expected_output is not None:
                stdout_val = stdout_buf.getvalue().strip()
                if str(task.expected_output).strip() in stdout_val:
                    score += 0.60
                    return VerificationResult(passed=True, score=1.0, failures=[])
                else:
                    failures.append(
                        VerificationFailure(
                            category="output_mismatch",
                            message="Expected output not found in stdout",
                            expected=str(task.expected_output),
                            observed=stdout_val,
                            severity=0.6,
                        )
                    )
                    return VerificationResult(passed=False, score=score, failures=failures)
            return VerificationResult(passed=True, score=1.0, failures=[])

        passed_tests = 0
        total_tests = len(test_cases)

        def _run_single_test(tc, idx):
            test_name = tc.get("name", f"test_case_{idx}")
            test_code = tc.get("assert_code")
            fn_name = tc.get("function_name")
            inputs = tc.get("inputs", [])
            expected = tc.get("expected")

            if test_code:
                exec(test_code, globals_dict)
                return True, None
            elif fn_name:
                if fn_name not in globals_dict:
                    return False, VerificationFailure(
                        category="missing_function",
                        message=f"Required function '{fn_name}' not defined in solution",
                        expected=f"def {fn_name}(...)",
                        observed="Function not found",
                        test_name=test_name,
                        severity=0.9,
                    )
                fn = globals_dict[fn_name]
                result = fn(*inputs) if isinstance(inputs, list) else fn(**inputs)
                if expected is not None:
                    if result == expected or str(result) == str(expected):
                        return True, None
                    else:
                        return False, VerificationFailure(
                            category="unit_test_failure",
                            message=f"{test_name} returned incorrect result for {fn_name}",
                            expected=str(expected),
                            observed=str(result),
                            test_name=test_name,
                            severity=0.6,
                        )
                return True, None
            return True, None

        for idx, tc in enumerate(test_cases, start=1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    fut = executor.submit(_run_single_test, tc, idx)
                    passed_single, fail_obj = fut.result(timeout=self.execution_timeout)
                if passed_single:
                    passed_tests += 1
                elif fail_obj:
                    failures.append(fail_obj)
            except TimeoutError:
                failures.append(
                    VerificationFailure(
                        category="test_timeout",
                        message=f"Test case {tc.get('name', idx)} timed out after {self.execution_timeout}s",
                        expected="Termination within limit",
                        observed="Timeout",
                        severity=0.8,
                    )
                )
            except Exception as e:
                failures.append(
                    VerificationFailure(
                        category="assertion_error",
                        message=f"Test {tc.get('name', idx)} raised {type(e).__name__}: {str(e)}",
                        expected=str(tc.get("expected", "Pass")),
                        observed=traceback.format_exc(),
                        severity=0.7,
                    )
                )

        test_ratio = (passed_tests / total_tests) if total_tests > 0 else 1.0
        score += (0.60 * test_ratio)
        passed = (passed_tests == total_tests) and len(failures) == 0

        return VerificationResult(
            passed=passed,
            score=min(1.0, score),
            failures=failures,
            execution_output=stdout_buf.getvalue(),
            details={"passed_tests": passed_tests, "total_tests": total_tests},
        )


# Register verifier
VerifierRegistry.register(TaskDomain.CODE, CodeVerifier())
