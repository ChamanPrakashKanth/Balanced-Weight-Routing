"""
Residual extraction engine for Verified Residual Routing (VRR).
Isolates unresolved sub-problems and formats minimal context prompts.
"""

from __future__ import annotations
from typing import List, Optional
from bwr.models import Task, VerificationResult, VerificationFailure, ResidualContext, ModelResponse
from bwr.cost import CostTracker


class ResidualExtractor:
    """
    Extracts minimal residual failure delta and synthesizes focused escalation prompts.
    """

    def __init__(self, include_test_failures_only: bool = True):
        self.include_test_failures_only = include_test_failures_only

    def extract_residual(
        self,
        task: Task,
        attempted_response: ModelResponse,
        verification: VerificationResult,
    ) -> ResidualContext:
        """
        Synthesizes a minimal residual prompt from:
        1. Minimal task core requirements
        2. Attempted partial solution
        3. Explicit verifier failure messages and expected vs observed deltas
        4. Target fix directive
        """
        # Full task context baseline (simulated or actual prompt tokens)
        full_context_prompt = (
            f"Original Task: {task.title}\n"
            f"{task.prompt}\n\n"
            f"Previous Attempt:\n{attempted_response.text}\n\n"
            f"Please fix all errors."
        )
        full_tokens = max(20, int(len(full_context_prompt.split()) * 1.3))

        # Build focused failure summary
        failure_blocks: List[str] = []
        for i, f in enumerate(verification.failures, start=1):
            block = f"Failure #{i} [{f.category}]: {f.message}"
            if f.expected is not None:
                block += f"\n  Expected: {f.expected}"
            if f.observed is not None:
                block += f"\n  Observed: {f.observed}"
            if f.snippet:
                block += f"\n  Context: {f.snippet}"
            failure_blocks.append(block)

        failures_text = "\n".join(failure_blocks) if failure_blocks else "Verification score below threshold."

        # Extract only the code/core solution from attempt
        clean_attempt = attempted_response.text
        if "```" in clean_attempt:
            from bwr.verifier import extract_code_block
            clean_attempt = extract_code_block(clean_attempt)

        # Construct minimal residual prompt
        residual_prompt = (
            f"Task: {task.title}\n"
            f"The previous attempt failed external verification on the following specific criteria:\n"
            f"--- VERIFICATION FAILURES ---\n"
            f"{failures_text}\n"
            f"-----------------------------\n\n"
            f"Current Code/Attempt:\n"
            f"```\n{clean_attempt}\n```\n\n"
            f"Fix ONLY the failed conditions listed above. Output the complete corrected solution."
        )

        residual_tokens = max(10, int(len(residual_prompt.split()) * 1.3))
        reduction = CostTracker.compute_context_savings(full_tokens, residual_tokens)

        return ResidualContext(
            residual_prompt=residual_prompt,
            context_reduction_ratio=reduction,
            full_context_tokens=full_tokens,
            residual_context_tokens=residual_tokens,
            isolated_failures=verification.failures,
            attempted_solution=clean_attempt,
        )
