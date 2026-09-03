"""
Abstract Verifier interface, response parsers, and verifier registry.
"""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
from bwr.models import Task, ModelResponse, VerificationResult, TaskDomain


class BaseVerifier(ABC):
    """
    Abstract contract for deterministic domain verifiers.
    All verifiers return VerificationResult with score in [0.0, 1.0] and structured failures.
    """

    @abstractmethod
    def verify(self, task: Task, response: ModelResponse) -> VerificationResult:
        pass


def extract_code_block(text: str, default_lang: str = "python") -> str:
    """
    Extracts code block from markdown ```python ... ``` or returns clean raw text.
    """
    pattern = rf"```(?:{default_lang})?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback to any code block
    generic_matches = re.findall(r"```\n?(.*?)```", text, re.DOTALL)
    if generic_matches:
        return generic_matches[-1].strip()

    return text.strip()


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extracts the final numeric answer from a model response.
    Looks for expressions like 'Answer: 42.5', '\\boxed{42.5}', 'result = 42.5', or last float.
    """
    # Look for \boxed{...}
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        try:
            return float(boxed.group(1).strip())
        except ValueError:
            pass

    # Look for 'Answer: <num>' or 'result = <num>'
    ans_match = re.search(r"(?:answer|result|final value|solution)\s*(?:is|=|:)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text, re.IGNORECASE)
    if ans_match:
        try:
            return float(ans_match.group(1))
        except ValueError:
            pass

    # Look for any number at the end
    numbers = re.findall(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


class VerifierRegistry:
    """
    Registry for domain-specific verifiers.
    """

    _registry: Dict[TaskDomain, BaseVerifier] = {}

    @classmethod
    def register(cls, domain: TaskDomain, verifier: BaseVerifier) -> None:
        cls._registry[domain] = verifier

    @classmethod
    def get_verifier(cls, domain: TaskDomain) -> BaseVerifier:
        if domain in cls._registry:
            return cls._registry[domain]
        # Default fallback
        from verifiers.code_verifier import CodeVerifier
        return cls._registry.get(TaskDomain.CODE, CodeVerifier())

    @classmethod
    def verify_task(cls, task: Task, response: ModelResponse) -> VerificationResult:
        verifier = cls.get_verifier(task.domain)
        return verifier.verify(task, response)
