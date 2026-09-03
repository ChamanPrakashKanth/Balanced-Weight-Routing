"""
Ollama and Mock model clients with real token/evaluation metadata tracking.
"""

from __future__ import annotations
import time
import re
import math
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import httpx
from bwr.models import ModelProfile, ModelResponse, Task


class BaseModelClient(ABC):
    """
    Abstract interface for local and simulated language models.
    """

    def __init__(self, profile: ModelProfile):
        self.profile = profile

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> ModelResponse:
        pass


class OllamaModelClient(BaseModelClient):
    """
    Client for live local Ollama instances with exact token telemetry.
    """

    def __init__(
        self,
        profile: ModelProfile,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 90.0,
    ):
        super().__init__(profile)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> ModelResponse:
        temp = temperature if temperature is not None else self.profile.default_temperature
        start_time = time.perf_counter()

        payload: Dict[str, Any] = {
            "model": self.profile.ollama_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_ctx": self.profile.context_length,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.host}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()

            elapsed = time.perf_counter() - start_time
            text = data.get("response", "")

            # Exact token stats from Ollama
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            # Fallback estimation only if Ollama does not return counts
            if prompt_tokens == 0:
                prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
            if completion_tokens == 0:
                completion_tokens = max(1, len(text.split()) * 4 // 3)

            confidence = self._extract_confidence(text)

            return ModelResponse(
                model_id=self.profile.id,
                model_name=self.profile.name,
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_seconds=elapsed,
                self_reported_confidence=confidence,
                metadata={
                    "total_duration_ns": data.get("total_duration"),
                    "eval_duration_ns": data.get("eval_duration"),
                    "eval_count": completion_tokens,
                },
                raw_response=data,
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return ModelResponse(
                model_id=self.profile.id,
                model_name=self.profile.name,
                text=f"ERROR: Failed to query Ollama model {self.profile.ollama_name}: {str(e)}",
                prompt_tokens=len(prompt.split()) * 4 // 3,
                completion_tokens=0,
                latency_seconds=elapsed,
                self_reported_confidence=0.0,
                metadata={"error": str(e)},
            )

    @staticmethod
    def _extract_confidence(text: str) -> Optional[float]:
        # Look for self-reported confidence patterns like "Confidence: 0.9" or "Confidence: 90%"
        match = re.search(r"confidence(?:\s*level)?[:\s]+(\d+(?:\.\d+)?)\s*(%?)", text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            is_pct = match.group(2) == "%" or val > 1.0
            return min(1.0, max(0.0, val / 100.0 if is_pct else val))
        return None


class MockModelClient(BaseModelClient):
    """
    Deterministic mock model client for test suites, offline benchmarking,
    and controlled capability ablation studies.
    """

    def __init__(self, profile: ModelProfile, seed: int = 42):
        super().__init__(profile)
        self.rng = random.Random(seed + hash(profile.id))
        self.mock_profile = profile.mock_profile or {}
        self.capabilities = self.mock_profile.get("capabilities", {})
        self.speed_factor = float(self.mock_profile.get("speed_factor", 1.0))
        self.token_multiplier = float(self.mock_profile.get("token_cost_multiplier", 1.0))

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> ModelResponse:
        # Prompt and completion token counts
        prompt_tokens = max(10, int(len(prompt.split()) * 1.3))
        
        # Determine capability for task domain
        domain_key = task.domain.value if task else "code"
        model_cap = self.capabilities.get(domain_key, 0.5)
        
        # Trap susceptibility
        if task and task.is_trap:
            trap_res = self.capabilities.get("trap_resistance", 0.5)
            # Low trap resistance means model gets tricked into common hallucination
            success_prob = trap_res * (1.0 - task.difficulty * 0.5)
        elif task:
            # Capability model: probability of success depends on model_cap vs task difficulty
            # Logistic curve: P(success) = 1 / (1 + exp(-k * (cap - diff)))
            margin = model_cap - task.difficulty
            success_prob = 1.0 / (1.0 + np.exp(-6.0 * margin))
        else:
            success_prob = model_cap

        # Determine if this attempt succeeds
        roll = self.rng.random()
        passed = roll < success_prob

        # Model generates text
        if passed and task and task.reference_solution:
            solution_text = task.reference_solution
            # Add synthetic confidence
            conf = min(0.99, max(0.70, model_cap + self.rng.uniform(-0.05, 0.05)))
            text = f"```python\n{solution_text}\n```\nConfidence: {int(conf*100)}%"
        elif task and task.is_trap and not passed:
            # Produce plausible hallucination
            wrong_sol = task.metadata.get("hallucinated_solution", "# Incorrect plausible assumption\ndef solve(): return -999")
            # Small models often exhibit false overconfidence
            conf = min(0.99, max(0.85, 0.90 + self.rng.uniform(-0.05, 0.05)))
            text = f"```python\n{wrong_sol}\n```\nI am completely confident. Confidence: {int(conf*100)}%"
        elif task and not passed:
            # Produce flawed partial attempt
            flawed = task.metadata.get("flawed_solution", "# Incomplete implementation\ndef solve(): return 0")
            conf = min(0.95, max(0.40, model_cap + self.rng.uniform(-0.15, 0.15)))
            text = f"```python\n{flawed}\n```\nConfidence: {int(conf*100)}%"
        else:
            text = f"Generated solution from {self.profile.name}.\nConfidence: 80%"
            conf = 0.80

        completion_tokens = max(15, int(len(text.split()) * 1.3))
        # Latency simulation: base latency proportional to completion tokens and tier speed
        latency = 0.05 + (completion_tokens * 0.003 * self.speed_factor)

        return ModelResponse(
            model_id=self.profile.id,
            model_name=self.profile.name,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            self_reported_confidence=conf,
            metadata={
                "simulated": True,
                "success_prob": success_prob,
                "passed_simulation": passed,
            },
        )


def create_model_client(
    profile: ModelProfile,
    use_mock: bool = False,
    server_host: str = "http://127.0.0.1:11434",
    seed: int = 42,
) -> BaseModelClient:
    if use_mock:
        return MockModelClient(profile, seed=seed)
    return OllamaModelClient(profile, host=server_host)
