"""
Orchestration pipeline executing routing loops, budget enforcement, verification, and telemetry.
"""

from __future__ import annotations
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from bwr.models import (
    Task,
    ModelProfile,
    RoutingState,
    RoutingHistoryItem,
    VerificationResult,
    ModelResponse,
)
from bwr.cost import CostTracker, CostConfig
from bwr.budget import BudgetController, BudgetLimits
from bwr.capability import EmpiricalCapabilityMatrix
from bwr.verifier import VerifierRegistry
from bwr.telemetry import TelemetryLogger
from bwr.ollama_client import BaseModelClient, create_model_client
from bwr.router import BaseRouter, create_router


class Orchestrator:
    """
    Executes tasks through a specified routing policy, enforcing verification, budget, and telemetry.
    """

    def __init__(
        self,
        profiles: List[ModelProfile],
        use_mock: bool = False,
        server_host: str = "http://127.0.0.1:11434",
        cost_config: Optional[CostConfig] = None,
        budget_limits: Optional[BudgetLimits] = None,
        telemetry_logger: Optional[TelemetryLogger] = None,
        seed: int = 42,
    ):
        self.profiles = profiles
        self.use_mock = use_mock
        self.server_host = server_host
        self.cost_tracker = CostTracker(cost_config or CostConfig())
        self.budget_controller = BudgetController(budget_limits or BudgetLimits())
        self.telemetry_logger = telemetry_logger or TelemetryLogger()
        self.seed = seed

        # Initialize clients for all profiles
        self.clients: Dict[str, BaseModelClient] = {
            p.id: create_model_client(p, use_mock=use_mock, server_host=server_host, seed=seed)
            for p in self.profiles
        }

    def run_task(
        self,
        task: Task,
        router: BaseRouter,
        run_id: str,
        experiment_name: str = "experiment",
        capability_matrix: Optional[EmpiricalCapabilityMatrix] = None,
    ) -> RoutingState:
        """
        Executes a single task through the routing loop until termination or budget exhaustion.
        """
        state = RoutingState(task=task)
        last_response: Optional[ModelResponse] = None
        last_verification: Optional[VerificationResult] = None

        while True:
            # Check budget before taking another step
            within_budget, breach_reason = self.budget_controller.check_budget(
                accumulated_cost=state.accumulated_cost,
                accumulated_tokens=state.accumulated_tokens,
                current_step=state.current_step,
                accumulated_latency=state.accumulated_latency,
            )
            if not within_budget:
                state.termination_reason = breach_reason or "budget_exceeded"
                break

            # Router makes decision
            decision = router.route_step(
                state=state,
                last_response=last_response,
                last_verification=last_verification,
            )

            # Prepare prompt (full or minimal residual)
            prompt_text, res_tokens, ctx_reduction = router.prepare_prompt(
                state=state,
                decision=decision,
                last_response=last_response,
                last_verification=last_verification,
            )

            # Generate response from chosen model
            client = self.clients[decision.selected_model_id]
            response = client.generate(
                prompt=prompt_text,
                task=task,
            )

            # Verify response deterministically
            v_start = time.perf_counter()
            verification = VerifierRegistry.verify_task(task, response)
            v_elapsed = time.perf_counter() - v_start
            verification.latency_seconds = v_elapsed

            # Compute cost for this step
            step_cost = self.cost_tracker.calculate_step_cost(
                response=response,
                model_id=decision.selected_model_id,
            )

            # Update accumulated statistics
            state.accumulated_cost += step_cost
            state.accumulated_tokens += response.total_tokens
            state.accumulated_latency += (response.latency_seconds + v_elapsed)
            state.current_residual = verification.residual_error

            # Append to history
            history_item = RoutingHistoryItem(
                step=state.current_step + 1,
                model_id=decision.selected_model_id,
                model_name=client.profile.name,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_seconds=response.latency_seconds,
                cost=step_cost,
                verification_score=verification.score,
                passed=verification.passed,
                failures=[f.__dict__ for f in verification.failures],
                is_residual=decision.is_residual,
                residual_context_tokens=res_tokens,
                context_reduction_ratio=ctx_reduction,
                self_reported_confidence=response.self_reported_confidence,
            )
            state.history.append(history_item)

            # Telemetry logging
            self.telemetry_logger.log_task_step(
                run_id=run_id,
                experiment_name=experiment_name,
                task_id=task.id,
                domain=task.domain.value,
                step=state.current_step + 1,
                model_id=decision.selected_model_id,
                model_name=client.profile.name,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_seconds=response.latency_seconds,
                cost=step_cost,
                verification_score=verification.score,
                passed=verification.passed,
                failures=[f.__dict__ for f in verification.failures],
                accumulated_cost=state.accumulated_cost,
                accumulated_tokens=state.accumulated_tokens,
                is_residual=decision.is_residual,
                residual_tokens=res_tokens,
                context_reduction_ratio=ctx_reduction,
                self_reported_confidence=response.self_reported_confidence,
            )

            # Update capability matrix if online learning is active
            if capability_matrix is not None:
                capability_matrix.record_attempt(
                    model_id=decision.selected_model_id,
                    domain=task.domain,
                    passed=verification.passed,
                    cost=step_cost,
                    residual_reduction=1.0 - verification.residual_error,
                )

            # Advance state pointers
            last_response = response
            last_verification = verification
            state.final_response = response
            state.final_verification = verification
            state.current_step += 1

            # Termination check
            should_term, term_reason = router.should_terminate(
                state=state,
                last_verification=last_verification,
                last_response=last_response,
            )
            if should_term:
                state.termination_reason = term_reason
                state.success = (term_reason == "verified_success")
                break

        return state

    def run_benchmark(
        self,
        tasks: List[Task],
        router: BaseRouter,
        experiment_name: str = "experiment",
        capability_matrix: Optional[EmpiricalCapabilityMatrix] = None,
    ) -> List[RoutingState]:
        """
        Runs a suite of tasks under a reproducible run ID.
        """
        run_id = self.telemetry_logger.generate_run_id(experiment_name)
        results: List[RoutingState] = []
        for task in tasks:
            st = self.run_task(
                task=task,
                router=router,
                run_id=run_id,
                experiment_name=experiment_name,
                capability_matrix=capability_matrix,
            )
            results.append(st)
        return results
