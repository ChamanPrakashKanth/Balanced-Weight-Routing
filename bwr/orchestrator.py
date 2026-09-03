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
    DemandVector,
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
            state.current_step += 1
            last_response = response
            last_verification = verification
            state.final_response = response
            state.final_verification = verification

            # Check success condition
            if verification.passed:
                state.success = True
                state.termination_reason = "verified_success"
                break

            # Closed-loop dynamic feedback: update remaining demand vector D_{t+1} <- R_obs
            r_obs = self._extract_observed_residual_vector(task, state.current_demand, verification)
            state.observed_residual_vector = r_obs
            state.current_demand = r_obs

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

    def _extract_observed_residual_vector(
        self,
        task: Task,
        current_demand: DemandVector,
        verification: VerificationResult,
    ) -> DemandVector:
        """
        Closed-loop residual vector mapping:
        Translates deterministic verification failures into remaining load dimensions R_obs.
        """
        arr = current_demand.to_array().copy()
        fail_categories = {f.category.lower() for f in verification.failures}

        # If math verification failed (tolerance/symbolic/delta)
        if any(cat in fail_categories for cat in ("math_numeric_tolerance", "symbolic_equivalence", "math_delta", "numerical_delta")):
            arr[0] = max(arr[0], task.demand.math)       # math
            arr[1] = max(arr[1], task.demand.reasoning)  # reasoning

        # If code verification failed (syntax/runtime/test assertion)
        if any(cat in fail_categories for cat in ("syntax_error", "test_assertion_failed", "runtime_error", "timeout", "missing_function")):
            arr[2] = max(arr[2], task.demand.code)       # code
            arr[5] = max(arr[5], max(0.5, task.demand.planning))  # planning
            if "test_assertion_failed" in fail_categories:
                arr[1] = max(arr[1], task.demand.reasoning)

        # If mechanics verification failed (unbalance/beam/spring/thermo)
        if any(cat in fail_categories for cat in ("mass_balance_error", "beam_moment_error", "vibration_freq_error", "thermo_energy_error", "rotating_mass_balance")):
            arr[4] = max(arr[4], task.demand.mechanics)  # mechanics
            arr[0] = max(arr[0], task.demand.math)       # math
            arr[1] = max(arr[1], task.demand.reasoning)

        # If trap verification failed (hallucinated misconception triggered)
        if any(cat in fail_categories for cat in ("hallucination_trap_triggered", "trap_violation", "forbidden_pattern_matched")):
            arr[6] = 1.0                                 # trap
            arr[1] = max(arr[1], 0.90)                   # reasoning
            arr[3] = max(arr[3], 0.70)                   # language

        # Scale by remaining error magnitude
        err_scale = max(0.2, verification.residual_error)
        arr = arr * err_scale

        return DemandVector.from_array(arr)
