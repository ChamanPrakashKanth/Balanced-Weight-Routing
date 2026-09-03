"""
Routing policies for LLM cascades: Strongest-only, Smallest-only, Fixed Cascade,
Confidence-based, Verified Full-Task, Verified Residual Routing (VRR), and Balanced Weight Routing (BWR).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path
import numpy as np

from bwr.models import (
    Task,
    ModelProfile,
    RoutingDecision,
    RoutingState,
    VerificationResult,
    ModelResponse,
)
from bwr.capability import EmpiricalCapabilityMatrix, FeatureVectorCapabilityMatrix
from bwr.residual import ResidualExtractor
from bwr.cost import CostTracker


class BaseRouter(ABC):
    """
    Abstract contract for routing policies.
    """

    def __init__(self, profiles: List[ModelProfile]):
        self.profiles = sorted(profiles, key=lambda p: p.cost_tier)
        self.model_map = {p.id: p for p in self.profiles}
        self.residual_extractor = ResidualExtractor()

    @abstractmethod
    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        pass

    def prepare_prompt(
        self,
        state: RoutingState,
        decision: RoutingDecision,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> Tuple[str, Optional[int], Optional[float]]:
        """
        Returns (prompt_text, residual_tokens, context_reduction_ratio).
        """
        if not decision.is_residual or last_response is None or last_verification is None:
            return state.task.prompt, None, None

        # Build residual context prompt
        res_ctx = self.residual_extractor.extract_residual(
            task=state.task,
            attempted_response=last_response,
            verification=last_verification,
        )
        return res_ctx.residual_prompt, res_ctx.residual_context_tokens, res_ctx.context_reduction_ratio

    def should_terminate(
        self,
        state: RoutingState,
        last_verification: Optional[VerificationResult],
        last_response: Optional[ModelResponse],
    ) -> Tuple[bool, str]:
        """
        Default termination rule: stops if verified passed or no more steps.
        """
        if last_verification and last_verification.passed:
            return True, "verified_success"
        if state.current_step >= len(self.profiles):
            return True, "ladder_exhausted"
        return False, "continue"


class StrongestOnlyRouter(BaseRouter):
    """
    Baseline: Directly routes all tasks to the strongest model (M_5).
    """

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        strongest = self.profiles[-1]
        return RoutingDecision(
            selected_model_id=strongest.id,
            reason="strongest_baseline",
            is_residual=False,
            estimated_capability=1.0,
        )

    def should_terminate(
        self,
        state: RoutingState,
        last_verification: Optional[VerificationResult],
        last_response: Optional[ModelResponse],
    ) -> Tuple[bool, str]:
        if state.current_step >= 1:
            passed = last_verification.passed if last_verification else False
            return True, "verified_success" if passed else "single_step_completed"
        return False, "continue"


class SmallestOnlyRouter(BaseRouter):
    """
    Baseline: Directly routes all tasks to the smallest model (M_1).
    """

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        smallest = self.profiles[0]
        return RoutingDecision(
            selected_model_id=smallest.id,
            reason="smallest_baseline",
            is_residual=False,
            estimated_capability=0.3,
        )

    def should_terminate(
        self,
        state: RoutingState,
        last_verification: Optional[VerificationResult],
        last_response: Optional[ModelResponse],
    ) -> Tuple[bool, str]:
        if state.current_step >= 1:
            passed = last_verification.passed if last_verification else False
            return True, "verified_success" if passed else "single_step_completed"
        return False, "continue"


class FixedCascadeRouter(BaseRouter):
    """
    Baseline: Strict sequential escalation M_1 -> M_2 -> M_3 -> M_4 -> M_5.
    Resends original full task prompt at each stage.
    """

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        idx = min(state.current_step, len(self.profiles) - 1)
        selected = self.profiles[idx]
        return RoutingDecision(
            selected_model_id=selected.id,
            reason=f"fixed_cascade_step_{state.current_step + 1}",
            is_residual=False,
        )


class ConfidenceRouter(BaseRouter):
    """
    Control Baseline: Escalates based on LLM self-reported confidence.
    Accepts answer if confidence >= confidence_threshold, otherwise escalates.
    """

    def __init__(self, profiles: List[ModelProfile], confidence_threshold: float = 0.85):
        super().__init__(profiles)
        self.confidence_threshold = confidence_threshold

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        idx = min(state.current_step, len(self.profiles) - 1)
        selected = self.profiles[idx]
        return RoutingDecision(
            selected_model_id=selected.id,
            reason=f"confidence_routing_step_{state.current_step + 1}",
            is_residual=False,
        )

    def should_terminate(
        self,
        state: RoutingState,
        last_verification: Optional[VerificationResult],
        last_response: Optional[ModelResponse],
    ) -> Tuple[bool, str]:
        if last_response and last_response.self_reported_confidence is not None:
            if last_response.self_reported_confidence >= self.confidence_threshold:
                # Terminate because model claimed high confidence (even if it hallucinated!)
                return True, f"accepted_by_confidence_{last_response.self_reported_confidence:.2f}"

        if state.current_step >= len(self.profiles):
            return True, "ladder_exhausted"

        return False, "continue"


class VerifiedFullTaskRouter(BaseRouter):
    """
    Verified Escalation: Uses external verification to decide escalation,
    but resends the complete original task prompt (no residual reduction).
    """

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        idx = min(state.current_step, len(self.profiles) - 1)
        selected = self.profiles[idx]
        return RoutingDecision(
            selected_model_id=selected.id,
            reason=f"verified_full_escalation_step_{state.current_step + 1}",
            is_residual=False,
        )


class VerifiedResidualRouter(BaseRouter):
    """
    Verified Residual Routing (VRR):
    Uses external verification, and escalates only the extracted residual context.
    """

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        idx = min(state.current_step, len(self.profiles) - 1)
        selected = self.profiles[idx]
        is_residual = state.current_step > 0
        return RoutingDecision(
            selected_model_id=selected.id,
            reason=f"vrr_step_{state.current_step + 1}",
            is_residual=is_residual,
        )


class BalancedWeightRouter(BaseRouter):
    """
    Balanced Weight Routing (BWR):
    Combines:
    1. Empirical capability estimation w_i(d)
    2. Efficiency-based initial model selection (eta_i = P(success) / E[K_i])
    3. Marginal escalation efficiency E_ij = Delta Q_ij / Delta K_ij
    4. Model skipping (e.g. M1 -> M4) when intermediate models are unlikely to resolve residual
    5. Minimal residual context formulation (optional via use_residual)
    """

    def __init__(
        self,
        profiles: List[ModelProfile],
        capability_matrix: Optional[EmpiricalCapabilityMatrix] = None,
        marginal_gain_min: float = 0.05,
        allow_skipping: bool = True,
        use_residual: bool = True,
    ):
        super().__init__(profiles)
        self.capability_matrix = capability_matrix or EmpiricalCapabilityMatrix(
            model_ids=[p.id for p in self.profiles]
        )
        self.marginal_gain_min = marginal_gain_min
        self.allow_skipping = allow_skipping
        self.use_residual = use_residual

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        domain = state.task.domain

        # Step 0: Initial model selection based on efficiency eta_i
        if state.current_step == 0:
            best_model_id = self._select_initial_model(domain)
            p_succ = self.capability_matrix.estimate_success_probability(best_model_id, domain)
            eff = self.capability_matrix.calculate_efficiency(best_model_id, domain)
            return RoutingDecision(
                selected_model_id=best_model_id,
                reason="bwr_initial_efficiency_selection",
                is_residual=False,
                expected_efficiency=eff,
                estimated_capability=p_succ,
            )

        # Escalation step: select next model via marginal gain E_ij and skipping
        current_model_id = state.history[-1].model_id if state.history else self.profiles[0].id
        next_model_id, skip_jump = self._select_escalation_model(current_model_id, domain)
        p_succ = self.capability_matrix.estimate_success_probability(next_model_id, domain)
        eff = self.capability_matrix.calculate_efficiency(next_model_id, domain)

        return RoutingDecision(
            selected_model_id=next_model_id,
            reason=f"bwr_marginal_escalation_skip_{skip_jump}",
            is_residual=self.use_residual,
            expected_efficiency=eff,
            estimated_capability=p_succ,
            skip_jump=skip_jump,
        )

    def _select_initial_model(self, domain: str | TaskDomain) -> str:
        """
        Rank all models by efficiency eta_i = P(success) / E[K_i] while ensuring baseline viability.
        """
        ranked = []
        for p in self.profiles:
            eff = self.capability_matrix.calculate_efficiency(p.id, domain)
            p_succ = self.capability_matrix.estimate_success_probability(p.id, domain)
            ranked.append((eff, p_succ, p.id))

        # Sort by efficiency descending
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return ranked[0][2]

    def _select_escalation_model(self, current_model_id: str, domain: str | TaskDomain) -> Tuple[str, int]:
        """
        Selects next model with highest marginal efficiency E_ij above threshold,
        allowing skipping weak intermediate models.
        """
        current_idx = next((i for i, p in enumerate(self.profiles) if p.id == current_model_id), 0)
        candidates = self.profiles[current_idx + 1:]

        if not candidates:
            # Fallback to strongest
            return self.profiles[-1].id, 0

        if not self.allow_skipping or len(candidates) == 1:
            return candidates[0].id, 0

        # Evaluate marginal gains across remaining candidates
        best_candidate = candidates[0]
        best_score = -1.0
        best_jump = 0

        for jump, candidate in enumerate(candidates):
            p_curr = self.capability_matrix.estimate_success_probability(current_model_id, domain)
            p_cand = self.capability_matrix.estimate_success_probability(candidate.id, domain)
            delta_q = max(0.0, p_cand - p_curr)

            cost_curr = self.capability_matrix.estimate_average_cost(current_model_id, domain)
            cost_cand = self.capability_matrix.estimate_average_cost(candidate.id, domain)
            delta_k = max(1e-6, cost_cand - cost_curr)

            marginal_eff = delta_q / delta_k
            score = float(delta_q * (marginal_eff ** 0.5)) if marginal_eff > 0 else 0.0

            if score > best_score and delta_q >= self.marginal_gain_min:
                best_score = score
                best_candidate = candidate
                best_jump = jump

        return best_candidate.id, best_jump


class FeatureVectorBWRRouter(BaseRouter):
    """
    Feature-Vector Balanced Weight Router (FV-BWR):
    Selects models by minimizing the joint residual imbalance and execution compute:
        J_i = lambda_R * || W (D_t - C_i)_+ ||_2^2 + lambda_K * K_i
    where D_t is the dynamic task demand vector, updated after verifier feedback.
    """

    def __init__(
        self,
        profiles: List[ModelProfile],
        feature_matrix: Optional[FeatureVectorCapabilityMatrix] = None,
        lambda_r: float = 1.0,
        lambda_k: float = 20.0,
        dimension_weights: Optional[np.ndarray] = None,
        use_residual: bool = True,
    ):
        super().__init__(profiles)
        self.feature_matrix = feature_matrix or FeatureVectorCapabilityMatrix(
            model_ids=[p.id for p in self.profiles]
        )
        self.lambda_r = lambda_r
        self.lambda_k = lambda_k
        self.dimension_weights = dimension_weights if dimension_weights is not None else np.ones(7, dtype=float)
        self.use_residual = use_residual

    def route_step(
        self,
        state: RoutingState,
        last_response: Optional[ModelResponse] = None,
        last_verification: Optional[VerificationResult] = None,
    ) -> RoutingDecision:
        # Use dynamic demand vector from state (D_0 or D_{t+1} = R_obs)
        demand = state.current_demand
        attempted_model_ids = {h.model_id for h in state.history}

        # Evaluate objective J_i for all candidate models
        candidates_scores = []
        for p in self.profiles:
            # Imbalance norm R_i = || W (D - C_i)_+ ||_2
            r_norm = self.feature_matrix.calculate_imbalance(
                p.id,
                demand,
                dimension_weights=self.dimension_weights,
            )
            # Cost proxy: parameter-scaled proxy per 1k tokens
            cost_factor = p.mock_profile.get("token_cost_multiplier", p.cost_tier * 0.2)
            k_proxy = float(cost_factor * 0.02)

            # J_i = lambda_r * R_i^2 + lambda_k * K_i
            j_score = float(self.lambda_r * (r_norm ** 2) + self.lambda_k * k_proxy)

            # Penalty if model was already attempted and failed this task without resolving
            if p.id in attempted_model_ids:
                j_score += 10.0

            candidates_scores.append((j_score, r_norm, k_proxy, p))

        # Sort by minimum balancing objective J_i
        candidates_scores.sort(key=lambda x: x[0])
        best_j, best_r, best_k, best_profile = candidates_scores[0]

        # If all unattempted models have high imbalance, pick the strongest available
        if best_profile.id in attempted_model_ids and len(attempted_model_ids) < len(self.profiles):
            unattempted = [cs for cs in candidates_scores if cs[3].id not in attempted_model_ids]
            if unattempted:
                best_j, best_r, best_k, best_profile = unattempted[0]

        is_res = self.use_residual if state.current_step > 0 else False
        expected_coverage = max(0.0, 1.0 - best_r)

        return RoutingDecision(
            selected_model_id=best_profile.id,
            reason=f"fv_bwr_imbalance_{best_r:.2f}_cost_{best_k:.3f}_J_{best_j:.3f}",
            is_residual=is_res,
            expected_efficiency=expected_coverage / max(1e-6, best_k),
            estimated_capability=expected_coverage,
        )


def create_router(
    router_name: str,
    profiles: List[ModelProfile],
    capability_matrix: Optional[EmpiricalCapabilityMatrix] = None,
    feature_matrix: Optional[FeatureVectorCapabilityMatrix] = None,
    routing_config_path: Optional[str | Path] = None,
) -> BaseRouter:
    r_name = router_name.lower().replace("-", "_")

    if r_name in ("strongest", "strongest_only", "exp01"):
        return StrongestOnlyRouter(profiles)
    elif r_name in ("smallest", "smallest_only"):
        return SmallestOnlyRouter(profiles)
    elif r_name in ("fixed_cascade", "cascade", "exp02"):
        return FixedCascadeRouter(profiles)
    elif r_name in ("confidence", "confidence_router", "exp03"):
        return ConfidenceRouter(profiles)
    elif r_name in ("verified", "verified_full", "verified_router", "exp04"):
        return VerifiedFullTaskRouter(profiles)
    elif r_name in ("residual", "residual_router", "vrr", "exp05"):
        return VerifiedResidualRouter(profiles)
    elif r_name in ("bwr", "balanced_weight", "balanced_weight_router", "exp06"):
        return BalancedWeightRouter(profiles, capability_matrix=capability_matrix)
    elif r_name in ("fv_bwr", "feature_vector_bwr", "vector_bwr", "exp07"):
        return FeatureVectorBWRRouter(profiles, feature_matrix=feature_matrix)
    else:
        raise ValueError(f"Unknown router name: {router_name}")
