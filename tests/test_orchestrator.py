"""
Integration tests for Orchestrator execution, telemetry, and benchmark evaluation.
"""

import pytest
from bwr.models import Task, TaskDomain, ModelProfile
from bwr.orchestrator import Orchestrator
from bwr.router import StrongestOnlyRouter, VerifiedResidualRouter
from bwr.telemetry import TelemetryLogger


@pytest.fixture
def mock_ladder():
    return [
        ModelProfile(
            id="model_1",
            name="M1",
            ollama_name="qwen:0.5b",
            cost_tier=1,
            mock_profile={"capabilities": {"code": 0.20}, "speed_factor": 1.0, "token_cost_multiplier": 0.1},
        ),
        ModelProfile(
            id="model_5",
            name="M5",
            ollama_name="qwen:14b",
            cost_tier=5,
            mock_profile={"capabilities": {"code": 0.99}, "speed_factor": 5.0, "token_cost_multiplier": 1.0},
        ),
    ]


def test_orchestrator_run_task(tmp_path, mock_ladder):
    task = Task(
        id="task_orch_01",
        title="Add Numbers",
        prompt="Write a function `add(a, b)`",
        domain=TaskDomain.CODE,
        reference_solution="def add(a, b): return a + b",
        test_cases=[{"function_name": "add", "inputs": [2, 3], "expected": 5}],
    )

    logger = TelemetryLogger(log_dir=tmp_path / "logs")
    orchestrator = Orchestrator(
        profiles=mock_ladder,
        use_mock=True,
        telemetry_logger=logger,
        seed=42,
    )

    router = StrongestOnlyRouter(mock_ladder)
    state = orchestrator.run_task(
        task=task,
        router=router,
        run_id="test_run_01",
        experiment_name="test_exp",
    )

    assert state.current_step == 1
    assert state.success is True
    assert state.accumulated_cost > 0
    assert len(state.history) == 1
    assert state.history[0].model_id == "model_5"

    # Verify JSONL log was written
    log_files = list((tmp_path / "logs").glob("*.jsonl"))
    assert len(log_files) == 1
