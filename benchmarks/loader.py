"""
Benchmark task loader and dataset manager.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict
from bwr.models import Task, TaskDomain
from bwr.task import TaskLoader


class BenchmarkSuite:
    """
    Loads benchmark datasets across Code, Math, Mechanics, Structured, and Traps.
    """

    def __init__(self, benchmark_dir: str | Path = "benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)

    def load_domain_tasks(self, domain_folder: str) -> List[Task]:
        json_file = self.benchmark_dir / domain_folder / "tasks.json"
        if not json_file.exists():
            return []
        return TaskLoader.load_from_json(json_file)

    def load_all_tasks(
        self,
        domains: Optional[List[TaskDomain | str]] = None,
        max_tasks_per_domain: Optional[int] = None,
    ) -> List[Task]:
        folders = ["code", "math", "mechanics", "traps", "mixed"]
        all_tasks: List[Task] = []

        for folder in folders:
            tasks = self.load_domain_tasks(folder)
            if domains:
                dom_strs = [d.value if isinstance(d, TaskDomain) else str(d) for d in domains]
                tasks = [t for t in tasks if t.domain.value in dom_strs]
            if max_tasks_per_domain and max_tasks_per_domain > 0:
                tasks = tasks[:max_tasks_per_domain]
            all_tasks.extend(tasks)

        return all_tasks
