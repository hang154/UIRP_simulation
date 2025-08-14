# Core/Scheduler/interface.py
from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence

from Model.tasks import Tasks, Task
from Model.providers import Providers, Provider

# Assignment: (task_id, scene_id, start, finish, provider_index)
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

class TaskSelector(ABC):
    @abstractmethod
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]: ...

class ComboGenerator(ABC):
    @abstractmethod
    def best_combo(
        self,
        task: Task,
        providers: Providers,
        sim_time: datetime.datetime,
        evaluator: "MetricEvaluator",
        verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]: ...
    # return: (combo[scene_id->provider_idx or -1], t_tot_hours, cost_usd)

    def time_complexity(
        self,
        task: Task,
        providers: Providers,
        sim_time: datetime.datetime,
        evaluator: "MetricEvaluator",
    ) -> int:
        """Return exact count of feasible assignments (excluding all-skip)."""
        return 0

class MetricEvaluator(ABC):
    @abstractmethod
    def time_cost(self, task: Task, scene_id: int, prov: Provider) -> Tuple[float, float]: ...
    # return: (duration_hours, incremental_cost_usd)

    @abstractmethod
    def feasible(
        self,
        task: Task,
        combo: List[int],
        sim_time: datetime.datetime,
        providers: Providers,
    ) -> Tuple[bool, float, float, int, float, float]:
        """
        Returns:
          ok:                물리적으로 '지금(now)' 연속 가용구간 안에서 배치 가능한지 (하드)
          t_tot[h]:          이번 스텝 동시 시작 가정의 makespan (max per provider)
          cost[$]:           이번 스텝까지의 누적 비용(과거+이번)
          deferred[int]:     이번 스텝에서 미배치(-1)된 미완료 씬 수
          over_budget[$]:    max(0, cost - task.budget)
          over_deadline_h[h]:max(0, now + t_tot - task.deadline)
        """

    @abstractmethod
    def efficiency(
        self,
        task: Task,
        combo: List[int],
        ps: Providers,
        now: datetime.datetime,
        t_tot: float,
        cost: float,
        deferred: int,
        over_budget: float,
        over_deadline_h: float,
    ) -> float: ...
    # 클수록 좋은 점수 (보통 -가중합)

class Dispatcher(ABC):
    @abstractmethod
    def dispatch(
        self,
        task: Task,
        combo: List[int],
        sim_time: datetime.datetime,
        providers: Providers,
        evaluator: MetricEvaluator,
        verbose: bool,
    ) -> List[Assignment]: ...
