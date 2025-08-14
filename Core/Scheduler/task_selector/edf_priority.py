"""Earliest-Deadline-First task selector with budget-aware tie-break."""

from __future__ import annotations

import datetime
from typing import List, Sequence

from Core.Scheduler.interface import TaskSelector
from Model.tasks import Task


class EDFPriorityTaskSelector(TaskSelector):
    """Order tasks by slack and budget efficiency.

    Tasks with the smallest time *slack* (deadline minus current time) are
    prioritized. When two tasks have similar slack, the one with higher budget
    per remaining scene is selected first.
    """

    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]:
        def score(t: Task) -> tuple[float, float]:
            remaining = sum(st is None for st, _ in t.scene_allocation_data)
            if remaining <= 0:
                remaining = 1  # avoid division by zero; complete tasks are filtered elsewhere
            slack_hours = (t.deadline - now).total_seconds() / 3600.0
            return (slack_hours, -t.budget / remaining)

        return sorted(waiting, key=score)
