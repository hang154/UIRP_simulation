# Core/Scheduler/task_selector/fifo.py
from __future__ import annotations
from Core.Scheduler.interface import TaskSelector
from typing import List, Sequence
from Model.tasks import Task
import datetime

class FIFOTaskSelector(TaskSelector):
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]:
        return list(waiting)
