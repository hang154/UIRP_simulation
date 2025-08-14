# ================================================================
# scheduler.py – common interface wrapper
# ================================================================
from __future__ import annotations
import datetime
from typing import Protocol, Optional, List, Tuple

from Model.tasks import Tasks
from Model.providers import Providers
from Core.Scheduler.scheduler import BaselineScheduler as _BaselineScheduler

# ---------- Common Assignment type ----------
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

# ---------- Minimum scheduler protocol ----------
class Scheduler(Protocol):
    def run(
        self,
        tasks: Tasks,
        providers: Providers,
        time_start: Optional[datetime.datetime] = None,
        time_end:   Optional[datetime.datetime] = None,
    ) -> List[Assignment]: ...

# ---------- re-export ----------
BaselineScheduler = _BaselineScheduler

__all__ = ["Scheduler", "Assignment", "BaselineScheduler"]
