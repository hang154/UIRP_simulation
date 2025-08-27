# Core/Scheduler/scheduler.py
import datetime
import math
import time
from typing import List
from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.Scheduler.interface import TaskSelector, MetricEvaluator
from Core.Scheduler.registry import COMBO_REG, DISP_REG

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable=None, **kwargs):
        return iterable

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class BaselineScheduler:
    def __init__(self, *, algo="bf", time_gap=datetime.timedelta(minutes=5),
                 selector: TaskSelector = None,
                 evaluator: MetricEvaluator = None,
                 verbose: int = 0):
        from Core.Scheduler.task_selector.fifo import FIFOTaskSelector
        from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
        self.selector = selector or FIFOTaskSelector()
        self.generator = COMBO_REG[algo]()
        self.dispatcher = DISP_REG[algo]()
        self.evaluator = evaluator or BaselineEvaluator()
        self.time_gap = time_gap
        self.verbose = verbose
        self.waiting_tasks: List[Task] = []
        self.results: List[Assignment] = []
        # Tasks that were attempted but could not be scheduled under the
        # current provider state. These will be skipped until the provider
        # availability changes.
        self._unschedulable: set[str] = set()
        # Timestamp of the next provider availability change. Scheduling is
        # skipped until this time unless new tasks arrive.
        self._next_provider_event: datetime.datetime | None = None

    def _feed(self, prev: datetime.datetime, now: datetime.datetime, tasks):
        """Add newly available tasks and drop expired ones.

        Tasks whose deadlines have passed are removed from the waiting list.
        New tasks are added only if their start_time is within ``(prev, now]`` to
        avoid re-adding completed tasks.
        """

        # Drop expired tasks
        expired = {t.id for t in self.waiting_tasks if t.deadline < now}
        if expired:
            self.waiting_tasks = [t for t in self.waiting_tasks if t.deadline >= now]
            self._unschedulable.difference_update(expired)

        ids = {t.id for t in self.waiting_tasks}
        for t in tasks:
            # Add task to queue if it becomes available in this interval and it
            # still has unassigned scenes
            if (
                prev < t.start_time <= now
                and t.deadline >= now
                and t.id not in ids
                and any(st is None for st, _ in t.scene_allocation_data)
            ):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t: t.start_time)

    def _schedule_once(self, now, ps):
        new: List[Assignment] = []
        remain = []
        for t in self.selector.select(now, self.waiting_tasks):
            # Skip tasks that are already complete
            if all(st is not None for st, _ in t.scene_allocation_data):
                continue
            # Skip tasks known to be unschedulable until provider state changes
            if t.id in self._unschedulable:
                remain.append(t)
                continue

            best = self.generator.best_combo(t, ps, now, self.evaluator, verbose=self.verbose >= 2)
            if best is None:
                remain.append(t)
                self._unschedulable.add(t.id)
                continue
            cmb, t_tot, cost = best
            if self.verbose >= 2:
                print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            new_assgn = self.dispatcher.dispatch(t, cmb, now, ps, self.evaluator, self.verbose >= 2)
            new += new_assgn
            after_missing = sum(1 for st, _ in t.scene_allocation_data if st is None)
            # Keep tasks with remaining scenes for the next iteration
            if after_missing > 0:
                remain.append(t)
        self.waiting_tasks = remain
        return new

    def _compute_next_event(self, ps: Providers, after: datetime.datetime) -> datetime.datetime | None:
        """Earliest time when provider availability may change."""
        times: List[datetime.datetime] = []
        for p in ps:
            times += [f for *_, _, f in p.schedule if f > after]
            times += [s for s, _ in getattr(p, "available_hours", []) if s > after]
        return min(times) if times else None

    def run(self, tasks: Tasks, ps: Providers,
            time_start: datetime.datetime | None = None,
            time_end: datetime.datetime | None = None) -> List[Assignment]:
        if time_start is None:
            # Note: provider available_hours may be empty
            starts = []
            for p in ps:
                if getattr(p, 'available_hours', None):
                    starts.append(min(a[0] for a in p.available_hours))
            time_start = min(starts) if starts else min(t.start_time for t in tasks)
        now = time_start
        prev = now - self.time_gap
        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)
        steps = math.ceil((time_end - time_start) / self.time_gap)
        pbar = tqdm(range(steps), disable=self.verbose < 1)
        for step in pbar:
            step_start = time.time()
            self._feed(prev, now, tasks)
            feed_elapsed = time.time() - step_start
            waiting_before = len(self.waiting_tasks)
            sched_start = time.time()

            need_schedule = any(t.id not in self._unschedulable for t in self.waiting_tasks)
            if self._next_provider_event and now >= self._next_provider_event:
                self._unschedulable.clear()
                need_schedule = True
            if need_schedule:
                new = self._schedule_once(now, ps)
                self._next_provider_event = self._compute_next_event(ps, now)
            else:
                new = []

            sched_elapsed = time.time() - sched_start
            waiting_after = len(self.waiting_tasks)
            self.results += new
            total_elapsed = time.time() - step_start
            if self.verbose >= 1:
                msg = (
                    f"[step {step}] waiting={waiting_before}->{waiting_after} "
                    f"assigned={len(new)} feed={feed_elapsed:.3f}s "
                    f"schedule={sched_elapsed:.3f}s total={total_elapsed:.3f}s"
                )
                if hasattr(pbar, "write"):
                    pbar.write(msg)
                else:
                    print(msg)
            if all(all(st is not None for st, _ in t.scene_allocation_data) for t in tasks):
                break
            prev, now = now, now + self.time_gap
        return self.results
