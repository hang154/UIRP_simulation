"""Provider model definitions."""

from __future__ import annotations
import datetime
from typing import Dict, Any, List, Tuple, Optional
from utils.utils import merge_intervals


class Provider:
    def __init__(self, d: Dict[str, Any]):
        self.throughput: float = float(d.get("throughput", 1.0))  # GFLOP/s
        self.price_per_gpu_hour: float = float(d.get("price", 0.0))  # $
        self.bandwidth: float = float(d.get("bandwidth", 0.0))  # MB/s

        raw = d.get("available_hours", [])
        self.available_hours: List[Tuple[datetime.datetime, datetime.datetime]] = [
            (
                datetime.datetime.fromisoformat(s) if isinstance(s, str) else s,
                datetime.datetime.fromisoformat(e) if isinstance(e, str) else e,
            )
            for s, e in raw
        ]

        # (task_id, scene_id, start, finish)
        self.schedule: List[Tuple[str, int, datetime.datetime, datetime.datetime]] = []

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    def idle_ratio(
        self,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ) -> float:
        if not self.schedule:
            return 1.0
        if start is None:
            start = min(s for *_, s, _ in self.schedule)
        if end is None:
            end = max(f for *_, f in self.schedule)
        horizon = (end - start).total_seconds() / 3600.0
        if horizon <= 0:
            return 1.0
        busy = merge_intervals([(s, f) for *_, s, f in self.schedule])
        busy_h = sum((f - s).total_seconds() / 3600.0 for s, f in busy)
        return max(0.0, 1.0 - busy_h / horizon)

    # ---------------------------------------------------
    # Scheduling helpers
    # ---------------------------------------------------
    def earliest_available(self, dur_h: float, after: datetime.datetime) -> Optional[datetime.datetime]:
        """Earliest time (>= after) to allocate a contiguous block of length dur_h."""
        for a_s, a_e in sorted(self.available_hours, key=lambda t: t[0]):
            if a_e <= after:
                continue
            cur = max(a_s, after)

            # Find a gap within the window that does not clash with existing schedule
            clashes = sorted(
                [(s, f) for *_, s, f in self.schedule if s < a_e and f > cur],
                key=lambda iv: iv[0],
            )
            for s, f in clashes:
                if (s - cur).total_seconds() / 3600.0 >= dur_h:
                    return cur
                cur = max(cur, f)
            if (a_e - cur).total_seconds() / 3600.0 >= dur_h:
                return cur
        return None

    def assign(self, task_id: str, scene_id: int, start: datetime.datetime, dur_h: float):
        finish = start + datetime.timedelta(hours=dur_h)
        self.schedule.append((task_id, scene_id, start, finish))

        # Update available_hours by removing the allocated interval
        new: List[Tuple[datetime.datetime, datetime.datetime]] = []
        for s, e in self.available_hours:
            if finish <= s or start >= e:
                new.append((s, e))
                continue
            if s < start:
                new.append((s, start))
            if finish < e:
                new.append((finish, e))
        self.available_hours = new


class Providers:
    def __init__(self):
        self._list: List[Provider] = []

    def initialize_from_data(self, data):
        self._list = [Provider(d) for d in data]

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __len__(self):
        return len(self._list)
