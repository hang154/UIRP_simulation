# Core/Scheduler/combo_generator/monte_carlo.py
from __future__ import annotations

import random
from typing import List

from Core.Scheduler.interface import ComboGenerator


class MonteCarloComboGenerator(ComboGenerator):
    """Randomised search for scene-provider assignments.

    The algorithm samples random combinations while respecting the
    "one scene per provider" constraint. Among all sampled combinations
    the one with the highest efficiency score is returned. The number of
    samples is bounded by ``(scene_cnt * provider_cnt) ** 1.5`` to keep
    the time complexity below the required ``timestep * (scene *
    provider) ** 1.5`` limit.
    """

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def time_complexity(self, t, ps, now, ev):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        n = len(scene_ids) * len(ps)
        return int(self.factor * (n ** 1.5))

    def _random_combo(self, t, ps, scene_ids: List[int]) -> List[int]:
        cmb = [-1] * t.scene_number
        used = set()
        for sid in scene_ids:
            choices = [-1] + [pid for pid in range(len(ps)) if pid not in used]
            pid = random.choice(choices)
            if pid != -1:
                used.add(pid)
            cmb[sid] = pid
        return cmb

    def best_combo(self, t, ps, now, ev, verbose=False):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        samples = self.time_complexity(t, ps, now, ev)
        best_score = float("-inf")
        best_res = None

        for _ in range(samples):
            cmb = self._random_combo(t, ps, scene_ids)
            if all(pid == -1 for pid in cmb):
                continue
            ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
            if not ok:
                continue
            score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
            if score > best_score:
                best_score = score
                best_res = (cmb.copy(), t_tot, cost)

        return best_res

