# Core/Scheduler/combo_generator/simulated_annealing.py
from __future__ import annotations

import math
import random
from typing import List

from Core.Scheduler.interface import ComboGenerator
from Core.Scheduler.combo_generator.greedy import GreedyComboGenerator


class SimulatedAnnealingComboGenerator(ComboGenerator):
    """Simulated annealing based combo generator.

    Starting from a greedy solution, the algorithm performs random local
    modifications and accepts them based on the Metropolis criterion. The
    temperature is decreased multiplicatively each iteration. The number of
    iterations is capped at ``(scene_cnt * provider_cnt) ** 1.5``.
    """

    def __init__(self, factor: float = 1.0, start_temp: float = 1.0, cooling: float = 0.99):
        self.factor = factor
        self.start_temp = start_temp
        self.cooling = cooling

    def time_complexity(self, t, ps, now, ev):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        n = len(scene_ids) * len(ps)
        return int(self.factor * (n ** 1.5))

    def _initial_solution(self, t, ps, now, ev):
        greedy = GreedyComboGenerator().best_combo(t, ps, now, ev)
        if greedy is None:
            return None
        cmb, t_tot, cost = greedy
        ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
        if not ok:
            return None
        score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
        return cmb, t_tot, cost, score

    def _random_neighbor(self, t, ps, scene_ids: List[int], cmb: List[int]) -> List[int]:
        new_cmb = cmb.copy()
        sid = random.choice(scene_ids)
        used = {pid for pid in new_cmb if pid != -1 and pid != new_cmb[sid]}
        choices = [-1] + [pid for pid in range(len(ps)) if pid not in used]
        new_cmb[sid] = random.choice(choices)
        return new_cmb

    def best_combo(self, t, ps, now, ev, verbose=False):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        start = self._initial_solution(t, ps, now, ev)
        if start is None:
            return None
        current_cmb, current_t, current_cost, current_score = start
        best_cmb, best_t, best_cost, best_score = start

        max_iter = self.time_complexity(t, ps, now, ev)
        temp = self.start_temp
        for _ in range(max_iter):
            cand = self._random_neighbor(t, ps, scene_ids, current_cmb)
            if all(pid == -1 for pid in cand):
                temp *= self.cooling
                continue
            ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cand, now, ps)
            if not ok:
                temp *= self.cooling
                continue
            score = ev.efficiency(t, cand, ps, now, t_tot, cost, deferred, overB, overDL)
            delta = score - current_score
            if delta > 0 or math.exp(delta / max(temp, 1e-6)) > random.random():
                current_cmb, current_t, current_cost, current_score = cand, t_tot, cost, score
                if score > best_score:
                    best_cmb, best_t, best_cost, best_score = cand.copy(), t_tot, cost, score
            temp *= self.cooling

        return best_cmb, best_t, best_cost

