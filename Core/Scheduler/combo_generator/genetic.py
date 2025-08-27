# Core/Scheduler/combo_generator/genetic.py
from __future__ import annotations

import random
from typing import List, Tuple

from Core.Scheduler.interface import ComboGenerator


class GeneticComboGenerator(ComboGenerator):
    """Genetic algorithm for combo generation.

    A small population of candidate combinations is evolved through
    crossover and mutation. The population size and number of generations
    are chosen so that the total number of evaluated combinations does not
    exceed ``(scene_cnt * provider_cnt) ** 1.5``.
    """

    def __init__(self, pop_size: int = 20, factor: float = 1.0, mutation: float = 0.1):
        self.pop_size = pop_size
        self.factor = factor
        self.mutation = mutation

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

    def _evaluate(self, t, ps, now, ev, cmb: List[int]) -> Tuple[float, Tuple[List[int], float, float]]:
        ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
        if not ok:
            return float("-inf"), (cmb, 0.0, 0.0)
        score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
        return score, (cmb, t_tot, cost)

    def _crossover(self, a: List[int], b: List[int], ps_len: int) -> List[int]:
        child = [-1] * len(a)
        used = set()
        for i in range(len(a)):
            pid = random.choice([a[i], b[i]])
            if pid != -1 and pid in used:
                pid = -1
            if pid != -1:
                used.add(pid)
            child[i] = pid
        return child

    def _mutate(self, cmb: List[int], ps_len: int):
        used = {pid for pid in cmb if pid != -1}
        for i in range(len(cmb)):
            if random.random() < self.mutation:
                choices = [-1] + [pid for pid in range(ps_len) if pid not in used or pid == cmb[i]]
                new_pid = random.choice(choices)
                if cmb[i] != -1:
                    used.discard(cmb[i])
                if new_pid != -1:
                    used.add(new_pid)
                cmb[i] = new_pid

    def best_combo(self, t, ps, now, ev, verbose=False):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        eval_budget = self.time_complexity(t, ps, now, ev)
        pop_size = min(self.pop_size, max(1, eval_budget))
        generations = max(1, eval_budget // pop_size)

        population: List[Tuple[float, Tuple[List[int], float, float]]] = []
        while len(population) < pop_size:
            cmb = self._random_combo(t, ps, scene_ids)
            score, res = self._evaluate(t, ps, now, ev, cmb)
            if score > float("-inf"):
                population.append((score, res))
        best_score, best_res = max(population, key=lambda x: x[0])

        for _ in range(generations):
            population.sort(key=lambda x: x[0], reverse=True)
            elites = population[: max(1, pop_size // 2)]
            new_pop = elites.copy()
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(elites, k=2)
                child = self._crossover(p1[1][0], p2[1][0], len(ps))
                self._mutate(child, len(ps))
                score, res = self._evaluate(t, ps, now, ev, child)
                if score > float("-inf"):
                    new_pop.append((score, res))
                    if score > best_score:
                        best_score, best_res = score, res
            population = new_pop

        return best_res if best_score > float("-inf") else None

