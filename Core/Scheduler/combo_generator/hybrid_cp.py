"""Hybrid combo generator combining greedy pruning with CP-SAT optimisation."""

from __future__ import annotations

import math
from typing import List

from Core.Scheduler.interface import ComboGenerator
from Core.Scheduler.combo_generator.cpsat import CPSatComboGenerator
from Core.Scheduler.combo_generator.greedy import GreedyComboGenerator


class HybridCPComboGenerator(ComboGenerator):
    """CP-SAT with provider pruning and greedy fallback.

    For each unassigned scene only the top ``k`` providers (shortest estimated
    duration) are considered. The reduced provider set is optimised via the
    existing :class:`CPSatComboGenerator`. If CP-SAT fails to find a feasible
    solution, the algorithm falls back to the greedy heuristic.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._cp = CPSatComboGenerator()
        self._greedy = GreedyComboGenerator()

    # --- helpers ---------------------------------------------------------
    def _select_providers(self, t, ps, ev):
        """Return subset of providers and mapping to original indices."""
        # Always keep providers that already host some scenes
        chosen = {p for _, p in t.scene_allocation_data if p is not None}
        unassigned = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        for sid in unassigned:
            candidates = []
            for pid, prov in enumerate(ps):
                d, _ = ev.time_cost(t, sid, prov)
                if math.isfinite(d):
                    candidates.append((d, pid))
            candidates.sort(key=lambda x: x[0])
            for _, pid in candidates[: self.k]:
                chosen.add(pid)
        idx = sorted(chosen)
        subset_ps = [ps[i] for i in idx]
        mapping = {new: old for new, old in enumerate(idx)}
        return subset_ps, mapping

    # --- interface -------------------------------------------------------
    def time_complexity(self, t, ps, now, ev):
        subset, _ = self._select_providers(t, ps, ev)
        return self._cp.time_complexity(t, subset, now, ev)

    def best_combo(self, t, ps, now, ev, verbose=False):
        subset_ps, mapping = self._select_providers(t, ps, ev)
        if not subset_ps:
            return None
        res = self._cp.best_combo(t, subset_ps, now, ev, verbose)
        if res is None:
            return self._greedy.best_combo(t, ps, now, ev, verbose)
        cmb_subset, t_tot, cost = res
        # Map indices back to original provider list
        cmb_full: List[int] = []
        for p in cmb_subset:
            if p == -1:
                cmb_full.append(-1)
            else:
                cmb_full.append(mapping[p])
        return cmb_full, t_tot, cost
