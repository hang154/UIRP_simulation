# Core/Scheduler/combo_generator/brute_force.py
from __future__ import annotations
import math
from functools import reduce
from operator import mul
from Core.Scheduler.interface import ComboGenerator

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable=None, **kwargs):
        return iterable

class BruteForceGenerator(ComboGenerator):
    """
    한 타임스텝에서 'provider당 최대 1개 씬'을 하드 제약.
    씬별로 시간 짧은 상위 k 후보 + 연기(-1)를 조합해 탐색.
    """
    def __init__(self, kprov: int = 3):
        self.kprov = kprov

    def _best_providers(self, t, ps, sid, ev, kprov=3):
        cand = []
        for p_idx, prov in enumerate(ps):
            d, _ = ev.time_cost(t, sid, prov)
            if math.isfinite(d) and d > 0:
                cand.append((d, p_idx))
        cand.sort(key=lambda x: x[0])
        return [p for _, p in cand[:max(1, min(kprov, len(cand)))]]

    def time_complexity(self, t, ps, now, ev):
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return 0
        kprov = min(self.kprov, len(ps))
        feasible: list[list[int]] = []
        for sid in scene_ids:
            cands = self._best_providers(t, ps, sid, ev, kprov=kprov)
            feasible.append(cands)

        # Iterative dynamic programming to count assignments while enforcing
        # the "one scene per provider" constraint. ``dp`` maps a bitmask of
        # used providers to the number of ways it can be achieved. For each
        # scene we either skip it or assign it to a provider that has not been
        # used yet. The all-skip configuration is removed at the end.
        dp = {0: 1}
        for cands in feasible:
            new_dp = dict(dp)  # skipping this scene
            for mask, cnt in dp.items():
                for p in cands:
                    if mask & (1 << p) == 0:
                        new_mask = mask | (1 << p)
                        new_dp[new_mask] = new_dp.get(new_mask, 0) + cnt
            dp = new_dp
        return sum(dp.values()) - 1

    def best_combo(self, t, ps, now, ev, verbose=False):
        # 미배정 씬만
        scene_ids = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        if not scene_ids:
            return None

        # 씬별 후보: 시간 짧은 상위 k + skip(-1)
        kprov = min(self.kprov, len(ps))
        cand_lists = []
        for sid in scene_ids:
            cands = self._best_providers(t, ps, sid, ev, kprov=kprov)
            cands.append(-1)  # 연기 옵션
            cand_lists.append((sid, cands))

        # Total number of unique combinations (excluding all-skip) for progress
        # reporting. ``reduce`` is used for the original cartesian product to
        # keep backward compatible iteration count in verbose mode.
        if verbose:
            iter_total = self.time_complexity(t, ps, now, ev)
            prod_total = reduce(mul, (len(c[1]) for c in cand_lists), 1)
            print(f"[BF] search space={iter_total} (iterations={prod_total})")
        else:
            iter_total = None

        best_score = float("-inf")
        best_res = None

        def generate(idx: int, used: set[int], cmb: list[int]):
            if idx == len(cand_lists):
                if any(pid != -1 for pid in cmb):
                    yield cmb.copy()
                return
            sid, candidates = cand_lists[idx]
            for pid in candidates:
                if pid != -1 and pid in used:
                    continue
                cmb[sid] = pid
                if pid != -1:
                    used.add(pid)
                yield from generate(idx + 1, used, cmb)
                if pid != -1:
                    used.remove(pid)
                cmb[sid] = -1

        iterator = generate(0, set(), [-1] * t.scene_number)
        if iter_total is not None:
            iterator = tqdm(iterator, total=iter_total, disable=not verbose)
        for cmb in iterator:
            ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
            if not ok:
                continue
            score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
            if score > best_score:
                best_score = score
                best_res = (cmb, t_tot, cost)

        return best_res
