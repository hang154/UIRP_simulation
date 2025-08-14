# Core/Scheduler/combo_generator/brute_force.py
from __future__ import annotations
import math
from itertools import product
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
        feasible = []
        for sid in scene_ids:
            cands = self._best_providers(t, ps, sid, ev, kprov=kprov)
            feasible.append(cands)

        from functools import lru_cache

        @lru_cache(None)
        def dfs(i, mask):
            if i == len(feasible):
                return 1
            total = dfs(i + 1, mask)  # skip
            for p in feasible[i]:
                if mask & (1 << p) == 0:
                    total += dfs(i + 1, mask | (1 << p))
            return total

        return dfs(0, 0) - 1  # exclude all-skip

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

        if verbose:
            iter_total = reduce(mul, (len(c[1]) for c in cand_lists), 1)
            search_space = self.time_complexity(t, ps, now, ev)
            print(f"[BF] search space={search_space} (iterations={iter_total})")
        else:
            iter_total = None

        best = (float("-inf"), None)
        iterator = product(*[cl[1] for cl in cand_lists])
        if iter_total is not None:
            iterator = tqdm(iterator, total=iter_total, disable=not verbose)
        for picks in iterator:
            # 전량 연기 제외
            if not any(pid != -1 for pid in picks):
                continue
            # HARD: provider 중복 배정 금지
            used = [pid for pid in picks if pid != -1]
            if len(set(used)) != len(used):
                continue

            cmb = [-1] * t.scene_number
            for (sid, _), pid in zip(cand_lists, picks):
                cmb[sid] = pid

            ok, t_tot, cost, deferred, overB, overDL = ev.feasible(t, cmb, now, ps)
            if not ok:
                continue
            score = ev.efficiency(t, cmb, ps, now, t_tot, cost, deferred, overB, overDL)
            if score > best[0]:
                best = (score, (cmb, t_tot, cost))

        return None if best[1] is None else best[1]
