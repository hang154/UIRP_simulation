# Core/Scheduler/combo_generator/cpsat.py
from __future__ import annotations
import math
import datetime as dt
from typing import List, Tuple
from ortools.sat.python import cp_model

from Core.Scheduler.interface import ComboGenerator

_SCALE = 1000
_BIG   = 10**9

# weights: (time, budget_penalty, deadline_penalty, provider_price, idle)
DEFAULT_WEIGHTS = (1.0, 200.0, 500.0, 1.0, 0.0)


def _cap_now_hours_from_avail(prov, now: dt.datetime) -> float:
    """Length of current available window starting at now (hours)."""
    for s, e in getattr(prov, "available_hours", []):
        if s <= now < e:
            return max(0.0, (e - now).total_seconds() / 3600.0)
    return 0.0


def _build_common_model(t, ps, now):
    S, P = t.scene_number, len(ps)
    TOT  = [[0.0]*P for _ in range(S)]
    COST = [[0.0]*P for _ in range(S)]
    PROF = [[0.0]*P for _ in range(S)]

    cap_hours = [_cap_now_hours_from_avail(prov, now) for prov in ps]
    for s in range(S):
        for p in range(P):
            prov = ps[p]
            bw   = min(t.bandwidth, prov.bandwidth)
            thr  = getattr(prov, "throughput", 0.0)
            if bw <= 0 or thr <= 0:
                tx = cmp = float("inf")
            else:
                has_global = any(rec[0] == t.id for rec in getattr(prov, "schedule", []))
                size = t.scene_size(s)
                if not has_global:
                    size += t.global_file_size
                tx = size / bw / 3600.0
                cmp = t.scene_workload / thr
            tot = tx + cmp
            if tot - 1e-9 > cap_hours[p]:
                tot = float("inf")
            TOT[s][p]  = tot
            COST[s][p] = tot * prov.price_per_gpu_hour if math.isfinite(tot) else float("inf")
            PROF[s][p] = prov.price_per_gpu_hour

    # 이미 사용한 비용(부분 배정)
    spent = 0.0
    for prov in ps:
        for rec in getattr(prov, "schedule", []):
            if len(rec) == 3:
                tid, st, ft = rec
            else:
                tid, _, st, ft = rec
            if tid == t.id:
                spent += ((ft - st).total_seconds()/3600.0) * prov.price_per_gpu_hour
    remaining_budget = max(0.0, t.budget - spent)
    window_sec = int((t.deadline - now).total_seconds() * _SCALE)

    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
    y = [m.NewBoolVar(f"y{s}") for s in range(S)]  # 이번 스텝 배치 여부

    # 한 씬: 0개(미배치) 또는 1개 provider
    for s in range(S):
        m.Add(sum(x[s][p] for p in range(P)) == y[s])

    # provider당 1개 씬 제한
    for p in range(P):
        m.Add(sum(x[s][p] for s in range(S)) <= 1)

    # 불가능 금지 + 이미 배정된 씬 고정
    for s in range(S):
        assigned = (t.scene_allocation_data[s][0] is not None)
        fixed_p = t.scene_allocation_data[s][1] if assigned else None
        for p in range(P):
            if not math.isfinite(TOT[s][p]):
                m.Add(x[s][p] == 0)
        if assigned:
            m.Add(y[s] == 1)
            for p in range(P):
                m.Add(x[s][p] == (1 if p == fixed_p else 0))
            for p in range(P):
                TOT[s][p]  = 0.0 if p == fixed_p else float("inf")
                COST[s][p] = 0.0 if p == fixed_p else float("inf")
                PROF[s][p] = 0.0

    # provider 시간/비용/윈도우
    tot_int  = [[int(TOT[s][p]*3600*_SCALE) if math.isfinite(TOT[s][p]) else _BIG for p in range(P)] for s in range(S)]
    cost_int = [[int(COST[s][p]*_SCALE)     if math.isfinite(COST[s][p]) else _BIG for p in range(P)] for s in range(S)]
    prof_int = [[int(PROF[s][p]*_SCALE)     for p in range(P)] for s in range(S)]

    prov_time = [m.NewIntVar(0, _BIG, f"time_p{p}") for p in range(P)]
    for p in range(P):
        m.Add(prov_time[p] == sum(tot_int[s][p] * x[s][p] for s in range(S)))
    makespan = m.NewIntVar(0, _BIG, "makespan")
    m.AddMaxEquality(makespan, prov_time)

    total_cost = m.NewIntVar(0, _BIG, "total_cost")
    m.Add(total_cost == sum(cost_int[s][p] * x[s][p] for s in range(S) for p in range(P)))
    over_budget = m.NewIntVar(0, _BIG, "over_budget")
    m.Add(over_budget >= total_cost - int(remaining_budget * _SCALE))
    over_deadline = m.NewIntVar(0, _BIG, "over_deadline")
    m.Add(over_deadline >= makespan - window_sec)

    # 가능하면 최소 1개 이상 배치
    unassigned = [s for s in range(S) if t.scene_allocation_data[s][0] is None]
    has_any_finite = any(tot_int[s][p] < _BIG for s in unassigned for p in range(P))
    if has_any_finite:
        m.Add(sum(y[s] for s in unassigned) >= 1)

    return m, x, y, tot_int, cost_int, prof_int, total_cost, makespan, over_budget, over_deadline

class CPSatComboGenerator(ComboGenerator):
    def time_complexity(self, t, ps, now, ev):
        unassigned = [i for i, (st, _) in enumerate(t.scene_allocation_data) if st is None]
        feasible = []
        for sid in unassigned:
            cands = []
            for p_idx, prov in enumerate(ps):
                d, _ = ev.time_cost(t, sid, prov)
                if math.isfinite(d) and d > 0 and _cap_now_hours_from_avail(prov, now) >= d:
                    cands.append(p_idx)
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

        return dfs(0, 0) - 1
    def best_combo(self, t, ps, now, ev, verbose=False):
        if verbose:
            space = self.time_complexity(t, ps, now, ev)
            print(f"[CP] search space={space}")
        a1, a2, a3, b1, _ = DEFAULT_WEIGHTS

        # 공통 제약 구성
        m1, x1, y1, *_ = _build_common_model(t, ps, now)

        # 1단계: 배치 수 최대화
        m1.Maximize(sum(y1[s] for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        solver1 = cp_model.CpSolver()
        if verbose:
            solver1.parameters.log_search_progress = True
        solver1.parameters.max_time_in_seconds = 10
        status1 = solver1.Solve(m1)
        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        y_opt = int(sum(solver1.Value(y1[s]) for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        if y_opt == 0:
            return None  # 이번 스텝 배치 불가

        # 2단계: 배치 수를 y_opt 이상으로 고정하고, 시간/비용 패널티 최소화
        m2, x2, y2, tot_int2, cost_int2, prof_int2, total_cost2, makespan2, overB2, overDL2 = _build_common_model(t, ps, now)
        m2.Add(sum(y2[s] for s in range(t.scene_number)
                   if t.scene_allocation_data[s][0] is None) >= y_opt)

        prof_sum = sum(prof_int2[s][p] * x2[s][p] for s in range(t.scene_number) for p in range(len(ps)))
        m2.Minimize(int(a1*_SCALE) * makespan2 +
                    int(a2*_SCALE) * overB2 +
                    int(a3*_SCALE) * overDL2 +
                    int(b1*_SCALE) * prof_sum)

        solver2 = cp_model.CpSolver()
        if verbose:
            solver2.parameters.log_search_progress = True
        solver2.parameters.max_time_in_seconds = 10
        status2 = solver2.Solve(m2)
        if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # 해 추출
        cmb: List[int] = []
        for s in range(t.scene_number):
            if t.scene_allocation_data[s][0] is not None:
                cmb.append(t.scene_allocation_data[s][1])
                continue
            if solver2.BooleanValue(y2[s]) == 0:
                cmb.append(-1)
                continue
            chosen = None
            for p in range(len(ps)):
                if solver2.BooleanValue(x2[s][p]):
                    chosen = p
                    break
            cmb.append(chosen if chosen is not None else -1)

        ok, t_tot, cost, _, _, _ = ev.feasible(t, cmb, now, ps)
        if not ok:
            return None
        return cmb, t_tot, cost
