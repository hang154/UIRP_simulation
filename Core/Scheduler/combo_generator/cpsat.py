# Core/Scheduler/combo_generator/cpsat.py
from __future__ import annotations
import math
import datetime as dt
from typing import List, Tuple
from ortools.sat.python import cp_model

from Core.Scheduler.interface import ComboGenerator

_SCALE = 1000
_BIG   = 10**9


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

        # Use iterative dynamic programming instead of recursion to avoid
        # hitting Python's recursion depth limit when many scenes are
        # unassigned. `dp` maps a provider usage mask to the number of ways
        # it can be formed. For each scene we either skip assigning it or
        # assign it to a feasible provider that hasn't been used yet.
        dp = {0: 1}
        for cands in feasible:
            new_dp = dict(dp)  # accounts for skipping this scene
            for mask, cnt in dp.items():
                for p in cands:
                    if mask & (1 << p) == 0:
                        new_mask = mask | (1 << p)
                        new_dp[new_mask] = new_dp.get(new_mask, 0) + cnt
            dp = new_dp
        return sum(dp.values()) - 1
    def best_combo(self, t, ps, now, ev, verbose=False):
        if verbose:
            space = self.time_complexity(t, ps, now, ev)
            print(f"[CP] search space={space}")

        m, x, y, *_rest = _build_common_model(t, ps, now)
        total_cost, makespan, over_budget, over_deadline = _rest[-4:]

        # 미배치 씬 수
        unassigned = [s for s in range(t.scene_number)
                      if t.scene_allocation_data[s][0] is None]
        n_unassigned = len(unassigned)
        if n_unassigned > 0:
            deferred = m.NewIntVar(0, n_unassigned, "deferred")
            m.Add(deferred + sum(y[s] for s in unassigned) == n_unassigned)
        else:
            deferred = m.NewIntVar(0, 0, "deferred")

        # 시간/데드라인 초과를 시간 단위로 변환
        makespan_h = m.NewIntVar(0, _BIG, "makespan_h")
        m.Add(makespan_h * 3600 == makespan)
        over_deadline_h = m.NewIntVar(0, _BIG, "over_deadline_h")
        m.Add(over_deadline_h * 3600 == over_deadline)

        wt = getattr(ev, "WT", 1.0)
        wc = getattr(ev, "WC", 1.0)
        wd = getattr(ev, "WD", 1.0)
        wb = getattr(ev, "WB", 1.0)
        wdl = getattr(ev, "WDL", 1.0)

        m.Minimize(
            int(wt * _SCALE) * makespan_h +
            int(wc * _SCALE) * total_cost +
            int(wd * _SCALE) * deferred +
            int(wb * _SCALE) * over_budget +
            int(wdl * _SCALE) * over_deadline_h
        )

        solver = cp_model.CpSolver()
        if verbose:
            solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = 10
        status = solver.Solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        cmb: List[int] = []
        for s in range(t.scene_number):
            if t.scene_allocation_data[s][0] is not None:
                cmb.append(t.scene_allocation_data[s][1])
                continue
            chosen = -1
            for p in range(len(ps)):
                if solver.BooleanValue(x[s][p]):
                    chosen = p
                    break
            cmb.append(chosen)

        ok, t_tot, cost, _, _, _ = ev.feasible(t, cmb, now, ps)
        if not ok:
            return None
        return cmb, t_tot, cost
