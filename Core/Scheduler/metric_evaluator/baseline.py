# Core/Scheduler/metric_evaluator/baseline.py
from __future__ import annotations
import math
import datetime as dt
from typing import Dict, List, Tuple
from Core.Scheduler.interface import MetricEvaluator

class BaselineEvaluator(MetricEvaluator):
    """
    - 하드 제약: 한 provider에 중복 할당을 하거나 이미 작업 중인 provider에
      할당하는 조합은 불가능
    - 소프트 제약: 예산/데드라인 초과는 허용하되 efficiency에서 패널티
    """
    def __init__(
        self,
        WT: float = 1.0,    # 시간 가중치 (hours)
        WC: float = 1.0,    # 비용 가중치 (USD)
        WD: float = 10.0,   # 연기(미배치 scene 수) 가중치
        WB: float = 200.0,  # 예산초과 패널티 ($)
        WDL: float = 500.0  # 데드라인초과 패널티 (hours)
    ):
        self._c: Dict[tuple, float] = {}
        # key: (task_id, sched_len_per_provider)
        self._spent_cache: Dict[tuple, float] = {}
        self.WT, self.WC, self.WD, self.WB, self.WDL = WT, WC, WD, WB, WDL

    # -------- 기본 시간 계산(전송+연산) 캐시 --------
    def _t_tx(self, t, s, p):
        has_global = any(rec[0] == t.id for rec in getattr(p, "schedule", []))
        k = ("tx", t.id, s, p, has_global)
        if k in self._c:
            return self._c[k]
        bw = min(t.bandwidth, p.bandwidth)
        if bw <= 0:
            v = float("inf")
        else:
            size = t.scene_size(s)
            if not has_global:
                size += t.global_file_size
            v = size / bw / 3600
        self._c[k] = v
        return v

    def _t_cmp(self, t, p):
        k = ("cmp", t.id, p)
        if k in self._c:
            return self._c[k]
        thr = getattr(p, "throughput", 0.0)
        v = float("inf") if thr <= 0 else t.scene_workload / thr
        self._c[k] = v
        return v

    def time_cost(self, t, s, p):
        if t.scene_allocation_data[s][0] is not None:
            return 0.0, 0.0
        d = self._t_tx(t, s, p) + self._t_cmp(t, p)
        return d, d * p.price_per_gpu_hour

    # -------- 메인 판정 --------
    def feasible(self, t, cmb, now: dt.datetime, ps) -> Tuple[bool, float, float, int, float, float]:
        # 이번 스텝 미배치 수
        deferred = sum(
            1 for sid, pid in enumerate(cmb)
            if (t.scene_allocation_data[sid][0] is None) and (pid == -1)
        )

        # provider별 이번 스텝 배치 목록
        group: Dict[int, List[int]] = {}
        for sid, pid in enumerate(cmb):
            if t.scene_allocation_data[sid][0] is not None or pid == -1:
                continue
            group.setdefault(pid, []).append(sid)

        # HARD: 같은 provider에 2개 이상 배정되거나 이미 작업 중이면 불가
        for pid, sids in group.items():
            if len(sids) > 1:
                return False, math.inf, math.inf, deferred, math.inf, math.inf
            prov = ps[pid]
            for rec in getattr(prov, "schedule", []):
                if len(rec) == 3:
                    _tid, st, ft = rec
                else:
                    _tid, _sid, st, ft = rec
                if st <= now < ft:
                    return False, math.inf, math.inf, deferred, math.inf, math.inf

        # 과거 같은 task의 지출 (provider 스케줄 길이 기반 캐시)
        key = (t.id, tuple(len(getattr(p, "schedule", [])) for p in ps))
        spent = self._spent_cache.get(key)
        if spent is None:
            spent = 0.0
            for prov in ps:
                for rec in getattr(prov, "schedule", []):
                    if len(rec) == 3:
                        tid, st, ft = rec
                    else:
                        tid, _sid, st, ft = rec
                    if tid == t.id:
                        spent += ((ft - st).total_seconds()/3600.0) * prov.price_per_gpu_hour
            self._spent_cache[key] = spent

        incr_cost = 0.0
        per_prov_h: Dict[int, float] = {}

        for pidx, sids in group.items():
            prov = ps[pidx]
            sid = sids[0]
            dur, c = self.time_cost(t, sid, prov)
            if not math.isfinite(dur) or dur <= 0.0:
                return False, math.inf, math.inf, deferred, math.inf, math.inf
            per_prov_h[pidx] = dur
            incr_cost += c

        # 이번 스텝 makespan: 동시 시작 가정의 max(dur_p)
        t_tot = max(per_prov_h.values(), default=0.0)
        total_cost = spent + incr_cost

        # 소프트 초과량 (패널티용)
        over_budget = max(0.0, total_cost - t.budget)
        over_deadline_h = max(0.0, (now + dt.timedelta(hours=t_tot) - t.deadline).total_seconds()/3600.0)

        return True, t_tot, total_cost, deferred, over_budget, over_deadline_h

    def efficiency(
        self, t, cmb, ps, now,
        t_tot: float, cost: float, deferred: int, over_budget: float, over_deadline_h: float
    ) -> float:
        # 작을수록 좋은 가중합을 음수화하여 최대화로 사용
        score = -(
            self.WT  * t_tot +
            self.WC  * cost +
            self.WD  * deferred +
            self.WB  * over_budget +
            self.WDL * over_deadline_h
        )
        if score != score or score in (float("-inf"), float("inf")):
            return float("-inf")
        return score
