# Core/Scheduler/dispatcher/sequential.py
from __future__ import annotations
import datetime as dt
from typing import List, Dict
from Core.Scheduler.interface import Dispatcher

Assignment = tuple[str, int, dt.datetime, dt.datetime, int]

class SequentialDispatcher(Dispatcher):
    def dispatch(self, t, cmb, now, ps, ev, verbose):
        """
        now 시점부터 provider별로 '순차'로 바로 실행.
        본 설계에선 한 provider당 이번 스텝에 최대 1개 씬만 오므로,
        사실상 각 provider는 now에 한 씬을 시작하게 됨.
        """
        out: List[Assignment] = []
        groups: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            if p == -1:
                continue
            if t.scene_allocation_data[sid][0] is not None:
                continue
            groups.setdefault(p, []).append(sid)

        for p, sids in groups.items():
            prov, cur = ps[p], now
            for sid in sids:
                # 기본 시간/비용 계산
                dur, cost = ev.time_cost(t, sid, prov)

                if verbose:
                    # 전송 파일 크기 (scene + optional global)
                    has_global = any(rec[0] == t.id for rec in getattr(prov, "schedule", []))
                    size = t.scene_size(sid)
                    if not has_global:
                        size += t.global_file_size

                    # 전송 및 연산 시간 (hours)
                    bw = min(t.bandwidth, prov.bandwidth)
                    tx_time = float("inf") if bw <= 0 else size / bw / 3600.0
                    thr = getattr(prov, "throughput", 0.0)
                    cmp_time = float("inf") if thr <= 0 else t.scene_workload / thr

                    total_time = tx_time + cmp_time
                    cost = total_time * prov.price_per_gpu_hour

                st = cur
                ft = st + dt.timedelta(hours=dur)
                prov.assign(t.id, sid, st, dur)
                t.scene_allocation_data[sid] = (st, p)
                out.append((t.id, sid, st, ft, p))
                if verbose:
                    print(
                        f"      scene{sid}->P{p} {st.strftime('%m-%d %H:%M')} "
                        f"tot={total_time:.4f}h size={size:.2f}MB "
                        f"tx={tx_time:.4f}h cmp={cmp_time:.4f}h cost=${cost:.4f}"
                    )
                cur = ft
        return out

class CPSatDispatcher(SequentialDispatcher):
    pass
