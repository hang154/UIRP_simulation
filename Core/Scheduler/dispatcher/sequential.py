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
                dur, _ = ev.time_cost(t, sid, prov)
                st = cur
                ft = st + dt.timedelta(hours=dur)
                prov.assign(t.id, sid, st, dur)
                t.scene_allocation_data[sid] = (st, p)
                out.append((t.id, sid, st, ft, p))
                if verbose:
                    print(f"      scene{sid}->P{p} {st.strftime('%m-%d %H:%M')} dur={dur:.2f}")
                cur = ft
        return out

class CPSatDispatcher(SequentialDispatcher):
    pass
