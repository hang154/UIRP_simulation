#!/usr/bin/env python3
"""
simulator_with_visualizer.py  ☆ 2025-07-31 rev-3
--------------------------------------------------
변경 요약
• Provider 가용 시간을 provider 비용에 비례한 회색 음영으로 시각화
  - 비용이 높을수록 짙은 회색
  - alpha=0.3 유지
• x축 padding: 전체 스팬의 5% 만큼 앞뒤로 여백 추가
• y축 padding: 상하로 0.2 단위 지정하여 첫·마지막 provider 영역이 잘리지 않음
"""

from __future__ import annotations
import json, datetime, argparse, pprint
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.scheduler import BaselineScheduler, Assignment
from Core.objective import calc_objective
from utils.utils import merge_intervals

# ---------------------------------------------------------------------------
# COLOR UTILS ----------------------------------------------------------------

_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _task_color_map(task_ids: List[str]):
    return {tid: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, tid in enumerate(task_ids)}

def _is_light(rgb_hex: str) -> bool:
    r, g, b = mcolors.to_rgb(rgb_hex)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.5

# ---------------------------------------------------------------------------
# SIMULATOR ------------------------------------------------------------------

class Simulator:
    def __init__(self, cfg_path: str):
        cfg = json.loads(Path(cfg_path).read_text())
        self.tasks = Tasks();      self.tasks.initialize_from_data(cfg["tasks"])
        self.providers = Providers(); self.providers.initialize_from_data(cfg["providers"])
        self.results: List[Assignment] = []

    def schedule(self, sch: BaselineScheduler):
        self.results = sch.run(self.tasks, self.providers)

    @staticmethod
    def _missing(task: Task) -> List[int]:
        return [i for i, (st, _) in enumerate(task.scene_allocation_data) if st is None]

    def _idle(self) -> float:
        if not self.results:
            return 1.0
        s = min(r[2] for r in self.results)
        f = max(r[3] for r in self.results)
        horizon = (f - s).total_seconds() / 3600
        prov_busy = {i: 0.0 for i in range(len(self.providers))}
        for i, p in enumerate(self.providers):
            busy = merge_intervals([(st, ft) for _, _, st, ft in p.schedule])
            prov_busy[i] = sum((b2 - b1).total_seconds() / 3600 for b1, b2 in busy)
        return max(0.0, 1.0 - sum(prov_busy.values()) / (horizon * len(self.providers)))

    def evaluate(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("schedule() 먼저 호출 필요")
        tasks_out: Dict[str, Any] = {}
        finished: List[Assignment] = []
        tot_cost = tot_obj = 0.0
        for t in self.tasks:
            rec = [r for r in self.results if r[0] == t.id]
            miss = self._missing(t)
            cost = sum((r[3] - r[2]).total_seconds() / 3600 *
                       self.providers[r[4]].price_per_gpu_hour for r in rec)
            tot_cost += cost
            if miss:
                tasks_out[t.id] = {"completed": False,
                                   "missing": miss,
                                   "cost": cost,
                                   "budget_ok": cost <= t.budget}
                continue
            starts = [r[2] for r in rec]; finishes = [r[3] for r in rec]
            for r in rec:
                tot_obj += calc_objective(t, r[1], self.providers[r[4]], scene_start=r[2])
            tasks_out[t.id] = {
                "completed": True,
                "completion_h": (max(finishes) - min(starts)).total_seconds() / 3600,
                "cost": cost,
                "budget_ok": cost <= t.budget,
                "deadline_ok": max(finishes) <= t.deadline,
            }
            finished += rec
        makespan = ((max(r[3] for r in finished) - min(r[2] for r in finished)).total_seconds() / 3600
                    if finished else 0.0)
        return {"tasks": tasks_out,
                "makespan_h": makespan,
                "overall_idle_ratio": self._idle(),
                "total_cost": tot_cost,
                "objective_sum": tot_obj}

    def visualize(self, save_path: str | None = None, show: bool = True,
                  figsize: tuple[int, int] | None = None):
        """Gantt-style visualization with provider availability shading."""
        if not self.results:
            raise RuntimeError("schedule() 먼저 호출 필요")

        # dynamic figsize
        if figsize is None:
            figsize = (14, max(3, 1 + 0.8 * len(self.providers)))

        start_min = min(r[2] for r in self.results)
        finish_max = max(r[3] for r in self.results)
        task_ids = [t.id for t in self.tasks]
        color_map = _task_color_map(task_ids)

        # 가격 기반 회색 음영 계산
        prices = [prov.price_per_gpu_hour for prov in self.providers]
        max_price = max(prices) if prices else 1

        fig, ax = plt.subplots(figsize=figsize)

                        # 1) provider availability shading limited to available hours
        for prov_idx, prov in enumerate(self.providers):
            shade_norm = prov.price_per_gpu_hour / max_price
            shade = 1 - shade_norm
            for avail_start, avail_end in prov.available_hours:
                st_num = mdates.date2num(avail_start)
                ft_num = mdates.date2num(avail_end)
                # shading
                ax.add_patch(Rectangle(
                    (st_num, prov_idx - 0.5),
                    ft_num - st_num,
                    1.0,
                    facecolor=(shade, shade, shade),
                    alpha=0.3,
                    zorder=0,
                    linewidth=0
                ))
                # availability markers: start arrow and end arrow
                ax.plot(st_num, prov_idx,
                        marker="^", color="black", markersize=6, zorder=5,
                        label="Availability Start" if prov_idx==0 else "")
                ax.plot(ft_num, prov_idx,
                        marker="v", color="black", markersize=6, zorder=5,
                        label="Availability End" if prov_idx==0 else "")

        # 2) task allocations
        for prov_idx, _ in enumerate(self.providers):
            prov_results = [r for r in self.results if r[4] == prov_idx]
            for task_id, scene_idx, st, ft, _ in prov_results:
                st_num, ft_num = mdates.date2num(st), mdates.date2num(ft)
                facecolor = color_map[task_id]
                ax.add_patch(Rectangle((st_num, prov_idx - 0.4),
                                        ft_num - st_num, 0.8,
                                        facecolor=facecolor, edgecolor="black", linewidth=0.5,
                                        alpha=0.8, zorder=2, label=task_id))
                text_color = "black" if _is_light(facecolor) else "white"
                ax.text((st_num + ft_num) / 2, prov_idx,
                        f"{task_id}\nS{scene_idx}",
                        va="center", ha="center", fontsize=7,
                        color=text_color, zorder=3)

        # axis formatting and padding
        # x-axis padding 5%
        total_span = (finish_max - start_min).total_seconds() / 3600
        pad_h = total_span * 0.05
        ax.set_xlim(mdates.date2num(start_min - datetime.timedelta(hours=pad_h)),
                    mdates.date2num(finish_max + datetime.timedelta(hours=pad_h)))
        # y-axis padding
        ax.set_ylim(-0.5 - 0.2, len(self.providers) - 1 + 0.5 + 0.2)

        ax.set_yticks(range(len(self.providers)))
        ax.set_yticklabels([f"Prov {i}" for i in range(len(self.providers))])
        ax.set_xlabel("Time"); ax.set_ylabel("Provider")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Task", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title("Schedule Visualization"); ax.grid(True, axis="x", linestyle=":", linewidth=0.5, zorder=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✔ Schedule image saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

# ---------------- CLI 테스트 러너 ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Run simulation and visualize schedule")
    pa.add_argument("--config", default="config.json", help="Config JSON path")
    pa.add_argument("--algo",   default="bf", choices=["bf", "cp"], help="Scheduler algorithm")
    pa.add_argument("--out-img", default="schedule.png", help="Output image path")
    pa.add_argument("-v", action="count", default=0, help="-v: verbose logging")
    args = pa.parse_args()

    sim = Simulator(args.config)
    sch = BaselineScheduler(
        algo=args.algo,
        verbose=(args.v >= 1),
        time_gap=datetime.timedelta(hours=1)
    )
    sim.schedule(sch)
    pprint.pprint(sim.evaluate(), sort_dicts=False)
    sim.visualize(save_path=args.out_img, show=True)
