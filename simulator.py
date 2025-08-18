"""Simulation runner and visualization helpers."""

from __future__ import annotations
import argparse
import datetime
import json
import pprint
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from Model.tasks import Tasks
from Model.providers import Providers
from Core.scheduler import BaselineScheduler, Assignment
from Core.Scheduler import system_evaluator

_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def _task_color_map(task_ids):
    return {tid: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, tid in enumerate(task_ids)}

def _is_light(rgb_hex):
    r, g, b = mcolors.to_rgb(rgb_hex)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.5


class Simulator:
    def __init__(self, cfg_path: str):
        cfg = json.loads(Path(cfg_path).read_text())
        self.tasks = Tasks()
        self.tasks.initialize_from_data(cfg["tasks"])
        self.providers = Providers()
        self.providers.initialize_from_data(cfg["providers"])
        self.results: List[Assignment] = []
        self.metrics: Dict[str, Any] | None = None
        # For injecting BaselineScheduler.evaluator
        self.evaluator = None

    # Run scheduler with tasks and providers
    def schedule(self, sch: BaselineScheduler):
        # Store evaluator to reuse weights in evaluate()
        self.evaluator = getattr(sch, "evaluator", None)
        self.results = sch.run(self.tasks, self.providers)
        # Evaluate system metrics immediately after scheduling
        self.metrics = system_evaluator.evaluate(self.tasks, self.providers)
        system_evaluator.print_report(self.metrics)

    def evaluate(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("schedule() must be called first")
        if self.metrics is None:
            self.metrics = system_evaluator.evaluate(self.tasks, self.providers)
        return self.metrics

    def visualize(self, save_path: str | None = None, show: bool = True,
                  figsize: tuple[int, int] | None = None):
        if not self.results:
            raise RuntimeError("schedule() must be called first")

        if figsize is None:
            figsize = (14, max(3, 1 + 0.8 * len(self.providers)))

        # --- 1. Determine absolute min/max times including all provider availability
        # Note: available_hours contains remaining intervals after assignments
        avail_points = [ts for p in self.providers for ts in p.available_hours]
        if avail_points:
            avail_min = min(s for (s, _) in avail_points)
            avail_max = max(e for (_, e) in avail_points)
        else:
            # If all availability is consumed, fallback to results
            avail_min = min(r[2] for r in self.results)
            avail_max = max(r[3] for r in self.results)

        # --- 2. Safeguard when results are empty (already checked above)
        task_min = min(r[2] for r in self.results)
        task_max = max(r[3] for r in self.results)

        start_min = min(avail_min, task_min)
        finish_max = max(avail_max, task_max)

        task_ids = [t.id for t in self.tasks]
        color_map = _task_color_map(task_ids)

        prices = [prov.price_per_gpu_hour for prov in self.providers]
        max_price = max(prices) if prices else 1.0

        fig, ax = plt.subplots(figsize=figsize)

        # Draw remaining availability windows (gray) first
        for prov_idx, prov in enumerate(self.providers):
            shade_norm = prov.price_per_gpu_hour / max_price
            shade = 1 - shade_norm
            for avail_start, avail_end in prov.available_hours:
                st_num = mdates.date2num(avail_start)
                ft_num = mdates.date2num(avail_end)
                ax.add_patch(Rectangle(
                    (st_num, prov_idx - 0.5),
                    ft_num - st_num,
                    1.0,
                    facecolor=(shade, shade, shade),
                    alpha=0.3,
                    zorder=0,
                    linewidth=0
                ))
                ax.plot(st_num, prov_idx, marker="^", color="black", markersize=6, zorder=5,
                        label="Availability Start" if prov_idx == 0 else "")
                ax.plot(ft_num, prov_idx, marker="v", color="black", markersize=6, zorder=5,
                        label="Availability End" if prov_idx == 0 else "")

        # Overlay actual assigned blocks
        for prov_idx, _ in enumerate(self.providers):
            prov_results = [r for r in self.results if r[4] == prov_idx]
            for task_id, scene_idx, st, ft, _ in prov_results:
                st_num, ft_num = mdates.date2num(st), mdates.date2num(ft)
                facecolor = color_map[task_id]
                ax.add_patch(Rectangle(
                    (st_num, prov_idx - 0.4),
                    ft_num - st_num, 0.8,
                    facecolor=facecolor, edgecolor="black", linewidth=0.5,
                    alpha=0.8, zorder=2, label=task_id
                ))
                text_color = "black" if _is_light(facecolor) else "white"
                ax.text((st_num + ft_num) / 2, prov_idx,
                        f"{task_id}\nS{scene_idx}",
                        va="center", ha="center", fontsize=7,
                        color=text_color, zorder=3)

        total_span = (finish_max - start_min).total_seconds() / 3600.0
        pad_h = total_span * 0.05
        ax.set_xlim(mdates.date2num(start_min - datetime.timedelta(hours=pad_h)),
                    mdates.date2num(finish_max + datetime.timedelta(hours=pad_h)))
        ax.set_ylim(-0.7, len(self.providers) - 1 + 0.7)

        ax.set_yticks(range(len(self.providers)))
        ax.set_yticklabels([f"Prov {i}" for i in range(len(self.providers))])
        ax.set_xlabel("Time")
        ax.set_ylabel("Provider")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Task", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title("Schedule Visualization")
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5, zorder=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✔ Schedule image saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)

# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.json")
    pa.add_argument("--algo",   default="bf", help="bf | cp")
    pa.add_argument("--out-img", default="schedule.png", help="Output image path")
    pa.add_argument("-v", action="count", default=0, help="-v: step logs, -vv: detailed logs")
    args = pa.parse_args()

    sim = Simulator(args.config)
    sch = BaselineScheduler(algo=args.algo,
                            verbose=args.v,
                            time_gap=datetime.timedelta(minutes=5))
    sim.schedule(sch)
    pprint.pprint(sim.evaluate(), sort_dicts=False)
    sim.visualize(save_path=args.out_img, show=True)
