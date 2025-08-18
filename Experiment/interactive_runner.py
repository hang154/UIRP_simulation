from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import Simulator
from Core.scheduler import BaselineScheduler


def _prompt(msg: str, default: str) -> str:
    """Prompt the user and return the entered value or default."""
    entered = input(f"{msg} [{default}]: ").strip()
    return entered or default


def main() -> None:
    cfg = _prompt("Path to config", "config.json")
    algo = _prompt("Algorithm (bf/cp)", "bf")
    out_img = _prompt("Output image filename", "schedule.png")
    v_str = _prompt("Verbosity level (0/1/2)", "0")
    try:
        v = int(v_str)
    except ValueError:
        v = 0

    sim = Simulator(cfg)
    sch = BaselineScheduler(
        algo=algo,
        verbose=v >= 1,
        time_gap=datetime.timedelta(minutes=5),
    )
    sim.schedule(sch)
    results = sim.evaluate()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of used config if available
    cfg_path = Path(cfg)
    if cfg_path.is_file():
        shutil.copy(cfg, exp_dir / cfg_path.name)

    # Save visualization
    img_path = exp_dir / out_img
    sim.visualize(save_path=str(img_path), show=False)

    # Save metadata and results
    metadata = {
        "timestamp": timestamp,
        "args": {
            "config": cfg,
            "algo": algo,
            "out_img": str(img_path),
            "verbosity": v,
        },
    }
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (exp_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"✔ Experiment saved to {exp_dir}")


if __name__ == "__main__":
    main()
