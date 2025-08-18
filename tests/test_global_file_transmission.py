import datetime as dt
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Model.tasks import Tasks
from Model.providers import Providers
from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
try:
    from Core.Scheduler.combo_generator import cpsat
except ModuleNotFoundError:  # pragma: no cover - ortools missing
    cpsat = None


def setup_task_provider():
    task_data = [{
        "id": "T1",
        "scene_number": 2,
        "scene_file_size": 10.0,
        "global_file_size": 100.0,
        "scene_workload": 0.0,
        "bandwidth": 10.0,
        "budget": 100.0,
        "start_time": dt.datetime(2024, 1, 1, 8, 0),
        "deadline": dt.datetime(2024, 1, 1, 12, 0),
    }]
    tasks = Tasks(); tasks.initialize_from_data(task_data)
    t = tasks["T1"]

    prov_data = [{
        "throughput": 1.0,
        "price": 1.0,
        "bandwidth": 10.0,
        "available_hours": [(dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 12, 0))],
    }]
    providers = Providers(); providers.initialize_from_data(prov_data)
    p = providers[0]
    return t, p, providers


def test_baseline_time_cost_skips_global_file_after_first_scene():
    t, p, _ = setup_task_provider()
    ev = BaselineEvaluator()

    d1, _ = ev.time_cost(t, 0, p)
    # 110MB total / 10MB/s = 11s => hours
    assert d1 == pytest.approx(110.0 / 10.0 / 3600.0)

    # First scene scheduled on provider -> global file already transmitted
    p.schedule.append((t.id, 0, dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 9, 0)))
    ev2 = BaselineEvaluator()
    d2, _ = ev2.time_cost(t, 1, p)
    assert d2 == pytest.approx(10.0 / 10.0 / 3600.0)


@pytest.mark.skipif(cpsat is None, reason="ortools not installed")
def test_cpsat_model_skips_global_file_after_first_scene():
    t, p, ps = setup_task_provider()
    now = dt.datetime(2024, 1, 1, 9, 0)
    _, _, _, tot_int, *_ = cpsat._build_common_model(t, ps, now)
    tot_scene0 = tot_int[0][0] / (3600 * cpsat._SCALE)
    assert tot_scene0 == pytest.approx(110.0 / 10.0 / 3600.0)

    p.schedule.append((t.id, 0, dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 9, 0)))
    _, _, _, tot_int2, *_ = cpsat._build_common_model(t, ps, now)
    tot_scene1 = tot_int2[1][0] / (3600 * cpsat._SCALE)
    assert tot_scene1 == pytest.approx(10.0 / 10.0 / 3600.0)
