import datetime as dt
import re
import sys
import pathlib

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Model.tasks import Tasks
from Model.providers import Providers
from Core.Scheduler.dispatcher.sequential import SequentialDispatcher
from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator


def setup_basic():
    task_data = [{
        "id": "T1",
        "scene_number": 1,
        "scene_file_size": 10.0,
        "global_file_size": 100.0,
        "scene_workload": 3600.0,
        "bandwidth": 10.0,
        "budget": 100.0,
        "start_time": dt.datetime(2024, 1, 1, 8, 0),
        "deadline": dt.datetime(2024, 1, 2, 8, 0),
    }]
    tasks = Tasks(); tasks.initialize_from_data(task_data)
    t = tasks["T1"]

    prov_data = [{
        "throughput": 3600.0,
        "price": 2.0,
        "bandwidth": 20.0,
        "available_hours": [(dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 12, 0))],
    }]
    providers = Providers(); providers.initialize_from_data(prov_data)
    p = providers[0]
    return t, p, providers


def test_verbose_assignment_logs_all_metrics(capsys):
    t, p, ps = setup_basic()
    dispatcher = SequentialDispatcher()
    ev = BaselineEvaluator()

    now = dt.datetime(2024, 1, 1, 8, 0)
    cmb = [0]

    dispatcher.dispatch(t, cmb, now, ps, ev, verbose=True)

    out = capsys.readouterr().out
    m = re.search(r"tot=([0-9.]+)h size=([0-9.]+)MB tx=([0-9.]+)h cmp=([0-9.]+)h cost=\$([0-9.]+)", out)
    assert m, out

    tot, size, tx, cmp_t, cost = map(float, m.groups())

    # expected values
    size_exp = 110.0
    tx_exp = size_exp / min(t.bandwidth, p.bandwidth) / 3600.0
    cmp_exp = t.scene_workload / p.throughput
    tot_exp = tx_exp + cmp_exp
    cost_exp = tot_exp * p.price_per_gpu_hour

    assert size == pytest.approx(size_exp)
    assert tx == pytest.approx(tx_exp, abs=1e-4)
    assert cmp_t == pytest.approx(cmp_exp, abs=1e-4)
    assert tot == pytest.approx(tot_exp, abs=1e-4)
    assert cost == pytest.approx(cost_exp, abs=1e-4)
