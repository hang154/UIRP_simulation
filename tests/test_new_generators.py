import datetime as dt
import random

from Model.tasks import Tasks
from Model.providers import Providers
from Core.scheduler import BaselineScheduler


def make_simple():
    task_data = [{
        "id": "T1",
        "scene_number": 2,
        "scene_file_size": [10.0, 5.0],
        "global_file_size": 0.0,
        "scene_workload": 3600.0,
        "bandwidth": 10.0,
        "budget": 100.0,
        "start_time": dt.datetime(2024, 1, 1, 8, 0),
        "deadline": dt.datetime(2024, 1, 1, 20, 0),
    }]
    tasks = Tasks(); tasks.initialize_from_data(task_data)
    prov_data = [
        {
            "throughput": 3600.0,
            "price": 2.0,
            "bandwidth": 20.0,
            "available_hours": [
                (dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 12, 0))
            ],
        },
        {
            "throughput": 3600.0,
            "price": 3.0,
            "bandwidth": 20.0,
            "available_hours": [
                (dt.datetime(2024, 1, 1, 8, 0), dt.datetime(2024, 1, 1, 12, 0))
            ],
        },
    ]
    providers = Providers(); providers.initialize_from_data(prov_data)
    return tasks, providers


@__import__('pytest').mark.parametrize(
    "algo",
    ["greedy", "monte_carlo", "anneal", "genetic"],
)
def test_scheduler_runs_with_registry_algorithms(algo):
    random.seed(0)
    tasks, providers = make_simple()
    sch = BaselineScheduler(algo=algo, time_gap=dt.timedelta(hours=1))
    assignments = sch.run(tasks, providers)
    assert assignments, f"{algo} produced no assignments"
    t = tasks["T1"]
    assert all(st is not None for st, _ in t.scene_allocation_data)
