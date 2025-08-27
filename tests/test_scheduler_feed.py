import datetime
from Core.Scheduler.scheduler import BaselineScheduler
from Model.tasks import Task

BASE = datetime.datetime(2023, 1, 1, 0, 0)

def make_task(task_id, start, deadline):
    return Task({
        "id": task_id,
        "scene_number": 1,
        "scene_file_size": 1,
        "start_time": start,
        "deadline": deadline,
    })

def test_feed_removes_expired_and_prevents_duplicates():
    scheduler = BaselineScheduler()
    expired = make_task(
        "expired",
        BASE - datetime.timedelta(minutes=10),
        BASE - datetime.timedelta(minutes=1),
    )
    active = make_task(
        "active",
        BASE - datetime.timedelta(minutes=10),
        BASE + datetime.timedelta(minutes=10),
    )
    scheduler.waiting_tasks = [expired, active]

    scheduler._feed(BASE - datetime.timedelta(minutes=5), BASE, [])
    assert scheduler.waiting_tasks == [active]

    new_task = make_task(
        "new",
        BASE + datetime.timedelta(minutes=5),
        BASE + datetime.timedelta(minutes=30),
    )
    scheduler._feed(BASE, BASE + datetime.timedelta(minutes=10), [new_task])
    assert scheduler.waiting_tasks == [active, new_task]

    scheduler.waiting_tasks.remove(new_task)
    scheduler._feed(
        BASE + datetime.timedelta(minutes=10),
        BASE + datetime.timedelta(minutes=20),
        [new_task],
    )
    assert new_task not in scheduler.waiting_tasks
