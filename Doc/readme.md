

## Repository Layout
```text
.
├── tasks.py          # Task & Tasks datamodels
├── providers.py      # Provider & Providers datamodels
├── utils.py          # Generic helpers (e.g., merge_intervals)
├── objective.py      # Multi‑factor objective function
├── scheduler.py      # Earliest‑Deadline‑First (EDF) scheduler
├── simulator.py      # CLI wrapper: load → schedule → evaluate
└── config.json       # Example workload/cluster definition
```
Each module has **zero external dependencies** beyond the standard library.

---
## 1. Datamodels

### `Task`
| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `scene_number` | `int` | Number of renderable *scenes* in the task |
| `scene_file_size` | `float or List[float]` | Size (GB) per scene<br/>Scalar ⇒ duplicated for every scene |
| `global_file_size` | `float` | Shared assets downloaded once per task (GB) |
| `scene_workload` | `float` | Compute workload per scene (GPU‑hours) |
| `bandwidth` | `float` | Client ↔ provider link speed (GB/hour) |
| `budget` | `float` | Max spend in USD (∞ by default) |
| `start_time` | `datetime` | Earliest moment scenes may enter the queue |
| `deadline` | `datetime` | Hard cutoff for final frame |

Internally the class maintains:
* `scene_allocation_data`: `List[Tuple[start_time?, provider_idx?]]`  
  Indexed by `scene_id`, filled in by the scheduler.

### `Tasks`
Thin container that builds a `dict[str, Task]` and proxies iteration / lookup.

---

### `Provider`
| Field | Type | Description |
|-------|------|-------------|
| `throughput` | `float` | Effective speed: *GPU‑hours per real hour* |
| `price_per_gpu_hour` | `float` | USD billed per GPU‑hour |
| `bandwidth` | `float` | Provider‑side link cap (GB/hour) |
| `available_hours` | `List[Tuple[start, end]]` | Calendar blocks the GPU is online |

Runtime‑generated:
* `schedule`: `List[(task_id, scene_id, start, finish)]`
* Methods  
  * `idle_ratio()` – share of time **not** running jobs, over a window  
  * `earliest_available(dur_h, after)` – next slot of length `dur_h`  
  * `assign()` – mutate `schedule` **and** shrink `available_hours`

### `Providers`
Ordered collection (`list`) of `Provider` instances.

---

## 2. Low‑Level Utilities

### `merge_intervals(iv: List[Tuple[dt, dt]]) -> List[Tuple[dt, dt]]`
Classic sweep that merges overlapping / touching intervals.  
Used by:
* **Provider.idle_ratio** – to union busy blocks
* **Simulator._provider_idle_ratio** – fleet‑level KPI

---

## 3. Objective Function

```python
DEFAULT_WEIGHTS = (
    a1,  # total time      (↑ bad)
    a2,  # budget penalty (↑ bad)
    a3,  # deadline overrun
    b1,  # provider profit (↓ desired → negative weight)
    b2,  # provider idle   (↑ bad)
)
```
The scalar objective for a *(task, scene, provider)* triple is:

```
J =  a1·T_total
   + a2·max(0, cost − budget)
   + a3·max(0, T_total − window)
   + b1·price_per_gpu_hour
   + b2·idle_ratio
```

Where  
* `T_total = transfer_time + compute_time`  
* `window = deadline − start_time`

The scheduler always picks the **provider with the minimum `J`.**

---

## 4. Scheduler(`scheduler.py`)

**Strategy: Earliest‑Deadline‑First at scene granularity**

1. **Feed queue** – At every simulation tick (`time_gap`, default 30 min) pull all scenes whose `task.start_time ≤ now` into an EDF‑sorted waiting list.  
2. **Per‑scene search** – For each waiting scene  
   1. compute transfer time (`T_tx`) via link bottleneck  
   2. compute compute time (`T_cmp`) via provider throughput  
   3. ask each provider for `earliest_available(T_cmp, after=now+T_tx)`  
   4. evaluate objective `J`; keep the best tuple  
3. **Commit** – Call `Provider.assign()` which locks the time block and updates its availability list.  
4. **Repeat** until either all scenes scheduled or `time_end` reached.

Guarantees:
* A provider runs **one job at a time** (`assign` enforces exclusivity)
* Scenes never start before files are fully transferred
* Budget/deadline violations are tracked for metrics but *not* blocked (tunable).

---

## 5. Simulator(`simulator.py`)

```text
config.json
├─ tasks      # ⇢ Tasks.initialize_from_data()
└─ providers  # ⇢ Providers.initialize_from_data()
```

1. Load config ⇒ build objects  
2. Invoke a `Scheduler` instance  
3. Gather `results: List[(task_id, scene_id, start, finish, provider_idx)]`  
4. Compute KPIs:
   * Per‑task: completion time, budget‑hit, deadline‑hit  
   * Global: makespan, provider idle ratio, total cost, Σ objective

Run via:

```bash
python simulator.py path/to/config.json
```

---

## 6. Extending the Framework

* **Custom objective** – drop‑in replacement for `calc_objective`.  
* **New scheduler** – implement `.run(tasks, providers, …)` with same signature.  
* **Heterogeneous resources** – add fields (e.g., GPU‑mem) to `Provider` and reference them in both the objective and `earliest_available`.  
* **Stochastic failures** – inject downtime by mutating `available_hours` at runtime.

---

## 7. Minimal Example

```python
from Model.tasks import Tasks
from Model.providers import Providers
from Core.scheduler import Scheduler
import json

cfg = json.loads(open("../config.json").read())
tasks = Tasks();
tasks.initialize_from_data(cfg["tasks"])
providers = Providers();
providers.initialize_from_data(cfg["providers"])

scheduler = Scheduler(time_gap=datetime.timedelta(minutes=15))
allocations = scheduler.run(tasks, providers)

for rec in allocations:
    print(rec)
```


---

*Last updated: 2025-07-17*
