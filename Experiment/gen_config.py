#!/usr/bin/env python3
"""
Integrated gen_config.py
------------------------
• providers  : Philly trace 기반 (throughput, bandwidth_mbps, price_per_gpu_h, available_hours)
• tasks      : train.json 영화‑신 정보 기반 (global_file_size, scene_number, scene_file_size,
                 scene_workload, bandwidth, budget, start_time, deadline)

CLI (Jupyter 호환)
------------------
python gen_config.py \
    --machines /path/cluster_machine_list \
    --jobs     /path/cluster_job_log       \
    --train    /path/train.json            \
    --out      gen_config.json             

기본 경로는 아래 상수로 지정돼 있으며, Jupyter가 붙이는 -f <kernel.json> 인수는 자동 무시된다.
"""
from __future__ import annotations
import csv, json, orjson, random, re, datetime, argparse, pathlib, math
from typing import List, Dict

# ── 기본 파일 경로 (필요 시 CLI로 override) ──────────────────────
DEFAULT_MACHINE = "./cluster_machine_list"
DEFAULT_JOBLOG   = "./cluster_job_log"
DEFAULT_TRAIN    = "./train.json"

# ── 분석 기간 (UTC) ─────────────────────────────────────────────
T0 = datetime.datetime(2017, 10, 1, 0, 0, 0)
T1 = datetime.datetime(2017, 11, 30, 23, 59, 59)
RANGE_SEC = int((T1 - T0).total_seconds())

# ── 난수 시드 ───────────────────────────────────────────────────
SEED = 20250807
rng  = random.Random(SEED)

# ── 가격표 & 상수 ───────────────────────────────────────────────
PRICE_MAP = {12: 0.35, 16: 2.48, 24: 0.50}
FPS = 30
SIZE_PER_FRAME_MB = 5 / 8 / FPS     # 5 Mbps 비트레이트 가정
WORKSET = [480, 720, 1080, 1440, 2160]

# ── 헬퍼 ────────────────────────────────────────────────────────

def norm(mid: str) -> str:
    m = re.search(r"(\d+)", mid)
    return f"m{m.group(1)}" if m else mid.strip()

# ────────────────────────────────────────────────────────────────
# Provider 파트
# ────────────────────────────────────────────────────────────────

def gpu_price(vram: int) -> float:
    return PRICE_MAP.get(vram, 0.99)

def merge(iv: List[List[float]]):
    iv.sort()
    out = []
    for s, e in iv:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def invert(merged: List[List[float]], lo: float, hi: float):
    gaps, cur = [], lo
    for s, e in merged:
        if s - cur >= 1:
            gaps.append((cur, s))
        cur = e
    if hi - cur >= 1:
        gaps.append((cur, hi))
    return gaps

def collect_busy(joblog_path: str) -> Dict[str, List[tuple[float, float]]]:
    busy: Dict[str, List[tuple[float, float]]] = {}
    with open(joblog_path, "rb") as f:
        for raw in f:
            try:
                rec = orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue
            records = rec if isinstance(rec, list) else [rec]
            for job in records:
                for att in job.get("attempts", []):
                    s, e = att.get("start_time"), att.get("end_time")
                    if not (s and e):
                        continue
                    try:
                        ts0 = datetime.datetime.fromisoformat(s).timestamp()
                        ts1 = datetime.datetime.fromisoformat(e).timestamp()
                    except ValueError:
                        continue
                    if ts1 < T0.timestamp() or ts0 > T1.timestamp():
                        continue
                    ts0, ts1 = max(ts0, T0.timestamp()), min(ts1, T1.timestamp())
                    for d in att.get("detail", []):
                        raw_mid = d.get("ip") or d.get("machine_id") or d.get("host", "")
                        if not raw_mid:
                            continue
                        mid = norm(raw_mid)
                        busy.setdefault(mid, []).append((ts0, ts1))
    return busy

def build_providers(machine_file: str, joblog_file: str):
    busy_map = collect_busy(joblog_file)
    providers = []
    with open(machine_file) as f:
        rdr = csv.reader(f, skipinitialspace=True)
        for parts in rdr:
            if len(parts) < 3:
                continue
            raw_mid, n_gpu, raw_vram = parts[:3]
            mid = norm(raw_mid)
            try:
                gpus = int(n_gpu)
                vram_gb = int(re.search(r"(\d+)", raw_vram).group(1))
            except Exception:
                continue
            merged = merge(busy_map.get(mid, []))
            idle  = invert(merged, T0.timestamp(), T1.timestamp())
            providers.append({
                "id": mid,
                "throughput": gpus * vram_gb,
                "bandwidth_mbps": round(max(50, rng.gauss(600, 180)), 1),
                "price_per_gpu_h": gpu_price(vram_gb),
                "available_hours": [
                    [datetime.datetime.utcfromtimestamp(s).isoformat(timespec="seconds"),
                     datetime.datetime.utcfromtimestamp(e).isoformat(timespec="seconds")]
                    for s, e in idle
                ]
            })
    return providers

# ────────────────────────────────────────────────────────────────
# Tasks 파트
# ────────────────────────────────────────────────────────────────

def nearest(vals: List[int], x: float):
    return min(vals, key=lambda v: abs(v - x))

def scene_sizes(total: int, trans: List[List[int]]):
    if not trans:
        return [total]
    sizes, prev = [], 0
    for end_prev, start_next in trans:
        sizes.append(end_prev - prev + 1)
        prev = start_next
    sizes.append(total - prev)
    return sizes

def budget_coef():
    base = rng.lognormvariate(-3.5, 0.6)
    if rng.random() < 0.05:  # 5% outlier
        base *= rng.uniform(4, 10)
    return max(0.001, base)

def build_tasks(train_json: str):
    raw = json.load(open(train_json, 'r'))
    tasks = []
    for vid, meta in raw.items():
        total_frames = int(meta['frame_num'])
        transitions  = meta['transitions']
        sizes        = scene_sizes(total_frames, transitions)
        scene_num    = len(sizes)

        global_mb = round(total_frames * SIZE_PER_FRAME_MB, 2)
        work      = nearest(WORKSET, rng.gauss(1080, 300))
        bw        = round(max(20, rng.gauss(150, 40)), 1)
        budget    = round(global_mb * budget_coef(), 2)

        # 시간 설정
        factor    = max(1.0, rng.gauss(1.2, 0.1))
        duration  = int(work * scene_num * factor)
        latest_start = max(0, RANGE_SEC - duration)
        start_off = rng.randint(0, latest_start)
        st = T0 + datetime.timedelta(seconds=start_off)
        dl = st + datetime.timedelta(seconds=duration)
        if dl > T1:
            dl = T1

        tasks.append({
            "id": vid,
            "global_file_size": global_mb,
            "scene_number": scene_num,
            "scene_file_size": sizes,
            "scene_workload": work,
            "bandwidth": bw,
            "budget": budget,
            "start_time": st.isoformat(timespec="seconds"),
            "deadline":   dl.isoformat(timespec="seconds")
        })
    return tasks

# ────────────────────────────────────────────────────────────────
# Main (Jupyter 호환)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--machines", default=DEFAULT_MACHINE)
    ap.add_argument("--jobs",     default=DEFAULT_JOBLOG)
    ap.add_argument("--train",    default=DEFAULT_TRAIN)
    ap.add_argument("--seed",     type=int, default=SEED)
    ap.add_argument("--out",      default="config.json")
    args, _ = ap.parse_known_args()

    rng.seed(args.seed)

    providers = build_providers(args.machines, args.jobs)
    tasks     = build_tasks(args.train)

    pathlib.Path(args.out).write_text(
        json.dumps({"providers": providers, "tasks": tasks}, indent=2, ensure_ascii=False)
    )
    print(f"✔ {args.out}  (providers={len(providers)}, tasks={len(tasks)})")
