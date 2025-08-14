#!/usr/bin/env python3
from __future__ import annotations
import json, math, argparse, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# SDV (>=1.x) API
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

# --------- 고정 상수 (학습 스크립트와 동일) ----------
HOURS_PER_WEEK = 7 * 24
FPS = 30
# gen_config에서 쓰던 비트레이트 모델: 5 Mbps -> MB/frame
SIZE_PER_FRAME_MB = 5 / 8 / FPS
WORKSET = [480, 720, 1080, 1440, 2160]

# -----------------------------------------------------
def _load_models(models_dir: Path):
    # SDV의 save()/load()는 synthesizer 클래스에서 직접 load
    prov_gc  = GaussianCopulaSynthesizer.load(str(models_dir / "providers_copula.pkl"))
    task_gan = CTGANSynthesizer.load(str(models_dir / "tasks_ctgan.pkl"))
    alpha = np.load(models_dir / "avail_alpha.npy")
    beta  = np.load(models_dir / "avail_beta.npy")
    hist  = np.load(models_dir / "tasks_start_how_hist.npy")
    dj    = json.loads((models_dir / "dirichlet.json").read_text()) if (models_dir / "dirichlet.json").exists() else {}
    meta  = json.loads((models_dir / "meta.json").read_text()) if (models_dir / "meta.json").exists() else {}
    return prov_gc, task_gan, alpha, beta, hist, dj, meta

def _to_iso(ts: dt.datetime) -> str:
    # naive ISO(초단위) — 타임존 접미사 없이 "YYYY-MM-DDTHH:MM:SS"
    return ts.replace(microsecond=0).isoformat(timespec="seconds")

def _hour_blocks_to_intervals(mask: np.ndarray, base: dt.datetime):
    """0/1 시계열(시간격자) -> ISO 구간 리스트 [[start,end], ...]"""
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i] == 1:
            j = i + 1
            while j < n and mask[j] == 1:
                j += 1
            st = base + dt.timedelta(hours=int(i))
            ed = base + dt.timedelta(hours=int(j))
            out.append([_to_iso(st), _to_iso(ed)])
            i = j
        else:
            i += 1
    return out

def _sample_provider_availability(weeks: int, alpha: np.ndarray, beta: np.ndarray, rng: np.random.Generator):
    """주차 수 만큼 provider 가용 마스크 생성 (시간단위)"""
    assert len(alpha) == HOURS_PER_WEEK and len(beta) == HOURS_PER_WEEK
    # 각 hour-of-week별 가용확률을 Beta에서 한 번 샘플 → provider 고유 주차 패턴
    p_how = rng.beta(alpha, beta)  # shape (168,)
    # weeks 만큼 반복해서 베르누이 샘플
    p = np.tile(p_how, weeks)
    return (rng.random(HOURS_PER_WEEK * weeks) < p).astype(np.uint8)

def _closest(val, candidates):
    return min(candidates, key=lambda v: abs(v - val))

def _safe_num(x, lo=None, hi=None, rnd=None):
    try:
        v = float(x)
    except Exception:
        v = np.nan
    if np.isnan(v) or np.isinf(v):
        v = 0.0
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    if rnd is not None: v = rnd(v)
    return v

def _frames_from_mb(global_mb: float) -> int:
    return int(max(1, round(global_mb / SIZE_PER_FRAME_MB)))

def _sample_scene_sizes(K: int, total_frames: int, dirichlet_map: dict[int, list] | None, rng: np.random.Generator):
    if K <= 0:
        return [total_frames]
    if dirichlet_map and int(K) in dirichlet_map:
        alpha = np.array(dirichlet_map[int(K)], dtype=float)
        if np.any(alpha <= 0): alpha = np.ones(K, dtype=float)
    else:
        alpha = np.ones(K, dtype=float)
    w = rng.dirichlet(alpha)
    # 정수합 보정
    raw = np.floor(w * total_frames).astype(int)
    diff = total_frames - int(raw.sum())
    # 남은 프레임을 큰 비율 순으로 1씩 분배
    if diff > 0:
        order = np.argsort(-w)
        raw[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(w)
        raw[order[:abs(diff)]] -= 1
    # 0 이하 방지
    raw = np.clip(raw, 1, None)
    # 총합 재보정
    shift = total_frames - int(raw.sum())
    if shift != 0:
        idx = np.argmax(raw)
        raw[idx] = max(1, raw[idx] + shift)
    return raw.astype(int).tolist()

def _sample_start_time(base: dt.datetime, weeks: int, how_hist: np.ndarray, rng: np.random.Generator) -> dt.datetime:
    prob = how_hist.astype(float)
    s = prob.sum()
    if s <= 0:
        prob = np.ones(HOURS_PER_WEEK) / HOURS_PER_WEEK
    else:
        prob = prob / s
    h = rng.choice(HOURS_PER_WEEK, p=prob)
    w = rng.integers(0, weeks) if weeks > 0 else 0
    return base + dt.timedelta(weeks=int(w), hours=int(h))

def _make_deadline(start: dt.datetime, scene_num: int, scene_workload: int, rng: np.random.Generator,
                   end_cap: dt.datetime | None):
    # gen_config와 같은 형태: duration_sec ≈ work * K * factor
    factor = max(1.0, rng.normal(loc=1.2, scale=0.1))
    duration_sec = int(max(60, scene_workload * max(1, scene_num) * factor))
    dl = start + dt.timedelta(seconds=duration_sec)
    if end_cap and dl > end_cap:
        dl = end_cap
    return dl

def generate_config(models_dir: str,
                    out_path: str,
                    weeks: int,
                    base_day: str | None,
                    n_providers: int | None,
                    n_tasks: int | None,
                    seed: int = 42):

    rng = np.random.default_rng(seed)
    models_dir = Path(models_dir)
    prov_gc, task_gan, alpha, beta, how_hist, dirichlet_map, meta = _load_models(models_dir)

    # 기본 수량: 학습 메타 기반
    if n_providers is None:
        n_providers = int(meta.get("providers_train_count", 500))
    if n_tasks is None:
        avg_pw = int(meta.get("tasks_avg_per_week", 400))
        n_tasks = int(max(1, weeks * avg_pw))

    # 기준 시작 시각
    if base_day:
        base = dt.datetime.fromisoformat(base_day)
    else:
        # 학습기간 직후로 기본
        base = dt.datetime(2017, 12, 1, 0, 0, 0)
    end_cap = base + dt.timedelta(weeks=weeks, seconds=-1)

    # ---------- Providers ----------
    prov_smpl = prov_gc.sample(n_providers)
    providers = []
    for i, row in prov_smpl.reset_index(drop=True).iterrows():
        throughput = int(max(1, round(_safe_num(row.get("throughput"), lo=1))))
        bandwidth  = round(_safe_num(row.get("bandwidth_mbps"), lo=10), 1)
        price_h    = round(_safe_num(row.get("price_per_gpu_h"), lo=0.01), 2)
        mask = _sample_provider_availability(weeks, alpha, beta, rng)
        intervals = _hour_blocks_to_intervals(mask, base)
        providers.append({
            "id": f"synth_m{i:04d}",
            "throughput": throughput,
            "bandwidth_mbps": bandwidth,
            "price_per_gpu_h": price_h,
            "available_hours": intervals
        })

    # ---------- Tasks ----------
    tasks_raw = task_gan.sample(n_tasks).reset_index(drop=True)

    tasks = []
    for i, row in tasks_raw.iterrows():
        # 스칼라 후처리 & 클램프
        global_mb = round(_safe_num(row.get("global_file_size"), lo=1.0), 2)
        scene_num = int(max(1, round(_safe_num(row.get("scene_number"), lo=1))))
        bw        = round(_safe_num(row.get("bandwidth"), lo=1.0), 1)
        budget    = round(_safe_num(row.get("budget"),   lo=0.01), 2)

        # scene_workload: 문자열로 나올 수 있음 → 정수화 후 근사 매핑
        sw_raw = row.get("scene_workload")
        try:
            sw_val = int(float(sw_raw))
        except Exception:
            # 혹시 "1080p" 같은 포맷이면 숫자만
            import re
            m = re.search(r"\d+", str(sw_raw))
            sw_val = int(m.group(0)) if m else 1080
        sw_val = _closest(sw_val, WORKSET)

        # scene_file_size: Dirichlet 비율로 프레임 수 분할
        total_frames = _frames_from_mb(global_mb)
        sizes = _sample_scene_sizes(scene_num, total_frames, dirichlet_map, rng)

        # 시간 샘플링 (naive ISO)
        st = _sample_start_time(base, weeks, how_hist, rng)
        dl = _make_deadline(st, scene_num, sw_val, rng, end_cap)

        tasks.append({
            "id": f"vid_{i:05d}",
            "global_file_size": global_mb,
            "scene_number": scene_num,
            "scene_file_size": sizes,
            "scene_workload": sw_val,
            "bandwidth": bw,
            "budget": budget,
            "start_time": _to_iso(st),
            "deadline":   _to_iso(dl),
        })

    out = {"providers": providers, "tasks": tasks}
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"✔ {out_path}  (providers={len(providers)}, tasks={len(tasks)}, weeks={weeks})")

# ---------------- CLI ----------------
def _parse():
    ap = argparse.ArgumentParser(
        description="Generate synthetic config.json using trained SDV models"
    )
    ap.add_argument("--models", default="synth_models", help="train_synth_models.py 출력 디렉토리")
    ap.add_argument("--out", default="config_generated.json")
    ap.add_argument("--weeks", type=int, default=2, help="생성할 주차 수")
    ap.add_argument("--base-day", type=str, default=None, help="기준 시작시각 ISO (예: 2017-12-01T00:00:00)")
    ap.add_argument("--n-providers", type=int, default=None, help="생성할 provider 수(기본: 학습 메타)")
    ap.add_argument("--n-tasks", type=int, default=None, help="생성할 task 수(기본: 주당 평균×weeks)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse()
    generate_config(models_dir=args.models,
                    out_path=args.out,
                    weeks=args.weeks,
                    base_day=args.base_day,
                    n_providers=args.n_providers,
                    n_tasks=args.n_tasks,
                    seed=args.seed)
