#!/usr/bin/env python3
from __future__ import annotations
import json, math, argparse, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

# 학습 데이터 기간(기존 Philly 범위)
T0 = dt.datetime(2017, 10, 1, 0, 0, 0)
T1 = dt.datetime(2017, 11, 30, 23, 59, 59)
HOURS_PER_WEEK = 7 * 24  # 168

RANDOM_SEED = 20250807
np.random.seed(RANDOM_SEED)

# ---------------- Helpers ----------------
def intervals_to_hour_mask(intervals, start: T0.__class__, end: T1.__class__):
    """providers.available_hours -> 1시간 격자 가용 마스크(1=available)"""
    n_hours = int((end - start).total_seconds() // 3600)
    if n_hours <= 0:
        return np.zeros(0, dtype=np.uint8)
    mask = np.zeros(n_hours, dtype=np.uint8)
    if not intervals:
        return mask
    for s_str, e_str in intervals:
        s = dt.datetime.fromisoformat(s_str)  # naive ISO
        e = dt.datetime.fromisoformat(e_str)
        if e <= start or s >= end:
            continue
        s, e = max(s, start), min(e, end)
        if e <= s:
            continue
        s_pos = max(0, int((s - start).total_seconds() // 3600))
        e_pos = min(n_hours, int(math.ceil((e - start).total_seconds() / 3600)))
        if e_pos > s_pos:
            mask[s_pos:e_pos] = 1
    return mask

def estimate_beta_by_hour(provider_hours: pd.DataFrame):
    """각 hour-of-week(0..167)에 대해 Beta(α,β) 추정 (method-of-moments + smoothing)"""
    alpha, beta = [], []
    for h in range(HOURS_PER_WEEK):
        col = provider_hours[h].values
        mu  = float(np.clip(np.nanmean(col), 1e-4, 1 - 1e-4))
        var = float(np.nanvar(col))
        if var <= 1e-6:
            a_b = 50.0
        else:
            a_b = max(2.0, (mu * (1 - mu) / var) - 1.0)
        a = mu * a_b; b = (1 - mu) * a_b
        alpha.append(a); beta.append(b)
    return np.array(alpha), np.array(beta)

def dirichlet_fit_for_K(list_of_vectors):
    """씬 비율 벡터들의 간단한 Dirichlet 파라미터 추정(모멘트 근사)"""
    if len(list_of_vectors) == 0:
        return None
    X = np.vstack(list_of_vectors)  # N x K
    mean = X.mean(axis=0)
    v = X.var(axis=0).mean()
    c = max(1.0, (mean * (1 - mean)).mean() / max(v, 1e-6) - 1.0)
    alpha = mean * c + 1e-3
    return alpha

def coerce_non_category(df: pd.DataFrame) -> pd.DataFrame:
    """CTGAN은 pandas Categorical dtype 미지원 → object/숫자로 강제"""
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        # pandas 2.2+ 권장 체크 방식
        if isinstance(dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)
    return out

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="synth_models")
    # ---- CTGAN 하이퍼파라미터(AssertionError 회피 핵심) ----
    ap.add_argument("--ctgan-epochs", type=int, default=300)
    ap.add_argument("--ctgan-batch", type=int, default=256,
                    help="batch_size; pac의 배수여야 함(기본 pac=2 → 256 OK)")
    ap.add_argument("--ctgan-pac", type=int, default=2,
                    help="PACGAN 묶음 크기(기본 2). batch_size %% pac == 0 이어야 함.")
    args = ap.parse_args()

    if args.ctgan_batch % args.ctgan_pac != 0:
        raise SystemExit(f"--ctgan-batch({args.ctgan_batch})는 --ctgan-pac({args.ctgan_pac})의 배수여야 합니다.")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())

    # ---------- Providers (scalars) ----------
    prov = pd.DataFrame(cfg["providers"])
    prov_tab = prov[["throughput", "bandwidth_mbps", "price_per_gpu_h"]].copy()

    prov_meta = SingleTableMetadata()
    prov_meta.detect_from_dataframe(prov_tab)
    prov_gc = GaussianCopulaSynthesizer(prov_meta)
    prov_gc.fit(prov_tab)
    prov_gc.save(str(outdir / "providers_copula.pkl"))

    # ---------- Providers (availability -> Beta by hour-of-week) ----------
    rows = []
    for _, row in prov.iterrows():
        mask = intervals_to_hour_mask(row.get("available_hours", []), T0, T1)
        if mask.size == 0:
            rows.append(pd.Series([0.0] * HOURS_PER_WEEK))
            continue
        df = pd.DataFrame({
            "t": pd.date_range(T0, periods=len(mask), freq="h"),
            "avail": mask
        })
        df["how"] = df["t"].dt.weekday * 24 + df["t"].dt.hour
        g = df.groupby("how", as_index=True)["avail"].mean()
        rows.append(g)
    prov_how = pd.DataFrame(rows).reindex(columns=range(HOURS_PER_WEEK)).fillna(0.0)
    alpha, beta = estimate_beta_by_hour(prov_how)
    np.save(outdir / "avail_alpha.npy", alpha)
    np.save(outdir / "avail_beta.npy",  beta)

    # ---------- Tasks (scalars) ----------
    tasks = pd.DataFrame(cfg["tasks"])
    # 타입 정리(일부가 float로 들어온 경우 대비)
    tasks["scene_number"]    = tasks["scene_number"].astype(int)
    tasks["global_file_size"] = pd.to_numeric(tasks["global_file_size"], errors="coerce")
    tasks["bandwidth"]        = pd.to_numeric(tasks["bandwidth"], errors="coerce")
    tasks["budget"]           = pd.to_numeric(tasks["budget"], errors="coerce")

    tasks_tab = tasks[["global_file_size", "scene_number", "scene_workload", "bandwidth", "budget"]].copy()
    tasks_tab = coerce_non_category(tasks_tab)
    # scene_workload는 범주형으로 학습하기 위해 문자열(object)로 통일
    tasks_tab["scene_workload"] = tasks_tab["scene_workload"].astype(str)

    task_meta = SingleTableMetadata()
    task_meta.detect_from_dataframe(tasks_tab)
    # 자동감지 안정화(명시적으로 categorical로 설정)
    task_meta.update_column(column_name="scene_workload", sdtype="categorical")

    task_ctgan = CTGANSynthesizer(
        task_meta,
        epochs=args.ctgan_epochs,
        batch_size=args.ctgan_batch,
        pac=args.ctgan_pac,
        verbose=True,
    )
    task_ctgan.fit(tasks_tab)
    task_ctgan.save(str(outdir / "tasks_ctgan.pkl"))

    # ---------- Tasks start_time histogram (hour-of-week) ----------
    st = pd.to_datetime(tasks["start_time"], errors="coerce", utc=False)
    try:
        if getattr(st.dt, "tz", None) is not None:
            st = st.dt.tz_localize(None)
    except Exception:
        pass
    how = (st.dt.weekday * 24 + st.dt.hour).astype("Int64")
    hist_counts = np.bincount(how.dropna().astype(int).values, minlength=HOURS_PER_WEEK)
    np.save(outdir / "tasks_start_how_hist.npy", hist_counts)

    # ---------- scene_file_size : Dirichlet per K ----------
    dirichlet_dict = {}
    for K, grp in tasks.groupby("scene_number"):
        vecs = []
        for sizes in grp["scene_file_size"]:
            v = np.array(sizes, dtype=float)
            s = v.sum()
            if s <= 0:
                continue
            vecs.append(v / s)
        alphaK = dirichlet_fit_for_K(vecs)
        if alphaK is not None:
            dirichlet_dict[int(K)] = alphaK.tolist()
    (outdir / "dirichlet.json").write_text(json.dumps(dirichlet_dict, indent=2))

    # ---------- meta sidecar (규모/주당 평균) ----------
    train_hours = max(1, int((T1 - T0).total_seconds() // 3600))
    train_weeks = max(1, round(train_hours / HOURS_PER_WEEK))
    meta = {
        "providers_train_count": int(len(prov_tab)),
        "tasks_train_count": int(len(tasks_tab)),
        "train_weeks": int(train_weeks),
        "tasks_avg_per_week": int(round(len(tasks_tab) / train_weeks))
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"✔ Saved models to {outdir}/")

if __name__ == "__main__":
    main()
