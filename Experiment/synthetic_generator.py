# synthetic_generator.py (minimal, copulas fallback)
"""
SDV/Copulas 기반 파라미터 생성기 ― *gen_config 의존형* (불필요 기능 제거)
===============================================================
- **gen_config.py** 에 정의된 상수/Provider 클래스를 그대로 임포트해 사용
- `generate_config()` 시그니처와 반환형(DataFrame 두 개)만 유지
- 필수 기능: 합성 데이터(작업) 생성, clamp·타임슬롯·카테고리 제약 준수, provider 목록 생성

의존: `pip install pandas numpy copulas` (SDV 설치 시 SDV 의 GaussianCopula 우선 사용)
"""
from __future__ import annotations

import datetime
import importlib
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# GaussianCopula: SDV 우선, 없으면 copulas.Multivariate 사용
try:
    from sdv.tabular import GaussianCopula
except ImportError:
    try:
        from sdv.single_table import GaussianCopula
    except ImportError:
        from copulas.multivariate import GaussianMultivariate as GaussianCopula

# -----------------------------------------------------------------
# gen_config.py 로부터 상수와 Provider 클래스를 가져온다
# -----------------------------------------------------------------
base = importlib.import_module("gen_config")
SEED_TASKS: pd.DataFrame = getattr(base, "SEED_TASKS")
SEED_PROVIDERS: pd.DataFrame = getattr(base, "SEED_PROVIDERS")
CLAMPS: Dict[str, Tuple[int, int]] = getattr(base, "CLAMPS")
TASK_TYPES = getattr(base, "TASK_TYPES")
WORK_START_MIN = getattr(base, "WORK_START_MIN")
WORK_END_MIN = getattr(base, "WORK_END_MIN")
Provider = getattr(base, "Provider", None)

# -----------------------------------------------------------------
# 모델 학습 & 샘플링 유틸리티
# -----------------------------------------------------------------

def _train_model(seed_df: pd.DataFrame) -> GaussianCopula:
    model = GaussianCopula()
    model.fit(seed_df)
    return model


def _apply_clamps(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in CLAMPS.items():
        if col in df:
            df[col] = df[col].clip(lo, hi)
    return df


def _apply_time_slot(df: pd.DataFrame) -> pd.DataFrame:
    span = WORK_END_MIN - WORK_START_MIN
    def mapper(x):
        return WORK_START_MIN + (int(x) % span)
    if "earliest_start" in df:
        df["earliest_start"] = df["earliest_start"].apply(mapper)
    return df


def _postprocess_tasks(df: pd.DataFrame) -> pd.DataFrame:
    df = _apply_clamps(df)
    df = _apply_time_slot(df)
    df["task_type"] = df["task_type"].apply(
        lambda t: t if t in TASK_TYPES else random.choice(TASK_TYPES)
    )
    return df


def _generate_tasks(n: int, seed: int | None = None) -> pd.DataFrame:
    rnd = seed or int(datetime.datetime.now().timestamp())
    np.random.seed(rnd)
    random.seed(rnd)

    model = _train_model(SEED_TASKS)
    sampled = model.sample(n)
    tasks = _postprocess_tasks(sampled)
    tasks.insert(0, "task_id", range(1, len(tasks) + 1))
    return tasks

# -----------------------------------------------------------------
# provider 생성: seed 복제 + 노이즈(필요 시)
# -----------------------------------------------------------------

def _generate_providers(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    if n is None or n <= len(SEED_PROVIDERS):
        return SEED_PROVIDERS.iloc[: (n or len(SEED_PROVIDERS))].reset_index(drop=True)
    rnd = seed or int(datetime.datetime.now().timestamp())
    np.random.seed(rnd)
    random.seed(rnd)
    extras = []
    for i in range(n - len(SEED_PROVIDERS)):
        row = SEED_PROVIDERS.iloc[i % len(SEED_PROVIDERS)].copy()
        row["name"] = f"auto-{i}"
        noise = 1 + np.random.uniform(-0.25, 0.25)
        row["cpu_cap"] = max(1, int(row["cpu_cap"] * noise))
        row["mem_cap"] = max(128, int(row["mem_cap"] * noise))
        extras.append(row)
    return pd.concat([SEED_PROVIDERS, pd.DataFrame(extras)], ignore_index=True)

# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

def generate_config(
    n_tasks: int = 100,
    n_providers: int | None = None,
    *,
    seed: int | None = None,
):
    """작업과 provider DataFrame 반환"""
    tasks_df = _generate_tasks(n_tasks, seed)
    providers_df = _generate_providers(n_providers, seed)
    return tasks_df, providers_df

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# synthetic_generator.py는 gen_config.generate_cfg를 래핑합니다
# -----------------------------------------------------------------
from gen_config import generate_cfg as base_generate_cfg

if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Gen and write config.json (delegated to gen_config)")
    ap.add_argument("--tasks", type=int, default=5, help="number of tasks")
    ap.add_argument("--providers", type=int, default=3, help="number of providers")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", type=str, default="config.json", help="output path")
    args = ap.parse_args()
    # gen_config.generate_cfg signature: (n_tasks, n_prov, seed, out_path)
    base_generate_cfg(args.tasks, args.providers, args.seed, args.out)
    print(f"✔ {args.out} 생성 완료")
