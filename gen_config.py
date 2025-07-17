# gen_config.py
"""
무작위 config.json 생성기
  • tasks.py 의 scene_file_size (list) 지원
CLI 예시:
  python gen_config.py --tasks 8 --providers 4 --seed 777 --out sample.json
"""
import json, random, datetime, argparse, sys
from pathlib import Path
from typing import List, Dict, Any

# 기준일
DAY0 = datetime.datetime(2025, 7, 12, 0, 0, 0)

def _g(mu: float, sigma: float, lo: float, hi: float) -> float:
    """정규분포 → [lo,hi] 클램핑"""
    return max(lo, min(hi, random.gauss(mu, sigma)))

def _rand_interval(base: datetime.datetime) -> List[str]:
    """provider 가용시간 하나 생성(ISO strings)"""
    start = base + datetime.timedelta(hours=_g(8, 2, 6, 21))
    dur   = _g(6, 2, 3, 12)
    end   = start + datetime.timedelta(hours=dur)
    return [start.isoformat(timespec="seconds"),
            end.isoformat(timespec="seconds")]

def _make_tasks(n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(1, n+1):
        # 시작 & 마감
        st = DAY0 + datetime.timedelta(hours=_g(7,3,0,23))
        dl = st  + datetime.timedelta(hours=_g(48,18,12,120))

        scene_n = int(_g(4,1.5,1,8))
        # 장면별 파일 크기 리스트
        base_sz = _g(150,40,20,400)
        scene_sizes = [round(_g(base_sz,15,10,600),1) for _ in range(scene_n)]

        t = {
            "id":                f"task_{i}",
            "global_file_size":  round(_g(600,200,50,1500),1),
            "scene_number":      scene_n,
            "scene_file_size":   scene_sizes,               # 리스트로 전달
            "scene_workload":    round(_g(200,60,30,500),1),
            "bandwidth":         round(_g(80,25,10,400),1),
            "budget":            round(_g(800,300,100,5000),1),
            "start_time":        st.isoformat(timespec="seconds"),
            "deadline":          dl.isoformat(timespec="seconds"),
        }
        out.append(t)
    return out

def _make_providers(n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(1, n+1):
        iv = [
            _rand_interval(DAY0),
            _rand_interval(DAY0 + datetime.timedelta(days=1))
        ]
        p = {
            "throughput":       round(_g(35,12,5,120),1),
            "available_hours":  iv,
            "price":            round(_g(5,1.2,1,20),2),
            "bandwidth":        round(_g(100,30,20,800),1),
        }
        out.append(p)
    return out

def generate_cfg(
    n_tasks: int,
    n_providers: int,
    seed: int,
    filepath: str = "config.json"
) -> None:
    random.seed(seed)
    cfg = {
        "tasks":     _make_tasks(n_tasks),
        "providers": _make_providers(n_providers),
    }
    Path(filepath).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✔ 생성 완료 → {filepath}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="랜덤 config.json 생성기")
    ap.add_argument("--tasks",     type=int, default=5)
    ap.add_argument("--providers", type=int, default=3)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--out",       type=str, default="config.json")
    args, _ = ap.parse_known_args(sys.argv[1:])   # Jupyter -f 무시
    generate_cfg(args.tasks, args.providers, args.seed, args.out)
