"""
run_experiment.py — 최신 스케줄러 스택 통합 실행기
-------------------------------------------------
• gen_config로 synthetic config 생성 (선택)
• Simulator + BaselineScheduler 사용
• detailed verbose logs를 파일에 저장 (--log-file)
• 결과를 콘솔 및 JSON(옵션) 저장

업로드된 코드베이스(Core.scheduler, simulator.py 등)와 완벽 호환되도록 작성.
"""

from __future__ import annotations
import argparse, datetime, json, pprint, sys
from pathlib import Path

# config.json 에 맞춘 기본 기준일
DEFAULT_BASE_DAY = datetime.datetime(2017, 9, 30, 15)

# 프로젝트 루트 경로를 모듈 검색 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# CLI 파싱
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None):
    pa = argparse.ArgumentParser(
        description="End-to-end scheduling experiment runner (config 생성 + 시뮬 + 로그 저장)"
    )
    pa.add_argument("--config", default="config.json",
                    help="기존 config.json 경로 (생략 시 --generate 필요)")
    pa.add_argument(
        "--generate",
        action="store_true",
        help="실행 전 synthetic config 생성(gen_config) 여부",
    )
    pa.add_argument("--tasks", type=int, default=5, help="--generate 시 Task 개수")
    pa.add_argument(
        "--providers", type=int, default=3, help="--generate 시 Provider 개수"
    )
    pa.add_argument("--seed", type=int, default=42, help="난수 시드")
    pa.add_argument(
        "--base-day",
        type=str,
        default=None,
        help="기준일시 ISO (예: 2017-09-30T15:00:00)",
    )
    pa.add_argument(
        "--out-config", default="config.json", help="--generate 출력 파일명"
    )
    pa.add_argument(
        "--algo", default="bf", choices=["bf", "cp"],
        help="BaselineScheduler 알고리즘: bf (Brute Force) 또는 cp (CP-SAT)"
    )
    pa.add_argument(
        "--time-gap-min", type=int, default=5,
        help="스케줄러 ticker 시간 간격 (minutes)"
    )
    pa.add_argument(
        "--result-out", default=None, help="평가 결과 JSON 저장 경로 (선택)"
    )
    pa.add_argument(
        "--log-file", default=None,
        help="상세 verbose 로그를 저장할 파일 경로 (선택)"
    )
    pa.add_argument("-v", action="count", default=0, help="-v: step logs, -vv: detailed logs")

    args, _ = pa.parse_known_args(argv)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # 1) config 준비 ---------------------------------------------------------
    if args.generate:
        base_day = (
            datetime.datetime.fromisoformat(args.base_day)
            if args.base_day
            else DEFAULT_BASE_DAY
        )
        from gen_config import generate_cfg

        generate_cfg(
            args.tasks, args.providers, args.seed,
            args.out_config, base_day
        )
        cfg_path = args.out_config
    else:
        if not args.config:
            raise SystemExit("--config 지정이 없으면 --generate 필요")
        cfg_path = args.config

    # 2) 시뮬 + 스케줄 --------------------------------------------------------
    from simulator import Simulator
    from Core.scheduler import BaselineScheduler

    sim = Simulator(cfg_path)
    sch = BaselineScheduler(
        algo=args.algo,
        verbose=args.v,
        time_gap=datetime.timedelta(minutes=args.time_gap_min),
    )

    # 3) verbose 로그 파일 저장 설정 ----------------------------------------
    log_enabled = False
    if args.log_file and args.v >= 1:
        log_fh = open(args.log_file, "w", encoding="utf-8")
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = log_fh, log_fh
        log_enabled = True

    # schedule 호출 (verbose prints는 파일로)
    sim.schedule(sch)

    # 로그 복원 및 파일 닫기 -----------------------------------------------
    if log_enabled:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_fh.close()
        print(f"✔ Detailed logs saved to {args.log_file}")

    # 4) 결과 평가 및 출력 ---------------------------------------------------
    res = sim.evaluate()
    pprint.pprint(res, sort_dicts=False)

    # 5) 결과 JSON 저장 -----------------------------------------------------
    if args.result_out:
        Path(args.result_out).write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✔ Results written to {args.result_out}")


if __name__ == "__main__":
    main()
