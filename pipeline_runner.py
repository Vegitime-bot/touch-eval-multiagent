"""
Pipeline Runner - 에이전트들이 Bash로 호출하는 계산 CLI
=========================================================
에이전트들은 이 스크립트를 통해 실제 numpy/scipy 연산을 실행합니다.

계산 결과는 pipeline_output/ 디렉토리에 파일로 저장됩니다.
에이전트들은 이 파일을 통해 서로 데이터를 공유합니다.

    DataAgent        → generate-data   → frames.npz, scenarios_meta.json
    AlgorithmAgent   → run-algorithm   → detections.json, params.json
    MetricsAgent     → compute-metrics → metrics.json

사용법:
    python3 pipeline_runner.py generate-data
    python3 pipeline_runner.py run-algorithm --threshold 55 --sigma 1.2 --min-area 4 --max-area 40
    python3 pipeline_runner.py compute-metrics
    python3 pipeline_runner.py show-status
"""
import argparse
import json
import os
import sys
import numpy as np

# 프로젝트 루트를 sys.path에 추가
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from core.frame_simulator import TouchFrameSimulator, TouchPoint, TestScenario
from core.touch_algorithm import TouchDetector, DetectedTouch
from core import metrics as metrics_module

OUTPUT_DIR = os.path.join(PROJECT_DIR, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAMES_FILE    = os.path.join(OUTPUT_DIR, "frames.npz")
SCENARIOS_FILE = os.path.join(OUTPUT_DIR, "scenarios_meta.json")
DETECTIONS_FILE = os.path.join(OUTPUT_DIR, "detections.json")
PARAMS_FILE    = os.path.join(OUTPUT_DIR, "params.json")
METRICS_FILE   = os.path.join(OUTPUT_DIR, "metrics.json")


# ── 명령 구현 ────────────────────────────────────────────────────────────────

def cmd_generate_data():
    """테스트 시나리오 생성 → frames.npz + scenarios_meta.json"""
    sim = TouchFrameSimulator()
    scenarios = sim.get_all()

    frames_dict = {}
    meta = {}

    for s in scenarios:
        frames_dict[s.name] = s.frames
        meta[s.name] = {
            "description": s.description,
            "n_frames": s.n_frames,
            "panel_h": s.panel_shape[0],
            "panel_w": s.panel_shape[1],
            # Ground truth: List[List[{x, y}]]
            "ground_truth": [
                [{"x": tp.x, "y": tp.y} for tp in frame_gt]
                for frame_gt in s.ground_truth
            ],
            "stats": {
                "max_signal": round(float(s.frames.max()), 1),
                "mean_signal": round(float(s.frames.mean()), 1),
                "bg_noise_std": round(float(s.frames[:, :2, :].std()), 1),
            }
        }

    np.savez(FRAMES_FILE, **frames_dict)
    with open(SCENARIOS_FILE, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    summary = {
        "status": "success",
        "saved_to": [FRAMES_FILE, SCENARIOS_FILE],
        "scenarios": {
            name: {
                "description": m["description"],
                "n_frames": m["n_frames"],
                "panel_size": f"{m['panel_h']}H × {m['panel_w']}W",
                "max_signal": m["stats"]["max_signal"],
                "bg_noise_std": m["stats"]["bg_noise_std"],
                "touches_per_frame": {
                    "min": min(len(g) for g in m["ground_truth"]),
                    "max": max(len(g) for g in m["ground_truth"]),
                }
            }
            for name, m in meta.items()
        }
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_run_algorithm(threshold, filter_sigma, min_area, max_area):
    """알고리즘 실행 → detections.json + params.json"""
    if not os.path.exists(FRAMES_FILE) or not os.path.exists(SCENARIOS_FILE):
        print(json.dumps({"status": "error", "message": "먼저 generate-data를 실행하세요"}))
        sys.exit(1)

    frames_data = np.load(FRAMES_FILE)
    with open(SCENARIOS_FILE) as f:
        meta = json.load(f)

    detector = TouchDetector(
        threshold=threshold,
        filter_sigma=filter_sigma,
        min_area=min_area,
        max_area=max_area
    )

    detections = {}
    stats = {}

    for name, m in meta.items():
        frames = frames_data[name]
        detected_seq = detector.detect_sequence(frames)

        # JSON 직렬화 가능한 형태로 저장
        detections[name] = [
            [{"x": d.x, "y": d.y, "confidence": round(d.confidence, 2), "area": d.area}
             for d in frame_detections]
            for frame_detections in detected_seq
        ]

        counts = [len(d) for d in detected_seq]
        stats[name] = {
            "avg_detected": round(float(np.mean(counts)), 2),
            "avg_expected": round(float(np.mean([len(g) for g in m["ground_truth"]])), 2),
            "frames_with_detection": sum(1 for c in counts if c > 0),
        }

    params = {
        "threshold": threshold,
        "filter_sigma": filter_sigma,
        "min_area": min_area,
        "max_area": max_area
    }

    with open(DETECTIONS_FILE, "w") as f:
        json.dump(detections, f, indent=2)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)

    print(json.dumps({
        "status": "success",
        "params_used": params,
        "saved_to": [DETECTIONS_FILE, PARAMS_FILE],
        "detection_stats": stats
    }, indent=2, ensure_ascii=False))


def cmd_compute_metrics():
    """성능 지표 계산 → metrics.json"""
    for f in [SCENARIOS_FILE, DETECTIONS_FILE, FRAMES_FILE]:
        if not os.path.exists(f):
            print(json.dumps({"status": "error", "message": f"파일 없음: {f}"}))
            sys.exit(1)

    frames_data = np.load(FRAMES_FILE)
    with open(SCENARIOS_FILE) as f:
        meta = json.load(f)
    with open(DETECTIONS_FILE) as f:
        detections_raw = json.load(f)

    all_metrics = {}

    for name, m in meta.items():
        # JSON에서 객체 재구성
        frames = frames_data[name]
        H, W = m["panel_h"], m["panel_w"]

        gt_seq = [
            [TouchPoint(tp["x"], tp["y"]) for tp in frame_gt]
            for frame_gt in m["ground_truth"]
        ]
        detected_seq = [
            [DetectedTouch(d["x"], d["y"], d["confidence"], d["area"])
             for d in frame_d]
            for frame_d in detections_raw[name]
        ]

        scenario = TestScenario(
            name=name,
            description=m["description"],
            frames=frames,
            ground_truth=gt_seq
        )

        result = metrics_module.compute_all(scenario, detected_seq)
        all_metrics[name] = result

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # 기준 대비 pass/fail 판정
    PASS_CRITERIA = {
        "mean_error_px": ("<", 2.0),
        "jitter_rms_px": ("<", 1.5),
        "snr_db":        (">", 20.0),
        "f1":            (">", 0.90),
    }

    summary = {}
    for name, m in all_metrics.items():
        passed = []
        failed = []
        for metric, (op, threshold) in PASS_CRITERIA.items():
            val = m.get(metric, None)
            ok = (val < threshold) if op == "<" else (val > threshold)
            (passed if ok else failed).append(f"{metric}={val}")

        summary[name] = {
            "mean_error_px": m["mean_error_px"],
            "jitter_rms_px": m["jitter_rms_px"],
            "snr_db":        m["snr_db"],
            "f1":            m["f1"],
            "total_fp":      m["total_fp"],
            "verdict":       "PASS" if not failed else "FAIL",
            "failed_metrics": failed,
        }

    print(json.dumps({
        "status": "success",
        "saved_to": METRICS_FILE,
        "results": summary
    }, indent=2, ensure_ascii=False))


def cmd_show_status():
    """현재 파이프라인 상태 확인"""
    status = {
        "generate_data": os.path.exists(SCENARIOS_FILE),
        "run_algorithm": os.path.exists(DETECTIONS_FILE),
        "compute_metrics": os.path.exists(METRICS_FILE),
    }

    metrics_summary = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            m = json.load(f)
        metrics_summary = {
            name: {"f1": v["f1"], "mean_error_px": v["mean_error_px"]}
            for name, v in m.items()
        }

    params = None
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            params = json.load(f)

    print(json.dumps({
        "completed_steps": status,
        "current_params": params,
        "metrics_summary": metrics_summary,
    }, indent=2, ensure_ascii=False))


# ── CLI 진입점 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Touch IC Evaluation Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("generate-data", help="테스트 시나리오 생성")
    subparsers.add_parser("show-status",   help="파이프라인 상태 확인")
    subparsers.add_parser("compute-metrics", help="성능 지표 계산")

    algo_parser = subparsers.add_parser("run-algorithm", help="알고리즘 실행")
    algo_parser.add_argument("--threshold",  type=float, default=50.0)
    algo_parser.add_argument("--sigma",      type=float, default=1.0, dest="filter_sigma")
    algo_parser.add_argument("--min-area",   type=int,   default=3)
    algo_parser.add_argument("--max-area",   type=int,   default=60)

    args = parser.parse_args()

    if args.command == "generate-data":
        cmd_generate_data()
    elif args.command == "run-algorithm":
        cmd_run_algorithm(args.threshold, args.filter_sigma, args.min_area, args.max_area)
    elif args.command == "compute-metrics":
        cmd_compute_metrics()
    elif args.command == "show-status":
        cmd_show_status()
    else:
        parser.print_help()
