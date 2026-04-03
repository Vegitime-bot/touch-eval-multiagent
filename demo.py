#!/usr/bin/env python3
"""
Touch IC Multi-Agent Pipeline — 교육용 시각화 데모
===================================================
사용법:
    python3 demo.py          # 인터랙티브 모드 (→/n=다음  ←/p=이전  q=종료)
    python3 demo.py --auto   # 자동 재생
    python3 demo.py --fast   # 자동 재생 (빠른 속도)
"""
import argparse
import json
import os
import subprocess
import sys
import termios
import time
import tty

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "pipeline_output")
RUNNER      = os.path.join(PROJECT_DIR, "pipeline_runner.py")

# ── ANSI ────────────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD   = "\033[1m"; DIM  = "\033[2m"; BLINK = "\033[5m"
RED    = "\033[91m"; GRN  = "\033[92m"; YLW = "\033[93m"
BLU    = "\033[94m"; MAG  = "\033[95m"; CYN = "\033[96m"; WHT = "\033[97m"
BG_BLU = "\033[44m"; BG_GRN = "\033[42m"; BG_YLW = "\033[43m"
BG_RED = "\033[41m"; BG_MAG = "\033[45m"; BG_CYN = "\033[46m"

AGENT_COLORS = {
    "DataAgent":      (BLU, BG_BLU),
    "AlgorithmAgent": (YLW, BG_YLW),
    "MetricsAgent":   (GRN, BG_GRN),
    "ReportAgent":    (MAG, BG_MAG),
}
AGENTS = ["DataAgent", "AlgorithmAgent", "MetricsAgent", "ReportAgent"]
CHARS  = " .:;+=xX$#"


# ── 키보드 입력 ──────────────────────────────────────────────────────────────

def _read_key():
    """단일 키 입력 읽기 (엔터 불필요). /dev/tty + raw mode 사용."""
    fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # ESC 시퀀스 (화살표 키 등)
            try:
                ch2 = os.read(fd, 1)
                if ch2 == b"[":
                    ch3 = os.read(fd, 1)
                    return {b"C": "next", b"D": "prev",   # ← →
                            b"A": "next", b"B": "prev"}.get(ch3, "")  # ↑ ↓
            except OSError:
                pass
            return ""
        return ch.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        os.close(fd)


def wait_nav(step_idx, total, fast=False, auto_delay=2.0):
    """화살표/키 입력 대기 (엔터 불필요). fast=True 이면 자동 진행."""
    print(f"\n  {DIM}{'─'*60}{R}")
    print(f"  [{step_idx+1}/{total}]  →/n=다음  ←/p=이전  q=종료")
    sys.stdout.flush()
    if fast:
        time.sleep(auto_delay)
        return "next"
    while True:
        try:
            key = _read_key()
        except (OSError, KeyboardInterrupt):
            return "quit"
        if key in ("next", "n", " ", ""):   return "next"
        if key in ("prev", "p"):            return "prev"
        if key in ("q", "\x03", "\x04"):    return "quit"   # q / Ctrl-C / Ctrl-D


# ── 유틸 ────────────────────────────────────────────────────────────────────

def W(n=80):
    try:    return os.get_terminal_size().columns
    except: return n

def clear():
    sys.stdout.write("\033[2J\033[H"); sys.stdout.flush()

def pause(t, fast=False):
    time.sleep(t * 0.2 if fast else t)

def print_slow(text, delay=0.018, fast=False):
    if fast:
        print(text); return
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def hline(ch="─", color=DIM):
    print(f"{color}{ch * min(W(), 76)}{R}")

def run_cmd(cmd_args):
    result = subprocess.run(
        ["python3", RUNNER] + cmd_args,
        capture_output=True, text=True, cwd=PROJECT_DIR
    )
    return json.loads(result.stdout) if result.returncode == 0 else {}


# ── 파이프라인 다이어그램 ────────────────────────────────────────────────────

def draw_pipeline(active=None, done=None):
    done = done or set()
    boxes = []
    for ag in AGENTS:
        fg, bg = AGENT_COLORS[ag]
        if ag == active:
            box = f"{bg}{BOLD}{BLINK} {ag:<15}{R}"
        elif ag in done:
            box = f"{GRN}{BOLD} {ag:<15}{R} {GRN}✓{R}"
        else:
            box = f"{DIM} {ag:<15}{R}  "
        boxes.append(box)

    print()
    print("  ┌" + "─"*17 + "┐     ┌" + "─"*17 + "┐     ┌" + "─"*17 + "┐     ┌" + "─"*17 + "┐")
    print(f"  │ {boxes[0]}│ ──▶ │ {boxes[1]}│ ──▶ │ {boxes[2]}│ ──▶ │ {boxes[3]}│")
    print("  └" + "─"*17 + "┘     └" + "─"*17 + "┘     └" + "─"*17 + "┘     └" + "─"*17 + "┘")
    print(f"  {DIM}   frames.npz            detections.json       metrics.json{R}")
    print(f"  {DIM}   scenarios.json        params.json{R}")
    print()


def draw_shared_data(pipeline_data):
    sc = len(pipeline_data.get("scenarios", {}))
    de = len(pipeline_data.get("detections", {}))
    me = len(pipeline_data.get("metrics", {}))

    def indicator(n):
        if n == 0: return f"{DIM}없음{R}"
        if n == 4: return f"{GRN}{BOLD}완료 ({n}){R}"
        return f"{YLW}진행 ({n}){R}"

    print(f"  {DIM}┌─ pipeline_data (공유 저장소) {'─'*30}┐{R}")
    print(f"  {DIM}│{R}  scenarios  : {indicator(sc)}")
    print(f"  {DIM}│{R}  detections : {indicator(de)}")
    print(f"  {DIM}│{R}  metrics    : {indicator(me)}")
    print(f"  {DIM}└{'─'*46}┘{R}")
    print()


def mini_frame(frame, gt):
    H, W = frame.shape
    vmax = max(float(frame.max()), 1.0)
    lines = []
    for r in range(H):
        row = ""
        for c in range(W):
            hit = any(abs(p["y"]-r) < 0.9 and abs(p["x"]-c) < 0.9 for p in gt)
            val = frame[r, c] / vmax
            ch  = CHARS[min(int(val * len(CHARS)), len(CHARS)-1)]
            if hit:           row += f"{RED}{BOLD}★{R}"
            elif val > 0.6:   row += f"{YLW}{BOLD}{ch}{R}"
            elif val > 0.3:   row += f"{CYN}{ch}{R}"
            elif val > 0.05:  row += f"{DIM}{ch}{R}"
            else:             row += ch
        lines.append("  │ " + row + " │")
    return lines


def log_tool(tool_name, args_str):
    print(f"  {CYN}┌─ Tool 호출{R}")
    print(f"  {CYN}│{R}  {BOLD}{tool_name}{R}({DIM}{args_str}{R})")

def log_tool_result(result_str):
    print(f"  {CYN}│{R}  → {GRN}{result_str}{R}")
    print(f"  {CYN}└─ 완료{R}")
    print()


# ════════════════════════════════════════════════════════════════════════════
#  사전 계산 (한 번만 실행)
# ════════════════════════════════════════════════════════════════════════════

def precompute():
    """파이프라인 전체 실행 → 데이터 반환. 화면은 건드리지 않음."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. generate-data
    run_cmd(["generate-data"])

    frames_data = np.load(os.path.join(OUTPUT_DIR, "frames.npz"))
    with open(os.path.join(OUTPUT_DIR, "scenarios_meta.json")) as f:
        meta_raw = json.load(f)

    scenarios_summary = {}
    for name in frames_data.files:
        arr = frames_data[name]
        m   = meta_raw[name]
        scenarios_summary[name] = {
            "description": m["description"],
            "n_frames":    m["n_frames"],
            "panel_size":  f"{m['panel_h']}H × {m['panel_w']}W",
            "max_signal":  m["stats"]["max_signal"],
            "bg_noise_std": m["stats"]["bg_noise_std"],
        }

    # 2. run-algorithm
    thr, sig, mn, mx = 60, 1.2, 4, 50
    algo_result = run_cmd(["run-algorithm",
                           "--threshold", str(thr), "--sigma", str(sig),
                           "--min-area",  str(mn),  "--max-area", str(mx)])
    detection_stats = algo_result.get("detection_stats", {})
    params          = algo_result.get("params_used", {})

    # 3. compute-metrics
    run_cmd(["compute-metrics"])
    metrics_all = {}
    mpath = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            metrics_all = json.load(f)
    metrics_summary = {}
    for name, m in metrics_all.items():
        failed = []
        if m.get("mean_error_px", 99) >= 2.0: failed.append(f"error={m['mean_error_px']:.2f}")
        if m.get("jitter_rms_px", 99) >= 1.5: failed.append(f"jitter={m['jitter_rms_px']:.2f}")
        if m.get("snr_db", 0) <= 20:          failed.append(f"snr={m['snr_db']:.1f}")
        if m.get("f1", 0) <= 0.9:             failed.append(f"f1={m['f1']:.3f}")
        metrics_summary[name] = {**m, "verdict": "PASS" if not failed else "FAIL"}

    return {
        "scenarios":    scenarios_summary,
        "detections":   detection_stats,
        "params":       params,
        "metrics":      metrics_all,
        "metrics_summary": metrics_summary,
        "frames_data":  frames_data,
        "meta_raw":     meta_raw,
        "thr": thr, "sig": sig,
    }


# ════════════════════════════════════════════════════════════════════════════
#  슬라이드 렌더러 (순수 출력 함수, 상태 변경 없음)
# ════════════════════════════════════════════════════════════════════════════

DONE_BY_STEP = [
    set(),                                                  # 0: 인트로
    set(),                                                  # 1: DataAgent
    {"DataAgent"},                                          # 2: AlgorithmAgent
    {"DataAgent", "AlgorithmAgent"},                        # 3: MetricsAgent
    {"DataAgent", "AlgorithmAgent", "MetricsAgent"},        # 4: ReportAgent
    {"DataAgent", "AlgorithmAgent", "MetricsAgent", "ReportAgent"},  # 5: 아키텍처
]
ACTIVE_BY_STEP = [None, "DataAgent", "AlgorithmAgent", "MetricsAgent", "ReportAgent", None]


def header(data, step_idx):
    print(f"\n  {BOLD}Touch IC Evaluation — Multi-Agent Pipeline Demo{R}\n")
    draw_pipeline(active=ACTIVE_BY_STEP[step_idx], done=DONE_BY_STEP[step_idx])
    data_state = {
        "scenarios":  data["scenarios"]  if step_idx >= 1 else {},
        "detections": data["detections"] if step_idx >= 2 else {},
        "metrics":    data["metrics"]    if step_idx >= 3 else {},
    }
    draw_shared_data(data_state)


# ── Slide 0: 인트로 ──────────────────────────────────────────────────────────

def slide_intro(data):
    hline("═")
    print_slow(f"  {BOLD}  Touch IC Evaluation — Multi-Agent Pipeline Demo{R}", fast=True)
    print_slow(f"  {DIM}  에이전트들이 어떻게 협력하는지 단계별로 보여줍니다{R}", fast=True)
    hline("═")
    print()
    draw_pipeline()
    print(f"  {DIM}← → 키로 단계를 탐색하세요. q 로 종료.{R}")
    print()
    print(f"  {BOLD}[ 파이프라인 구성 ]{R}")
    for ag in AGENTS:
        fg, _ = AGENT_COLORS[ag]
        descs = {
            "DataAgent":      "테스트 시나리오 (프레임 데이터) 생성",
            "AlgorithmAgent": "터치 감지 알고리즘 실행 & 파라미터 결정",
            "MetricsAgent":   "accuracy / jitter / SNR / F1 계산",
            "ReportAgent":    "종합 평가 리포트 작성",
        }
        print(f"   {fg}{BOLD}{ag:<16}{R}  {DIM}{descs[ag]}{R}")
    print()
    print(f"  {DIM}합격 기준: mean_error<2.0px  jitter<1.5px  snr>20dB  f1>0.90{R}")


# ── Slide 1: DataAgent ───────────────────────────────────────────────────────

def slide_data_agent(data):
    header(data, 1)
    hline("═")
    print(f"  {BG_BLU}{BOLD}  STEP 1/4 : DataAgent  {R}")
    hline("═")
    print()
    print(f"  {BLU}역할{R}: 테스트 시나리오(프레임 데이터)를 생성합니다.")
    print(f"  {BLU}도구{R}: list_scenarios, generate_scenario")
    print()

    log_tool("list_scenarios", "")
    log_tool_result("['single_touch_drag', 'multi_touch_pinch', 'emi_noise', 'water_blob']")

    for name, info in data["scenarios"].items():
        log_tool("generate_scenario", f'"{name}"')
        log_tool_result(
            f"생성 완료 — {info['panel_size']}, "
            f"max_signal={info['max_signal']}, noise={info['bg_noise_std']}"
        )

    print(f"  {BLU}{BOLD}[ 생성된 프레임 샘플 (각 시나리오 중간 프레임) ]{R}")
    print()
    frames_data = data["frames_data"]
    meta_raw    = data["meta_raw"]
    for name in frames_data.files:
        arr   = frames_data[name]
        mid   = len(arr) // 2
        frame = arr[mid]
        gt    = meta_raw[name]["ground_truth"][mid]
        fg2, _ = AGENT_COLORS["DataAgent"]
        gt_pts = [f"({p['x']:.1f},{p['y']:.1f})" for p in gt]
        print(f"  {fg2}{BOLD}{name}{R}  {DIM}(frame {mid+1}/{len(arr)}){R}  GT: {RED}★{R}{gt_pts}")
        print("  ├" + "─"*34 + "┤")
        for line in mini_frame(frame, gt):
            print(line)
        print("  └" + "─"*34 + "┘")
        print()

    print(f"  {DIM}Claude: \"4개 시나리오 생성 완료. AlgorithmAgent에 전달합니다.\"{R}")


# ── Slide 2: AlgorithmAgent ──────────────────────────────────────────────────

def slide_algorithm_agent(data):
    header(data, 2)
    hline("═")
    print(f"  {BG_YLW}{BOLD}  STEP 2/4 : AlgorithmAgent  {R}")
    hline("═")
    print()
    print(f"  {YLW}역할{R}: 시나리오 특성을 파악하고 최적 파라미터를 선택해 알고리즘을 실행합니다.")
    print(f"  {YLW}도구{R}: get_scenario_list, run_detection(threshold, filter_sigma, min_area, max_area)")
    print()

    log_tool("get_scenario_list", "")
    log_tool_result("4개 시나리오, 각 40프레임, 패널 18H×32W")

    param_guide = [
        ("single_touch_drag", 55, 1.2, 4, 40, "단일 터치, 표준 파라미터"),
        ("multi_touch_pinch",  50, 1.0, 3, 25, "멀티터치, 작은 blob 허용"),
        ("emi_noise",          65, 1.5, 5, 35, "EMI 환경, threshold 상향"),
        ("water_blob",         65, 1.2, 5, 40, "수분 오탐 방지, threshold 상향"),
    ]
    print(f"  {YLW}{BOLD}[ 파라미터 결정 근거 ]{R}")
    print(f"  {'시나리오':<22} {'thr':>4} {'σ':>5} {'min':>4} {'max':>4}  판단 근거")
    print(f"  {'─'*22} {'─'*4} {'─'*5} {'─'*4} {'─'*4}  {'─'*20}")
    for name, thr, sig, mn, mx, reason in param_guide:
        print(f"  {DIM}{name:<22}{R} {YLW}{thr:>4}{R} {YLW}{sig:>5}{R} {DIM}{mn:>4} {mx:>4}{R}  {DIM}{reason}{R}")
    print()

    thr = data["thr"]; sig = data["sig"]
    log_tool("run_detection",
             f"threshold={thr}, filter_sigma={sig}, min_area=4, max_area=50")

    stats = data["detections"]
    print(f"  {YLW}→ 검출 결과:{R}")
    print(f"  {'시나리오':<22} {'avg_detected':>13} {'avg_expected':>13} {'검출률':>8}")
    print(f"  {'─'*22} {'─'*13} {'─'*13} {'─'*8}")
    for name, s in stats.items():
        det  = s["avg_detected"]
        exp  = s["avg_expected"]
        rate = det / exp if exp > 0 else 0
        color = GRN if rate >= 0.9 else YLW if rate >= 0.7 else RED
        print(f"  {DIM}{name:<22}{R} {color}{det:>13.2f}{R} {DIM}{exp:>13.2f}{R} {color}{rate:>7.0%}{R}")
    print()

    print(f"  {DIM}Claude: \"검출 완료. MetricsAgent에 전달합니다.\"{R}")


# ── Slide 3: MetricsAgent ────────────────────────────────────────────────────

def slide_metrics_agent(data):
    header(data, 3)
    hline("═")
    print(f"  {BG_GRN}{BOLD}  STEP 3/4 : MetricsAgent  {R}")
    hline("═")
    print()
    print(f"  {GRN}역할{R}: 검출 결과와 GT를 비교하여 성능 지표를 계산합니다.")
    print(f"  {GRN}도구{R}: compute_metrics, get_all_metrics")
    print()

    print(f"  {GRN}{BOLD}[ 합격 기준 ]{R}")
    criteria = [
        ("mean_error_px", "<", 2.0,  "px", "위치 정확도"),
        ("jitter_rms_px", "<", 1.5,  "px", "지터 (손떨림)"),
        ("snr_db",        ">", 20.0, "dB", "신호 대 잡음비"),
        ("f1",            ">", 0.90, "",   "검출 F1 스코어"),
    ]
    for metric, op, val, unit, desc in criteria:
        print(f"   {GRN}▸{R}  {metric:<16} {op} {val}{unit}   {DIM}({desc}){R}")
    print()

    log_tool("compute_metrics", "scenarios + detections")
    log_tool_result(f"{len(data['metrics'])}개 시나리오 지표 계산 완료")

    summary = data["metrics_summary"]
    print(f"  {GRN}{BOLD}[ 성능 지표 결과 ]{R}")
    print(f"  {'시나리오':<22} {'error':>7} {'jitter':>7} {'snr':>7} {'f1':>6}  {'판정':>6}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*6}  {'─'*6}")
    for name, s in summary.items():
        verdict = s.get("verdict", "?")
        vc  = GRN if verdict == "PASS" else RED
        err = s.get("mean_error_px", 0)
        jit = s.get("jitter_rms_px", 0)
        snr = s.get("snr_db", 0)
        f1  = s.get("f1", 0)
        ec = GRN if err < 2.0 else RED
        jc = GRN if jit < 1.5 else RED
        sc = GRN if snr > 20  else RED
        fc = GRN if f1  > 0.9 else RED
        print(f"  {DIM}{name:<22}{R} "
              f"{ec}{err:>7.2f}{R} {jc}{jit:>7.2f}{R} "
              f"{sc}{snr:>7.1f}{R} {fc}{f1:>6.3f}{R}  "
              f"{vc}{BOLD}{verdict:>6}{R}")
    print()

    print(f"  {DIM}Claude: \"지표 계산 완료. ReportAgent에 전달합니다.\"{R}")


# ── Slide 4: ReportAgent ─────────────────────────────────────────────────────

def slide_report_agent(data):
    header(data, 4)
    hline("═")
    print(f"  {BG_MAG}{BOLD}  STEP 4/4 : ReportAgent  {R}")
    hline("═")
    print()
    print(f"  {MAG}역할{R}: 모든 결과를 종합하여 전문 평가 리포트를 작성합니다.")
    print(f"  {MAG}도구{R}: get_all_results")
    print()

    log_tool("get_all_results", "")
    log_tool_result("scenarios(4) + params + metrics(4) 수신 완료")

    metrics = data["metrics"]
    params  = data["params"]
    summary = data["metrics_summary"]

    passed = [n for n, m in summary.items() if m.get("verdict") == "PASS"]
    failed = [n for n, m in summary.items() if m.get("verdict") == "FAIL"]
    all_err = [m["mean_error_px"] for m in metrics.values()]
    all_snr = [m["snr_db"]        for m in metrics.values()]

    hline()
    print(f"\n  {MAG}{BOLD}📊 Touch IC 알고리즘 평가 리포트{R}\n")
    hline("─")

    print(f"\n  {BOLD}1. 전체 평가 요약{R}")
    vc = GRN if not failed else YLW if len(failed) <= 1 else RED
    print(f"     시나리오: 4개  |  합격: {GRN}{len(passed)}{R}개  |  불합격: {RED}{len(failed)}{R}개")
    print(f"     파라미터: threshold={params.get('threshold')}, sigma={params.get('filter_sigma')}")
    print(f"     종합 판정: {vc}{BOLD}{'전체 PASS ✓' if not failed else f'{len(failed)}개 시나리오 FAIL'}{R}\n")

    print(f"  {BOLD}2. 시나리오별 요약{R}")
    for name, m in summary.items():
        ok = m.get("verdict") == "PASS"
        c  = GRN if ok else RED
        print(f"     {c}{'PASS' if ok else 'FAIL'}{R}  {name:<22}  "
              f"f1={m['f1']:.3f}  error={m['mean_error_px']:.2f}px  snr={m['snr_db']:.1f}dB")

    print(f"\n  {BOLD}3. 강점{R}")
    if passed:
        print(f"     {GRN}▸{R}  {', '.join(passed)} 시나리오 전 지표 합격")
    if all_err and max(all_err) < 1.0:
        print(f"     {GRN}▸{R}  위치 오차 최대 {max(all_err):.2f}px — 매우 우수")

    print(f"\n  {BOLD}4. 약점 및 이슈{R}")
    for name in failed:
        m = metrics[name]
        issues = []
        if m.get("f1", 0) <= 0.9:         issues.append(f"F1={m['f1']:.3f}")
        if m.get("mean_error_px", 0) >= 2: issues.append(f"error={m['mean_error_px']:.2f}px")
        if m.get("snr_db", 0) <= 20:       issues.append(f"SNR={m['snr_db']:.1f}dB")
        print(f"     {RED}▸{R}  {name}: {', '.join(issues)}")
    if not failed:
        print(f"     {GRN}▸{R}  현재 파라미터에서 주요 이슈 없음")

    print(f"\n  {BOLD}5. 개선 권고사항{R}")
    min_snr = min(all_snr) if all_snr else 0
    if min_snr < 22:
        print(f"     {YLW}[P1]{R}  EMI/노이즈 환경 SNR {min_snr:.1f}dB — 적응형 threshold 검토")
    if any(m.get("total_fp", 0) > 3 for m in metrics.values()):
        print(f"     {YLW}[P2]{R}  일부 시나리오 FP 발생 — max_area 파라미터 재조정 검토")
    print(f"     {YLW}[P3]{R}  시나리오별 독립 파라미터 세트 적용 고려")

    print(f"\n  {BOLD}6. 결론{R}")
    if not failed:
        print(f"     {GRN}{BOLD}현재 파라미터(thr={params.get('threshold')}, σ={params.get('filter_sigma')})로")
        print(f"     4개 시나리오 모두 합격 기준 달성.{R}")
    else:
        print(f"     {YLW}일부 시나리오 기준 미달. 파라미터 재조정 필요.{R}")
    hline()


# ── Slide 5: 아키텍처 요약 ────────────────────────────────────────────────────

def slide_architecture(data):
    draw_pipeline(done=DONE_BY_STEP[5])
    hline("═")
    print(f"  {BOLD}{GRN}  ✓ 파이프라인 완료 — 아키텍처 핵심 패턴 요약{R}")
    hline("═")
    print(f"""
  {BOLD}[ 왜 Multi-Agent인가? ]{R}

  {CYN}단일 에이전트{R}:  모든 일을 혼자 → 컨텍스트 과부하, 역할 혼재
  {GRN}Multi-Agent{R} :  역할 분리 → 각자 전문성 + 병렬화 가능

  {BOLD}[ Tool-Use Loop 구조 ]{R}

  {DIM}while True:{R}
      response = claude.messages.create(tools=tools, messages=messages)
      {YLW}if response.stop_reason == "end_turn": break          {DIM}# 완료{R}
      {CYN}if response.stop_reason == "tool_use":               {DIM}# 도구 실행{R}
          result = tool_handlers[block.name](**block.input)
          messages.append(tool_result)                     {DIM}# 결과 전달{R}

  {BOLD}[ 에이전트 간 데이터 흐름 ]{R}

  {BLU}DataAgent{R}    →  frames.npz / scenarios_meta.json
  {YLW}AlgorithmAgent{R} →  detections.json / params.json
  {GRN}MetricsAgent{R}  →  metrics.json
  {MAG}ReportAgent{R}   →  (파일 읽기 후 리포트 출력)

  {BOLD}[ 실행 방법 ]{R}

   {DIM}# Python Orchestrator (순서 고정){R}
   python3 main.py

   {DIM}# LLM Orchestrator (Claude가 순서 결정 + 재시도 판단){R}
   python3 main_agent_sdk.py

   {DIM}# 프레임 시각화 (화살표 탐색){R}
   python3 visualize.py --compare
""")
    hline()


# ════════════════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════════════════

SLIDES = [
    slide_intro,
    slide_data_agent,
    slide_algorithm_agent,
    slide_metrics_agent,
    slide_report_agent,
    slide_architecture,
]

SLIDE_DELAYS = [2.0, 4.0, 4.0, 3.5, 5.0, 4.0]   # 자동 재생 시 각 슬라이드 대기 시간


def render_slide(idx, data):
    clear()
    SLIDES[idx](data)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Pipeline Demo")
    parser.add_argument("--auto", action="store_true", help="자동 재생 모드")
    parser.add_argument("--fast", action="store_true", help="자동 재생 (빠른 속도)")
    args = parser.parse_args()
    auto = args.auto or args.fast

    # 데이터 사전 계산
    clear()
    print(f"\n  {BOLD}파이프라인 실행 중...{R}")
    print(f"  {DIM}(데이터 생성 → 알고리즘 → 지표 계산){R}\n")
    sys.stdout.flush()
    data = precompute()

    # ── 자동 재생 ──────────────────────────────────────────────────────────
    if auto:
        for idx in range(len(SLIDES)):
            render_slide(idx, data)
            delay = SLIDE_DELAYS[idx] * (0.3 if args.fast else 1.0)
            print(f"\n  {DIM}{'─'*60}{R}")
            print(f"  [{idx+1}/{len(SLIDES)}]  {delay:.0f}초 후 다음 슬라이드...")
            sys.stdout.flush()
            time.sleep(delay)
        print(f"\n  {GRN}{BOLD}데모 완료.{R}\n")
        return

    # ── 인터랙티브 모드 (기본값) ────────────────────────────────────────────
    idx = 0
    while True:
        render_slide(idx, data)
        try:
            action = wait_nav(idx, len(SLIDES), fast=False)
        except (KeyboardInterrupt, EOFError):
            break
        if action == "quit":
            break
        elif action == "next" and idx < len(SLIDES) - 1:
            idx += 1
        elif action == "prev" and idx > 0:
            idx -= 1
    clear()


if __name__ == "__main__":
    main()
