#!/usr/bin/env python3
"""
Touch IC Frame Data Visualizer
터미널에서 프레임 데이터를 애니메이션으로 시각화합니다.

사용법:
    python3 visualize.py                              # 인터랙티브 모드 (기본)
    python3 visualize.py single_touch_drag            # 특정 시나리오만
    python3 visualize.py --raw                        # 로우 숫자값
    python3 visualize.py --compare                    # RAW | 필터 후 좌우 비교
    python3 visualize.py --compare --filter median    # RAW | 미디언 필터 후 비교
    python3 visualize.py --filter-compare             # GAUSSIAN | MEDIAN 동시 비교
    python3 visualize.py emi_noise --filter-compare --median-size 5
    python3 visualize.py --play                       # 자동 재생 애니메이션
    python3 visualize.py --play --fps 20

인터랙티브 키:
    ← → : 프레임 이동
    ↑ ↓ : 시나리오 전환
    0~9  : 10% 단위 점프
    m    : 모드 전환 (ascii → raw → compare → dual)
    f    : compare/dual 모드에서 필터 전환 (gaussian ↔ median)
    q    : 종료
"""
import argparse
import json
import os
import sys
import termios
import time
import tty

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "pipeline_output")

FRAMES_FILE    = os.path.join(OUTPUT_DIR, "frames.npz")
SCENARIOS_FILE = os.path.join(OUTPUT_DIR, "scenarios_meta.json")

# ── ANSI 색상 ────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

SCENARIO_COLORS = {
    "single_touch_drag": GREEN,
    "multi_touch_pinch": YELLOW,
    "emi_noise":         RED,
    "water_blob":        CYAN,
}

CHARS = " .:;+=xX$#"


def render_frame(frame, gt_points):
    H, W   = frame.shape
    vmax   = max(float(frame.max()), 1.0)
    lines  = []
    for r in range(H):
        row = ""
        for c in range(W):
            hit = any(
                abs(p["y"] - r) < 0.9 and abs(p["x"] - c) < 0.9
                for p in gt_points
            )
            val = frame[r, c] / vmax
            ch  = CHARS[min(int(val * len(CHARS)), len(CHARS) - 1)]
            if hit:
                row += RED + BOLD + "★" + RESET
            elif val > 0.6:
                row += YELLOW + BOLD + ch + RESET
            elif val > 0.3:
                row += CYAN + ch + RESET
            elif val > 0.05:
                row += DIM + ch + RESET
            else:
                row += ch
        lines.append(" " + row)
    return lines


def animate_scenario(name, arr, meta_entry, fps):
    n     = len(arr)
    color = SCENARIO_COLORS.get(name, WHITE)
    delay = 1.0 / fps

    for fi in range(n):
        frame = arr[fi]
        gt    = meta_entry["ground_truth"][fi]
        gt_str = "  ".join(f"({p['x']:.1f}, {p['y']:.1f})" for p in gt)

        # 커서 홈으로 이동 (화면 지우지 않고 덮어쓰기)
        sys.stdout.write("\033[H")

        # 헤더
        print(f"{color}{BOLD}▶  {name}{RESET}  {DIM}{meta_entry['description']}{RESET}")
        print(f"   frame {fi+1:02d}/{n}   panel {meta_entry['panel_h']}H×{meta_entry['panel_w']}W")
        print(f"   GT 터치: {RED}{BOLD}{gt_str}{RESET}")
        print("─" * 52)

        for line in render_frame(frame, gt):
            print(line)

        # 진행 바
        pct = (fi + 1) / n
        filled = int(pct * 44)
        bar = "█" * filled + "░" * (44 - filled)
        print(f"\n  {color}[{bar}]{RESET}  {int(pct * 100):3d}%")

        sys.stdout.flush()
        time.sleep(delay)


def render_raw(frame, gt_points):
    """숫자 히트맵: 값을 3자리로 출력, GT 위치는 색상 강조"""
    H, W  = frame.shape
    lines = []
    for r in range(H):
        row = ""
        for c in range(W):
            val = int(frame[r, c])
            hit = any(abs(p["y"] - r) < 0.9 and abs(p["x"] - c) < 0.9 for p in gt_points)
            cell = f"{val:3d}"
            if hit:
                row += RED + BOLD + cell + RESET
            elif val > 150:
                row += YELLOW + BOLD + cell + RESET
            elif val > 80:
                row += CYAN + cell + RESET
            elif val > 20:
                row += DIM + cell + RESET
            else:
                row += "   "
        lines.append(row)
    return lines


def render_compare(raw, filtered, threshold, gt_points, detections=None):
    """RAW 숫자 | 필터 후 숫자 좌우 병렬 출력. 검출된 터치는 초록 ◆ 표시."""
    H, W      = raw.shape
    vmax_raw  = max(float(raw.max()), 1.0)
    vmax_filt = max(float(filtered.max()), 1.0)
    lines     = []

    for r in range(H):
        left = right = ""
        for c in range(W):
            hit = any(abs(p["y"] - r) < 0.9 and abs(p["x"] - c) < 0.9 for p in gt_points)
            det = detections and any(abs(d["y"] - r) < 0.9 and abs(d["x"] - c) < 0.9 for d in detections)

            # 왼쪽: RAW
            rv   = int(raw[r, c])
            rcell = f"{rv:3d}"
            if hit:
                left += RED + BOLD + rcell + RESET
            elif rv > 150:
                left += YELLOW + BOLD + rcell + RESET
            elif rv > 80:
                left += CYAN + rcell + RESET
            elif rv > 20:
                left += DIM + rcell + RESET
            else:
                left += "   "

            # 오른쪽: 필터 후
            fv    = int(filtered[r, c])
            fcell = f"{fv:3d}"
            above = filtered[r, c] > threshold
            if det:
                right += GREEN + BOLD + fcell + RESET
            elif hit:
                right += RED + BOLD + fcell + RESET
            elif above:
                right += YELLOW + BOLD + fcell + RESET
            elif fv > 40:
                right += CYAN + fcell + RESET
            elif fv > 10:
                right += DIM + fcell + RESET
            else:
                right += "   "

        lines.append(f"{left}  {DIM}│{RESET}  {right}")
    return lines


def render_dual_filter(gauss_f, median_f, threshold, gt_points, gauss_dets, median_dets):
    """GAUSSIAN 필터 후 | MEDIAN 필터 후 좌우 병렬 출력"""
    H, W = gauss_f.shape
    lines = []

    for r in range(H):
        left = right = ""
        for c in range(W):
            hit = any(abs(p["y"] - r) < 0.9 and abs(p["x"] - c) < 0.9 for p in gt_points)
            gdet = any(abs(d["y"] - r) < 0.9 and abs(d["x"] - c) < 0.9 for d in gauss_dets)
            mdet = any(abs(d["y"] - r) < 0.9 and abs(d["x"] - c) < 0.9 for d in median_dets)

            # 왼쪽: Gaussian 필터 후
            gv = int(gauss_f[r, c])
            gcell = f"{gv:3d}"
            if gdet:
                left += GREEN + BOLD + gcell + RESET
            elif hit:
                left += RED + BOLD + gcell + RESET
            elif gauss_f[r, c] > threshold:
                left += YELLOW + BOLD + gcell + RESET
            elif gv > 40:
                left += CYAN + gcell + RESET
            elif gv > 10:
                left += DIM + gcell + RESET
            else:
                left += "   "

            # 오른쪽: Median 필터 후
            mv = int(median_f[r, c])
            mcell = f"{mv:3d}"
            if mdet:
                right += GREEN + BOLD + mcell + RESET
            elif hit:
                right += RED + BOLD + mcell + RESET
            elif median_f[r, c] > threshold:
                right += YELLOW + BOLD + mcell + RESET
            elif mv > 40:
                right += CYAN + mcell + RESET
            elif mv > 10:
                right += DIM + mcell + RESET
            else:
                right += "   "

        lines.append(f"{left}  {DIM}│{RESET}  {right}")
    return lines


def animate_raw(name, arr, meta_entry, fps, fixed_frame=None):
    n     = len(arr)
    color = SCENARIO_COLORS.get(name, WHITE)
    frames_to_show = [fixed_frame] if fixed_frame is not None else range(n)
    delay = 1.0 / fps

    for fi in frames_to_show:
        frame = arr[fi]
        gt    = meta_entry["ground_truth"][fi]
        gt_str = "  ".join(f"({p['x']:.1f},{p['y']:.1f})" for p in gt)

        sys.stdout.write("\033[H")
        print(f"{color}{BOLD}▶  {name}  [RAW]{RESET}  {DIM}{meta_entry['description']}{RESET}")
        print(f"   frame {fi+1:02d}/{n}   GT: {RED}{BOLD}{gt_str}{RESET}")
        print(f"   max={frame.max():.0f}  mean={frame.mean():.1f}  noise_std={frame[:2,:].std():.1f}")
        print(f"   {'col':>3}" + "".join(f"{c:3d}" for c in range(arr.shape[2])))
        print("   " + "─" * (arr.shape[2] * 3 + 1))

        for ri, line in enumerate(render_raw(frame, gt)):
            print(f"{DIM}{ri:2d}{RESET} │{line}")

        if fixed_frame is None:
            pct    = (fi + 1) / n
            filled = int(pct * 44)
            bar    = "█" * filled + "░" * (44 - filled)
            print(f"\n  {color}[{bar}]{RESET}  {int(pct*100):3d}%")
            sys.stdout.flush()
            time.sleep(delay)


# ── 키보드 입력 ──────────────────────────────────────────────────────────────

def read_key():
    """단일 키 입력 읽기 (non-blocking, raw mode)"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":                  # ESC 시퀀스 (화살표 키)
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(ch3, "")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def apply_filter(frame, filter_type, sigma, median_size):
    """선택된 노드 필터 알고리즘 적용"""
    if filter_type == "median":
        size = median_size if median_size % 2 == 1 else median_size + 1
        return ndi.median_filter(frame.astype(float), size=size)
    else:  # gaussian
        return gaussian_filter(frame.astype(float), sigma=sigma)


def get_detections(filtered, threshold):
    binary = (filtered > threshold).astype(int)
    labeled, n_comp = ndi.label(binary)
    result = []
    for lbl in range(1, n_comp + 1):
        coords  = np.argwhere(labeled == lbl)
        weights = filtered[labeled == lbl]
        cy = float(np.average(coords[:, 0], weights=weights))
        cx = float(np.average(coords[:, 1], weights=weights))
        result.append({"x": cx, "y": cy})
    return result


# ── 프레임 렌더러 (모드별) ───────────────────────────────────────────────────

def render_ascii(frame, gt_points):
    H, W  = frame.shape
    vmax  = max(float(frame.max()), 1.0)
    lines = []
    for r in range(H):
        row = ""
        for c in range(W):
            hit = any(abs(p["y"]-r) < 0.9 and abs(p["x"]-c) < 0.9 for p in gt_points)
            val = frame[r, c] / vmax
            ch  = CHARS[min(int(val * len(CHARS)), len(CHARS)-1)]
            if hit:             row += RED + BOLD + "★" + RESET
            elif val > 0.6:     row += YELLOW + BOLD + ch + RESET
            elif val > 0.3:     row += CYAN + ch + RESET
            elif val > 0.05:    row += DIM + ch + RESET
            else:               row += ch
        lines.append(" " + row)
    return lines


def print_frame(name, meta_entry, fi, n, mode, sigma, threshold, frame, gt, color,
                filter_type="gaussian", median_size=3):
    """현재 프레임을 화면에 출력 (커서는 호출자가 홈으로 이동)"""
    W = frame.shape[1]

    # ── 공통 헤더 ──────────────────────────────────────────────────────────
    gt_str   = "  ".join(f"({p['x']:.1f},{p['y']:.1f})" for p in gt)
    mode_tag = {"ascii": "ASCII", "raw": "RAW", "compare": "COMPARE", "dual": "DUAL"}[mode]
    print(f"{color}{BOLD}▶  {name}  [{mode_tag}]{RESET}  {DIM}{meta_entry['description']}{RESET}  " + " "*10)
    print(f"   frame {fi+1:02d}/{n}     ← → 프레임  ↑ ↓ 시나리오  m 모드  f 필터  q 종료")
    print(f"   GT: {RED}{BOLD}{gt_str}{RESET}" + " "*30)

    if mode == "compare":
        filtered   = apply_filter(frame, filter_type, sigma, median_size)
        detections = get_detections(filtered, threshold)
        det_str    = "  ".join(f"({d['x']:.1f},{d['y']:.1f})" for d in detections) or "없음"
        fp = f"sigma={sigma}" if filter_type == "gaussian" else f"median_size={median_size}"
        print(f"   [{filter_type}] {fp}  thr={threshold}   검출: {GREEN}{BOLD}{det_str}{RESET}" + " "*10)
    elif mode == "dual":
        gauss_f  = apply_filter(frame, "gaussian", sigma, median_size)
        median_f = apply_filter(frame, "median",   sigma, median_size)
        g_dets   = get_detections(gauss_f,  threshold)
        m_dets   = get_detections(median_f, threshold)
        g_str = "  ".join(f"({d['x']:.1f},{d['y']:.1f})" for d in g_dets) or "없음"
        m_str = "  ".join(f"({d['x']:.1f},{d['y']:.1f})" for d in m_dets) or "없음"
        print(f"   thr={threshold}  G검출:{GREEN}{BOLD}{g_str}{RESET}  M검출:{GREEN}{BOLD}{m_str}{RESET}" + " "*5)
    elif mode in ("raw",):
        filtered   = apply_filter(frame, filter_type, sigma, median_size)
        detections = get_detections(filtered, threshold)
        det_str    = "  ".join(f"({d['x']:.1f},{d['y']:.1f})" for d in detections) or "없음"
        print(f"   [{filter_type}] sigma={sigma}  thr={threshold}   검출: {GREEN}{BOLD}{det_str}{RESET}" + " "*10)
    else:
        filtered = detections = None
        print(" " * 50)

    print("─" * 52)

    # ── 모드별 본문 ────────────────────────────────────────────────────────
    if mode == "ascii":
        for line in render_ascii(frame, gt):
            print(line)

    elif mode == "raw":
        col_hdr = "".join(f"{c:3d}" for c in range(W))
        print(f"   {col_hdr}")
        print("   " + "─" * (W*3))
        for ri, line in enumerate(render_raw(frame, gt)):
            print(f"{DIM}{ri:2d}{RESET} │{line}")

    elif mode == "compare":
        filter_label = f"Gaussian 필터 후 (σ={sigma})" if filter_type == "gaussian" \
                       else f"Median 필터 후 (size={median_size})"
        col_hdr = "".join(f"{c:3d}" for c in range(W))
        sep     = "─" * (W*3)
        print(f"  {BOLD}{'RAW':^{W*3}}{RESET}    {BOLD}{filter_label:^{W*3}}{RESET}")
        print(f"  {col_hdr}    {col_hdr}")
        print(f"  {sep}    {sep}")
        for ri, line in enumerate(render_compare(frame, filtered, threshold, gt, detections)):
            print(f"{DIM}{ri:2d}{RESET} {line}")

    elif mode == "dual":
        col_hdr = "".join(f"{c:3d}" for c in range(W))
        sep     = "─" * (W*3)
        g_label = f"Gaussian 필터 (σ={sigma},  ◆검출)"
        m_label = f"Median 필터 (size={median_size}, ◆검출)"
        print(f"  {BOLD}{g_label:^{W*3}}{RESET}    {BOLD}{m_label:^{W*3}}{RESET}")
        print(f"  {col_hdr}    {col_hdr}")
        print(f"  {sep}    {sep}")
        for ri, line in enumerate(render_dual_filter(gauss_f, median_f, threshold, gt, g_dets, m_dets)):
            print(f"{DIM}{ri:2d}{RESET} {line}")

    # ── 진행 바 ────────────────────────────────────────────────────────────
    pct    = (fi + 1) / n
    filled = int(pct * 44)
    bar    = "█" * filled + "░" * (44 - filled)
    print(f"\n  {color}[{bar}]{RESET}  {int(pct*100):3d}%  ({fi+1}/{n})")
    sys.stdout.flush()


def animate_compare(name, arr, meta_entry, fps, sigma, threshold, fixed_frame=None):
    """RAW 숫자값과 가우시안 필터 후 숫자값을 좌우로 나란히 표시"""
    n     = len(arr)
    W     = arr.shape[2]
    color = SCENARIO_COLORS.get(name, WHITE)
    frames_to_show = [fixed_frame] if fixed_frame is not None else range(n)
    delay = 1.0 / fps

    col_header = "".join(f"{c:3d}" for c in range(W))
    sep_line   = "─" * (W * 3)

    for fi in frames_to_show:
        raw      = arr[fi]
        filtered = gaussian_filter(raw.astype(float), sigma=sigma)
        gt       = meta_entry["ground_truth"][fi]
        gt_str   = "  ".join(f"({p['x']:.1f},{p['y']:.1f})" for p in gt)

        # threshold 초과 영역에서 무게중심 간단 계산 (검출 위치 표시용)
        from scipy import ndimage as ndi
        binary = (filtered > threshold).astype(int)
        labeled, n_comp = ndi.label(binary)
        detections = []
        for lbl in range(1, n_comp + 1):
            coords = np.argwhere(labeled == lbl)
            weights = filtered[labeled == lbl]
            cy = float(np.average(coords[:, 0], weights=weights))
            cx = float(np.average(coords[:, 1], weights=weights))
            detections.append({"x": cx, "y": cy})

        det_str = "  ".join(f"({d['x']:.1f},{d['y']:.1f})" for d in detections) or "없음"

        sys.stdout.write("\033[H")
        print(f"{color}{BOLD}▶  {name}  [COMPARE]{RESET}  {DIM}{meta_entry['description']}{RESET}")
        print(f"   frame {fi+1:02d}/{n}   sigma={sigma}  threshold={threshold}")
        print(f"   GT:{RED}{BOLD} {gt_str}{RESET}   검출:{GREEN}{BOLD} {det_str}{RESET}")
        print()
        print(f"  {BOLD}{'RAW':^{W*3}}{RESET}    {BOLD}{'Gaussian 필터 후  (★GT  ◆검출)':^{W*3}}{RESET}")
        print(f"  {col_header}    {col_header}")
        print(f"  {sep_line}    {sep_line}")

        for ri, line in enumerate(render_compare(raw, filtered, threshold, gt, detections)):
            print(f"{DIM}{ri:2d}{RESET} {line}")

        if fixed_frame is None:
            pct    = (fi + 1) / n
            filled = int(pct * 44)
            bar    = "█" * filled + "░" * (44 - filled)
            print(f"\n  {color}[{bar}]{RESET}  {int(pct*100):3d}%")
            sys.stdout.flush()
            time.sleep(delay)


def interactive(names, frames_data, meta, init_mode, sigma, threshold, init_scenario=0,
                init_filter="gaussian", median_size=3):
    """화살표 키로 프레임/시나리오 탐색하는 인터랙티브 뷰어"""
    MODES   = ["ascii", "raw", "compare", "dual"]
    FILTERS = ["gaussian", "median"]
    si          = init_scenario
    fi          = 0
    mode        = init_mode
    filter_type = init_filter

    print("\033[2J\033[H", end="")

    while True:
        name  = names[si]
        arr   = frames_data[name]
        n     = len(arr)
        fi    = max(0, min(fi, n - 1))
        frame = arr[fi]
        gt    = meta[name]["ground_truth"][fi]
        color = SCENARIO_COLORS.get(name, WHITE)

        sys.stdout.write("\033[H")
        print_frame(name, meta[name], fi, n, mode, sigma, threshold, frame, gt, color,
                    filter_type=filter_type, median_size=median_size)

        # ── 키 입력 대기 ──────────────────────────────────────────────────
        key = read_key()

        if key in ("q", "\x03"):
            break
        elif key == "RIGHT":
            fi = min(fi + 1, n - 1)
        elif key == "LEFT":
            fi = max(fi - 1, 0)
        elif key == "UP":
            si = (si - 1) % len(names)
            fi = 0
        elif key == "DOWN":
            si = (si + 1) % len(names)
            fi = 0
        elif key.isdigit():
            fi = int(int(key) / 10 * n)
        elif key == "m":
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
        elif key == "f":                  # 필터 전환 (compare / dual 모드에서 유효)
            filter_type = FILTERS[(FILTERS.index(filter_type) + 1) % len(FILTERS)]

    print("\033[2J\033[H", end="")
    print(f"{GREEN}{BOLD}종료{RESET}")


def play(names, frames_data, meta, init_mode, sigma, threshold, fps,
         filter_type="gaussian", median_size=3):
    """자동 재생 모드"""
    print("\033[2J\033[H", end="")
    try:
        for name in names:
            arr   = frames_data[name]
            n     = len(arr)
            color = SCENARIO_COLORS.get(name, WHITE)
            delay = 1.0 / fps
            for fi in range(n):
                sys.stdout.write("\033[H")
                print_frame(name, meta[name], fi, n, init_mode, sigma, threshold,
                            arr[fi], meta[name]["ground_truth"][fi], color,
                            filter_type=filter_type, median_size=median_size)
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    print("\033[2J\033[H", end="")
    print(f"{GREEN}{BOLD}완료{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Touch IC Frame Visualizer")
    parser.add_argument("scenario",        nargs="?",                          help="시나리오 이름 (생략 시 전체)")
    parser.add_argument("--raw",           action="store_true",                help="로우 숫자값 모드")
    parser.add_argument("--compare",       action="store_true",                help="RAW | 선택 필터 후 좌우 비교")
    parser.add_argument("--filter-compare",action="store_true",                help="GAUSSIAN | MEDIAN 동시 비교 (dual 모드)")
    parser.add_argument("--filter",        choices=["gaussian", "median"],
                        default="gaussian",                                     help="compare 모드에서 사용할 필터 (기본: gaussian)")
    parser.add_argument("--play",          action="store_true",                help="자동 재생 (기본: 인터랙티브)")
    parser.add_argument("--fps",           type=float, default=12,             help="자동재생 fps (기본값: 12)")
    parser.add_argument("--sigma",         type=float, default=1.2,            help="가우시안 sigma (기본값: 1.2)")
    parser.add_argument("--median-size",   type=int,   default=3,              help="미디언 커널 크기, 홀수 (기본값: 3)")
    parser.add_argument("--threshold",     type=float, default=55.0,           help="검출 threshold (기본값: 55)")
    args = parser.parse_args()

    if not os.path.exists(FRAMES_FILE) or not os.path.exists(SCENARIOS_FILE):
        print("데이터 파일이 없습니다. 먼저 생성하세요:")
        print(f"  python3 pipeline_runner.py generate-data")
        sys.exit(1)

    frames_data = np.load(FRAMES_FILE)
    with open(SCENARIOS_FILE) as f:
        meta = json.load(f)

    all_names = list(frames_data.files)
    if args.scenario:
        if args.scenario not in all_names:
            print(f"시나리오 '{args.scenario}'를 찾을 수 없습니다.")
            print(f"사용 가능: {', '.join(all_names)}")
            sys.exit(1)
        names   = all_names
        init_si = all_names.index(args.scenario)
    else:
        names   = all_names
        init_si = 0

    if args.filter_compare:
        mode = "dual"
    elif args.compare:
        mode = "compare"
    elif args.raw:
        mode = "raw"
    else:
        mode = "ascii"

    if args.play:
        play(names, frames_data, meta, mode, args.sigma, args.threshold, args.fps,
             filter_type=args.filter, median_size=args.median_size)
    else:
        interactive(names, frames_data, meta, mode, args.sigma, args.threshold, init_si,
                    init_filter=args.filter, median_size=args.median_size)


if __name__ == "__main__":
    main()
