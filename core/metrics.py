"""
Touch IC 알고리즘 성능 지표 계산
==================================
터치 감지 알고리즘의 성능을 정량적으로 평가합니다.

주요 지표:
    - Position Accuracy : 위치 정확도 (mean/max/P90 error, 단위: 센서 노드)
    - Jitter            : 지터 (선형 추세 제거 후 RMS, 단위: 센서 노드)
    - SNR               : 신호 대 잡음비 (dB)
    - Detection Rate    : Precision / Recall / F1 Score
"""
import numpy as np
from typing import List, Dict, Any

from .frame_simulator import TestScenario, TouchPoint
from .touch_algorithm import DetectedTouch


def compute_accuracy(
    detected_seq: List[List[DetectedTouch]],
    gt_seq: List[List[TouchPoint]],
    panel_w: int,
    panel_h: int,
    match_threshold_px: float = 3.0
) -> Dict[str, Any]:
    """
    위치 정확도 및 검출률 계산

    매칭 방법: Greedy nearest-neighbor matching
    (GT와 Detection 간 거리가 match_threshold_px 이하이면 매칭)

    Returns:
        mean_error_px : 평균 위치 오차 [sensor nodes]
        max_error_px  : 최대 위치 오차
        p90_error_px  : 90th percentile 오차
        precision     : TP / (TP + FP) - 감지 정밀도
        recall        : TP / (TP + FN) - 감지 재현율
        f1            : 2 * precision * recall / (precision + recall)
    """
    errors = []
    tp = fp = fn = 0

    for detected, gt in zip(detected_seq, gt_seq):
        # 픽셀 좌표로 변환 (센서 노드 단위)
        det_px = [(d.x * (panel_w - 1), d.y * (panel_h - 1)) for d in detected]
        gt_px = [(g.x * (panel_w - 1), g.y * (panel_h - 1)) for g in gt]

        matched_det = set()

        # 각 GT 포인트에 대해 가장 가까운 Detection 찾기
        for gp in gt_px:
            best_dist, best_j = float('inf'), -1
            for j, dp in enumerate(det_px):
                if j in matched_det:
                    continue
                dist = np.sqrt((gp[0] - dp[0])**2 + (gp[1] - dp[1])**2)
                if dist < best_dist:
                    best_dist, best_j = dist, j

            if best_j >= 0 and best_dist < match_threshold_px:
                errors.append(best_dist)
                tp += 1
                matched_det.add(best_j)
            else:
                fn += 1  # GT는 있으나 매칭된 Detection 없음

        # 매칭되지 않은 Detection = False Positive
        fp += len(det_px) - len(matched_det)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "mean_error_px": round(float(np.mean(errors)), 3) if errors else 0.0,
        "max_error_px": round(float(np.max(errors)), 3) if errors else 0.0,
        "p90_error_px": round(float(np.percentile(errors, 90)), 3) if errors else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
    }


def compute_jitter(
    detected_seq: List[List[DetectedTouch]],
    panel_w: int,
    panel_h: int
) -> Dict[str, Any]:
    """
    지터 계산

    지터 = 선형 추세(이동 궤적)를 제거한 후 잔차의 RMS 값
    즉, 이동 중 발생하는 불규칙한 떨림의 크기

    Returns:
        jitter_rms_px : X/Y 합산 RMS 지터 [sensor nodes]
        jitter_x_px   : X 방향 지터
        jitter_y_px   : Y 방향 지터
    """
    positions = []
    for detected in detected_seq:
        if detected:
            d = detected[0]  # 첫 번째 터치 기준
            positions.append((d.x * (panel_w - 1), d.y * (panel_h - 1)))

    if len(positions) < 3:
        return {"jitter_rms_px": 0.0, "jitter_x_px": 0.0, "jitter_y_px": 0.0}

    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    t = np.arange(len(xs), dtype=float)

    # 1차 다항식(직선)으로 추세 제거 (등속 이동 가정)
    x_res = xs - np.polyval(np.polyfit(t, xs, 1), t)
    y_res = ys - np.polyval(np.polyfit(t, ys, 1), t)

    return {
        "jitter_rms_px": round(float(np.sqrt(np.mean(x_res**2 + y_res**2))), 3),
        "jitter_x_px": round(float(np.std(x_res)), 3),
        "jitter_y_px": round(float(np.std(y_res)), 3),
    }


def compute_snr(
    frames: np.ndarray,
    detected_seq: List[List[DetectedTouch]]
) -> Dict[str, Any]:
    """
    신호 대 잡음비(SNR) 계산

    Signal  : 터치 위치 주변 5×5 영역의 최대값
    Noise   : 패널 상단/하단 테두리(2행)의 표준편차 (터치 없는 영역)
    SNR(dB) : 20 * log10(Signal / Noise_std)

    Returns:
        snr_db     : 평균 SNR [dB]
        snr_min_db : 최소 SNR [dB] (worst case)
    """
    snr_values = []
    for frame, detected in zip(frames, detected_seq):
        if not detected:
            continue
        d = detected[0]
        cx = int(d.x * (frame.shape[1] - 1))
        cy = int(d.y * (frame.shape[0] - 1))

        # 신호: 터치 중심 주변 5×5 영역 최대값
        r0, r1 = max(0, cy - 2), min(frame.shape[0], cy + 3)
        c0, c1 = max(0, cx - 2), min(frame.shape[1], cx + 3)
        signal = float(frame[r0:r1, c0:c1].max())

        # 노이즈: 패널 상하단 테두리 (터치 없는 배경 영역)
        border = np.concatenate([frame[:2, :].flatten(), frame[-2:, :].flatten()])
        noise_std = float(border.std())

        if noise_std > 0 and signal > 0:
            snr_values.append(20 * np.log10(signal / noise_std))

    return {
        "snr_db": round(float(np.mean(snr_values)), 1) if snr_values else 0.0,
        "snr_min_db": round(float(np.min(snr_values)), 1) if snr_values else 0.0,
    }


def compute_all(
    scenario: TestScenario,
    detected_seq: List[List[DetectedTouch]]
) -> Dict[str, Any]:
    """
    모든 성능 지표를 한 번에 계산

    Args:
        scenario    : 테스트 시나리오 (ground truth 포함)
        detected_seq: 프레임별 감지된 터치 목록

    Returns:
        모든 지표를 포함한 딕셔너리
    """
    H, W = scenario.panel_shape
    detection_counts = [len(d) for d in detected_seq]
    gt_counts = [len(g) for g in scenario.ground_truth]

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "n_frames": scenario.n_frames,
        "expected_touches_per_frame": round(float(np.mean(gt_counts)), 2),
        "detected_touches_per_frame": round(float(np.mean(detection_counts)), 2),
        **compute_accuracy(detected_seq, scenario.ground_truth, W, H),
        **compute_jitter(detected_seq, W, H),
        **compute_snr(scenario.frames, detected_seq),
    }
