"""
터치 감지 알고리즘
==================
실제 Touch IC firmware에서 사용하는 알고리즘과 유사한 방식으로 구현합니다.

처리 파이프라인:
    1. Node Filter          → 노이즈 제거 (선택 가능: Gaussian / Median)
    2. Threshold Binarize   → 터치 후보 영역 추출
    3. Connected Labeling   → 블롭(Blob) 분리
    4. Area Filter          → 크기 기반 필터링 (먼지 / 물방울 제거)
    5. Centroid Calculation → 가중치 중심으로 위치 계산

필터 알고리즘 비교:
    gaussian  → 연속 분포 노이즈(열잡음, EMI)에 강함. 가중 평균으로 부드럽게 처리.
                파라미터: filter_sigma (클수록 강한 평활화, 공간 해상도 저하)
    median    → 임펄스성 노이즈(스파이크, 물방울)에 강함. 에지를 보존하며 이상값 제거.
                파라미터: median_size (홀수 커널 크기, 클수록 강한 이상값 억제)
"""
import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import List


# 선택 가능한 필터 알고리즘 설명 (AlgorithmAgent가 참조)
FILTER_ALGORITHMS = {
    "gaussian": {
        "name": "가우시안 필터 (Gaussian Filter)",
        "description": "정규분포 가중 평균. 연속적인 열잡음·EMI 노이즈 억제에 최적.",
        "param": "filter_sigma (float) — 클수록 강한 평활화, 공간 해상도 저하",
        "best_for": ["emi_noise", "single_touch_drag", "multi_touch_pinch"],
        "weakness": "임펄스성 노이즈(물방울 스파이크)는 희석만 될 뿐 제거 안 됨",
    },
    "median": {
        "name": "미디언 필터 (Median Filter)",
        "description": "이웃 픽셀의 중앙값 선택. 임펄스 노이즈·물방울 스파이크 제거에 최적.",
        "param": "median_size (int, 홀수) — 클수록 강한 이상값 억제, 에지 뭉개짐 증가",
        "best_for": ["water_blob", "emi_noise"],
        "weakness": "연속 분포 노이즈(가우시안 노이즈)에는 가우시안 필터보다 효과 낮음",
    },
}


@dataclass
class DetectedTouch:
    """감지된 터치 정보"""
    x: float          # 정규화 X 좌표 (0.0 ~ 1.0)
    y: float          # 정규화 Y 좌표 (0.0 ~ 1.0)
    confidence: float  # 블롭 평균 강도 (신뢰도 지표)
    area: int          # 블롭 크기 (노드 수)


class TouchDetector:
    """
    정전용량 터치 감지기

    주요 파라미터 설명:
        threshold (float):
            터치 감지 임계값 (ADC count 단위)
            높을수록 → 감도 낮아짐, false positive 감소
            낮을수록 → 감도 높아짐, false positive 증가

        filter_sigma (float):
            가우시안 필터 강도
            높을수록 → 노이즈 제거 강, 공간 해상도 저하
            낮을수록 → 노이즈 제거 약, 공간 해상도 유지

        min_area (int):
            최소 블롭 크기 (노드 수)
            작은 노이즈 스파이크, 먼지 제거용

        max_area (int):
            최대 블롭 크기 (노드 수)
            물방울, 손바닥 등 대면적 false positive 제거용
    """

    def __init__(self, threshold: float = 50.0, filter_sigma: float = 1.0,
                 min_area: int = 3, max_area: int = 60,
                 filter_type: str = "gaussian", median_size: int = 3):
        self.threshold = threshold
        self.filter_sigma = filter_sigma
        self.min_area = min_area
        self.max_area = max_area
        self.filter_type = filter_type   # "gaussian" | "median"
        self.median_size = median_size   # 미디언 필터 커널 크기 (홀수)

    def detect_frame(self, frame: np.ndarray) -> List[DetectedTouch]:
        """
        단일 프레임에서 터치 감지

        Args:
            frame: 정전용량 프레임 데이터 shape=(H, W)

        Returns:
            감지된 터치 목록
        """
        # 1단계: 노드 필터 적용 (필터 알고리즘에 따라 분기)
        if self.filter_type == "median":
            # 미디언 필터: 임펄스 노이즈·물방울 스파이크 제거, 에지 보존
            size = self.median_size if self.median_size % 2 == 1 else self.median_size + 1
            filtered = ndimage.median_filter(frame, size=size)
        else:
            # 가우시안 필터 (기본): 연속 분포 노이즈 억제
            filtered = ndimage.gaussian_filter(frame, sigma=self.filter_sigma)

        # 2단계: 임계값으로 바이너리 마스크 생성
        binary = filtered > self.threshold

        # 3단계: Connected Component Labeling (블롭 분리)
        labeled, n_components = ndimage.label(binary)

        touches = []
        for label_id in range(1, n_components + 1):
            region_mask = labeled == label_id
            area = int(region_mask.sum())

            # 4단계: 면적 필터링
            if not (self.min_area <= area <= self.max_area):
                continue

            # 5단계: 가중치 중심(Intensity-Weighted Centroid) 계산
            #         단순 중심보다 정확한 sub-pixel 위치 추정 가능
            intensities = filtered * region_mask
            total_intensity = float(intensities.sum())
            if total_intensity <= 0:
                continue

            row_weights = intensities.sum(axis=1)  # 행 방향 합산
            col_weights = intensities.sum(axis=0)  # 열 방향 합산

            cy = float(np.average(np.arange(frame.shape[0]), weights=row_weights))
            cx = float(np.average(np.arange(frame.shape[1]), weights=col_weights))

            touches.append(DetectedTouch(
                x=cx / (frame.shape[1] - 1),   # 0~1 정규화
                y=cy / (frame.shape[0] - 1),
                confidence=total_intensity / area,
                area=area
            ))

        return touches

    def detect_sequence(self, frames: np.ndarray) -> List[List[DetectedTouch]]:
        """
        프레임 시퀀스 전체 처리

        Args:
            frames: 프레임 배열 shape=(N, H, W)

        Returns:
            프레임별 감지된 터치 목록
        """
        return [self.detect_frame(frame) for frame in frames]
