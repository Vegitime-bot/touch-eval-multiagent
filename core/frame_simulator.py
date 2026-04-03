"""
터치 패널 정전용량 프레임 데이터 시뮬레이터
============================================
실제 Touch IC가 읽어오는 raw capacitance delta frame과 유사한 2D grid 데이터를 생성합니다.

패널 사양:
    - 18 rows × 32 columns 센서 노드
    - 터치 신호: 가우시안 분포의 정전용량 변화
    - 노이즈: 열잡음, EMI, 물방울 간섭 등

사용 예시:
    sim = TouchFrameSimulator()
    scenario = sim.single_touch_drag()
    print(scenario.frames.shape)  # (40, 18, 32)
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TouchPoint:
    """Ground truth 터치 위치 (정규화 좌표 0.0 ~ 1.0)"""
    x: float
    y: float


@dataclass
class TestScenario:
    """
    테스트 시나리오: 프레임 시퀀스 + ground truth

    Attributes:
        name: 시나리오 식별자
        description: 시나리오 설명
        frames: 정전용량 프레임 배열 shape=(N_frames, H, W)
        ground_truth: 프레임별 실제 터치 위치 목록
    """
    name: str
    description: str
    frames: np.ndarray
    ground_truth: List[List[TouchPoint]]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def panel_shape(self) -> Tuple[int, int]:
        """(H, W) 반환"""
        return self.frames.shape[1], self.frames.shape[2]


class TouchFrameSimulator:
    """
    정전용량 방식 터치 패널의 프레임 데이터를 시뮬레이션합니다.

    실제 Touch IC에서 나오는 raw frame data를 모사:
    - 터치 블롭: 가우시안 형태의 정전용량 변화 (delta Capacitance)
    - 노이즈: 가우시안 열잡음, EMI 패턴 노이즈, 물방울 신호
    """

    H = 18   # 센서 행 수 (row electrodes)
    W = 32   # 센서 열 수 (column electrodes)

    def _blob(self, x_norm: float, y_norm: float,
              intensity: float = 200.0, sigma: float = 1.8) -> np.ndarray:
        """
        가우시안 형태의 터치 블롭 생성

        실제 터치 시 정전용량 변화는 터치 중심에서 가우시안 형태로 퍼집니다.

        Args:
            x_norm: 정규화 X 좌표 (0.0 ~ 1.0)
            y_norm: 정규화 Y 좌표 (0.0 ~ 1.0)
            intensity: 블롭 최대값 (ADC count)
            sigma: 블롭 확산 반경 (센서 노드 단위)
        """
        cx = x_norm * (self.W - 1)
        cy = y_norm * (self.H - 1)
        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        return intensity * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    # ── 시나리오 생성 메서드 ────────────────────────────────────────────────

    def single_touch_drag(self, n_frames: int = 40) -> TestScenario:
        """
        단일 터치 좌→우 드래그 시나리오

        평가 목적: 위치 정확도, 선형성, 지터
        """
        frames, gt = [], []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            x, y = 0.15 + 0.7 * t, 0.5
            frame = self._blob(x, y) + np.random.normal(0, 8.0, (self.H, self.W))
            frames.append(np.clip(frame, 0, None))
            gt.append([TouchPoint(x=x, y=y)])
        return TestScenario(
            name="single_touch_drag",
            description="단일 터치 좌→우 드래그",
            frames=np.array(frames),
            ground_truth=gt
        )

    def multi_touch_pinch(self, n_frames: int = 40) -> TestScenario:
        """
        두 손가락 피치 아웃 시나리오

        평가 목적: 멀티터치 감지, 두 터치 분리 능력
        """
        frames, gt = [], []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            gap = 0.1 + 0.3 * t
            x1 = max(0.05, 0.5 - gap)
            x2 = min(0.95, 0.5 + gap)
            frame = (self._blob(x1, 0.5) + self._blob(x2, 0.5)
                     + np.random.normal(0, 8.0, (self.H, self.W)))
            frames.append(np.clip(frame, 0, None))
            gt.append([TouchPoint(x=x1, y=0.5), TouchPoint(x=x2, y=0.5)])
        return TestScenario(
            name="multi_touch_pinch",
            description="두 손가락 피치 아웃",
            frames=np.array(frames),
            ground_truth=gt
        )

    def emi_noise(self, n_frames: int = 40) -> TestScenario:
        """
        충전기 EMI 노이즈 환경 단일 터치 시나리오

        평가 목적: 노이즈 내성, SNR
        EMI 패턴: 수평 줄무늬 형태 (충전기 노이즈 특성)
        """
        frames, gt = [], []
        x_pos = np.linspace(0, 2 * np.pi, self.W)
        for i in range(n_frames):
            x, y = 0.5, 0.5
            frame = self._blob(x, y)
            # 충전기 EMI: 주파수가 있는 수평 줄무늬 패턴
            emi = 35.0 * np.sin(x_pos + i * 0.5)[np.newaxis, :]
            frame += emi + np.random.normal(0, 12.0, (self.H, self.W))
            frames.append(np.clip(frame, 0, None))
            gt.append([TouchPoint(x=x, y=y)])
        return TestScenario(
            name="emi_noise",
            description="충전기 EMI 노이즈 환경",
            frames=np.array(frames),
            ground_truth=gt
        )

    def water_blob(self, n_frames: int = 40) -> TestScenario:
        """
        물방울 false touch 시나리오

        평가 목적: False positive 제거 능력
        특성: 물방울은 실제 터치보다 넓고(sigma 큼) 약한(intensity 낮음) 신호
        Ground truth에는 실제 터치만 포함 (물방울은 false positive)
        """
        frames, gt = [], []
        for i in range(n_frames):
            x, y = 0.65, 0.5   # 실제 터치 위치
            # 실제 터치: 좁고 강한 신호
            frame = self._blob(x, y, intensity=200, sigma=1.8)
            # 물방울: 넓고 약한 신호 (sigma 크고 intensity 낮음)
            frame += self._blob(0.2, 0.3, intensity=65, sigma=4.5)
            frame += np.random.normal(0, 6.0, (self.H, self.W))
            frames.append(np.clip(frame, 0, None))
            gt.append([TouchPoint(x=x, y=y)])  # 물방울은 GT 아님
        return TestScenario(
            name="water_blob",
            description="물방울 false touch 시나리오",
            frames=np.array(frames),
            ground_truth=gt
        )

    def get_all(self) -> List[TestScenario]:
        """모든 시나리오 반환"""
        return [
            self.single_touch_drag(),
            self.multi_touch_pinch(),
            self.emi_noise(),
            self.water_blob(),
        ]
