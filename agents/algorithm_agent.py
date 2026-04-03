"""
AlgorithmAgent: 터치 감지 알고리즘 실행 전문 에이전트
======================================================
역할: 각 테스트 시나리오의 특성에 맞는 파라미터를 선택하고 알고리즘을 실행합니다.

Claude가 담당하는 부분:
    - 시나리오 특성 분석 (노이즈 환경, 물방울 등)
    - 시나리오별 최적 파라미터 결정 (threshold, filter_sigma 등)
    - 실행 결과 해석

Python이 담당하는 부분:
    - 실제 TouchDetector 알고리즘 실행 (numpy/scipy 연산)
    - 감지 결과를 pipeline_data에 저장
"""
from typing import Dict, Any
import numpy as np

from .base import BaseAgent
from core.touch_algorithm import TouchDetector, FILTER_ALGORITHMS


SYSTEM_PROMPT = """당신은 Touch IC 알고리즘 실행 전문가입니다.

당신의 역할:
- 노드 필터 알고리즘을 선택하고 각 시나리오에 맞는 파라미터를 결정합니다
- select_filter_algorithm → get_scenario_list → run_detection 순서로 실행하세요

━━━ 노드 필터 알고리즘 ━━━
    gaussian (가우시안 필터)
        - 연속 분포 노이즈(열잡음, EMI)에 강함
        - 파라미터: filter_sigma — 클수록 강한 평활화 (권장: 0.8~2.0)
        - 약점: 물방울 같은 임펄스 노이즈는 희석만 될 뿐 제거 불완전

    median (미디언 필터)
        - 임펄스 노이즈·물방울 스파이크 제거, 에지 보존
        - 파라미터: median_size — 홀수 커널 크기 (권장: 3~7)
        - 약점: 연속 가우시안 노이즈에는 gaussian보다 효과 낮음

━━━ 검출 파라미터 가이드 ━━━
    threshold (기본: 50.0)
        - 노이즈가 많은 환경 → 높게 설정 (70~90)
        - 신호가 약한 환경   → 낮게 설정 (30~40)

    min_area (기본: 3)
        - 스파이크 노이즈 많음 → 높게 설정 (5~8)

    max_area (기본: 60)
        - 물방울 제거 필요   → 낮게 설정 (25~35)
        - 손바닥 지원 필요   → 높게 설정 (100+)

select_filter_algorithm으로 필터를 선택하고,
get_scenario_list로 시나리오를 확인한 후,
각 시나리오 특성에 맞는 파라미터로 run_detection을 실행하세요.
모든 시나리오 처리 후 필터 선택 이유와 파라미터 결정 이유를 포함한 요약을 작성해주세요."""


class AlgorithmAgent(BaseAgent):

    # ── Tool 구현 메서드 ────────────────────────────────────────────────────

    def _select_filter_algorithm(self, filter_type: str, reason: str) -> Dict[str, Any]:
        """
        노드 필터 알고리즘 선택 및 pipeline_data에 저장

        Args:
            filter_type: "gaussian" 또는 "median"
            reason: 선택 이유 (교육용 출력에 활용)
        """
        if filter_type not in FILTER_ALGORITHMS:
            return {"error": f"지원하지 않는 필터: {filter_type}. 선택 가능: {list(FILTER_ALGORITHMS.keys())}"}

        self.data["filter_type"] = filter_type
        info = FILTER_ALGORITHMS[filter_type]

        print(f"  → 필터 선택: [{filter_type.upper()}] — {reason}")
        return {
            "status": "selected",
            "filter_type": filter_type,
            "filter_name": info["name"],
            "description": info["description"],
            "param_guide": info["param"],
            "best_for_scenarios": info["best_for"],
            "weakness": info["weakness"],
            "agent_reason": reason,
        }

    def _get_scenario_list(self) -> Dict[str, Any]:
        """생성된 테스트 시나리오 목록과 특성 정보 반환"""
        scenarios_info = {}
        for name, scenario in self.data["scenarios"].items():
            # 배경 노이즈 수준 계산 (상단 2행 기준)
            bg_noise = float(scenario.frames[:, :2, :].std())
            max_signal = float(scenario.frames.max())

            scenarios_info[name] = {
                "description": scenario.description,
                "n_frames": scenario.n_frames,
                "max_signal": round(max_signal, 1),
                "background_noise_std": round(bg_noise, 1),
                "estimated_snr_ratio": round(max_signal / max(bg_noise, 0.1), 1),
                "expected_touches_per_frame": {
                    "min": min(len(g) for g in scenario.ground_truth),
                    "max": max(len(g) for g in scenario.ground_truth),
                }
            }

        return {
            "available_scenarios": list(scenarios_info.keys()),
            "scenario_details": scenarios_info
        }

    def _run_detection(
        self,
        scenario_name: str,
        threshold: float = 50.0,
        filter_sigma: float = 1.0,
        min_area: int = 3,
        max_area: int = 60,
        median_size: int = 3,
    ) -> Dict[str, Any]:
        """
        지정 시나리오에 터치 감지 알고리즘 실행

        결과는 pipeline_data["detections"][scenario_name]에 저장.
        사용할 필터 알고리즘은 pipeline_data["filter_type"]에서 읽음
        (select_filter_algorithm으로 사전에 선택 필요).
        """
        if scenario_name not in self.data["scenarios"]:
            return {"error": f"시나리오 없음: {scenario_name}"}

        filter_type = self.data.get("filter_type", "gaussian")
        scenario = self.data["scenarios"][scenario_name]

        # 터치 감지기 생성 및 실행 (선택된 필터 알고리즘 적용)
        detector = TouchDetector(
            threshold=threshold,
            filter_sigma=filter_sigma,
            min_area=min_area,
            max_area=max_area,
            filter_type=filter_type,
            median_size=median_size,
        )
        detected_seq = detector.detect_sequence(scenario.frames)

        # 에이전트 간 공유 저장소에 결과 보관
        self.data["detections"][scenario_name] = detected_seq
        self.data["params"][scenario_name] = {
            "filter_type": filter_type,
            "threshold": threshold,
            "filter_sigma": filter_sigma,
            "median_size": median_size,
            "min_area": min_area,
            "max_area": max_area,
        }

        # Claude에게 요약 통계 반환
        detection_counts = [len(d) for d in detected_seq]
        gt_counts = [len(g) for g in scenario.ground_truth]

        return {
            "status": "success",
            "scenario_name": scenario_name,
            "params_used": {
                "filter_type": filter_type,
                "threshold": threshold,
                "filter_sigma": filter_sigma if filter_type == "gaussian" else "(unused)",
                "median_size": median_size if filter_type == "median" else "(unused)",
                "min_area": min_area,
                "max_area": max_area,
            },
            "detection_summary": {
                "total_frames": scenario.n_frames,
                "avg_detected_per_frame": round(float(np.mean(detection_counts)), 2),
                "avg_expected_per_frame": round(float(np.mean(gt_counts)), 2),
                "frames_with_any_detection": sum(1 for c in detection_counts if c > 0),
                "max_detected_in_single_frame": max(detection_counts) if detection_counts else 0,
            }
        }

    # ── 메인 실행 ───────────────────────────────────────────────────────────

    def run(self, filter_type: str = None) -> str:
        """
        Args:
            filter_type: 사전 지정 필터 ("gaussian" | "median" | None).
                         None이면 Claude가 자율적으로 선택.
        """
        tools = [
            {
                "name": "select_filter_algorithm",
                "description": (
                    "노드 필터 알고리즘을 선택합니다. "
                    "run_detection 실행 전에 반드시 호출하세요."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filter_type": {
                            "type": "string",
                            "enum": ["gaussian", "median"],
                            "description": (
                                "gaussian: 연속 노이즈(EMI)에 강함 | "
                                "median: 임펄스 노이즈·물방울 제거에 강함"
                            )
                        },
                        "reason": {
                            "type": "string",
                            "description": "이 필터를 선택한 이유 (시나리오 특성 기반)"
                        }
                    },
                    "required": ["filter_type", "reason"]
                }
            },
            {
                "name": "get_scenario_list",
                "description": "처리할 테스트 시나리오 목록과 특성 정보를 반환합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "run_detection",
                "description": "지정한 시나리오에 터치 감지 알고리즘을 실행합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_name": {
                            "type": "string",
                            "description": "처리할 시나리오 이름"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "터치 감지 임계값 (기본: 50.0)"
                        },
                        "filter_sigma": {
                            "type": "number",
                            "description": "[gaussian 필터 전용] 평활화 강도 (기본: 1.0)"
                        },
                        "median_size": {
                            "type": "integer",
                            "description": "[median 필터 전용] 커널 크기, 홀수 (기본: 3)"
                        },
                        "min_area": {
                            "type": "integer",
                            "description": "최소 블롭 크기 - 노드 수 (기본: 3)"
                        },
                        "max_area": {
                            "type": "integer",
                            "description": "최대 블롭 크기 - 노드 수 (기본: 60)"
                        }
                    },
                    "required": ["scenario_name"]
                }
            }
        ]

        # 사전 지정 필터가 있으면 pipeline_data에 미리 설정
        if filter_type is not None:
            self.data["filter_type"] = filter_type
            preset_instruction = (
                f"노드 필터는 '{filter_type}'으로 사전 지정되어 있습니다. "
                f"select_filter_algorithm을 호출할 때 filter_type='{filter_type}'으로 확정하고, "
                "이 필터의 특성에 맞게 파라미터를 최적화하세요."
            )
        else:
            preset_instruction = (
                "시나리오 특성을 보고 gaussian / median 중 더 적합한 필터를 선택하세요."
            )

        result = self._tool_loop(
            system=SYSTEM_PROMPT,
            user_message=(
                "모든 테스트 시나리오에 터치 감지 알고리즘을 실행해주세요. "
                f"{preset_instruction} "
                "각 시나리오의 특성(노이즈 환경, 물방울 등)을 고려하여 "
                "최적의 파라미터를 선택하세요."
            ),
            tools=tools,
            tool_handlers={
                "select_filter_algorithm": self._select_filter_algorithm,
                "get_scenario_list": self._get_scenario_list,
                "run_detection": self._run_detection,
            }
        )

        print(f"  → {len(self.data['detections'])}개 시나리오 처리 완료")
        return result
