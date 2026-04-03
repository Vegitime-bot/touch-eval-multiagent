"""
DataAgent: 테스트 데이터 생성 전문 에이전트
============================================
역할: 터치 IC 알고리즘 평가에 필요한 테스트 시나리오를 생성합니다.

Claude가 담당하는 부분:
    - 어떤 시나리오들이 필요한지 판단
    - 각 시나리오의 평가 목적 설명
    - 생성 결과 요약

Python이 담당하는 부분:
    - 실제 numpy 프레임 데이터 생성 (TouchFrameSimulator)
    - pipeline_data에 시나리오 저장
"""
from typing import Dict, Any
import numpy as np
import anthropic

from .base import BaseAgent
from core.frame_simulator import TouchFrameSimulator


SYSTEM_PROMPT = """당신은 Touch IC 평가를 위한 테스트 데이터 생성 전문가입니다.

당신의 역할:
- 다양한 터치 조건에 대한 테스트 시나리오를 생성합니다
- 각 시나리오의 특성과 평가 목적을 명확히 설명합니다

사용 가능한 시나리오:
1. single_touch_drag  - 단일 터치 드래그 (기본 정확도, 선형성, 지터 평가)
2. multi_touch_pinch  - 멀티터치 피치 아웃 (복수 터치 감지 능력 평가)
3. emi_noise          - EMI 노이즈 환경 (노이즈 내성, SNR 평가)
4. water_blob         - 물방울 간섭 (False positive 제거 능력 평가)

list_scenarios 도구로 시나리오 목록을 확인한 후,
generate_scenario 도구로 모든 시나리오를 순서대로 생성하세요.
완료 후 생성된 시나리오들을 간략히 요약해주세요."""


class DataAgent(BaseAgent):

    def __init__(self, client: anthropic.Anthropic, pipeline_data: Dict[str, Any]):
        super().__init__(client, pipeline_data)
        self.simulator = TouchFrameSimulator()

    # ── Tool 구현 메서드 ────────────────────────────────────────────────────

    def _list_scenarios(self) -> Dict[str, Any]:
        """사용 가능한 테스트 시나리오 목록과 설명 반환"""
        return {
            "available_scenarios": [
                {
                    "name": "single_touch_drag",
                    "description": "단일 터치 좌→우 드래그 (40 프레임)",
                    "evaluation_purpose": "위치 정확도, 선형성, 지터 측정"
                },
                {
                    "name": "multi_touch_pinch",
                    "description": "두 손가락 피치 아웃 (40 프레임)",
                    "evaluation_purpose": "멀티터치 감지, 두 터치 간 분리 능력"
                },
                {
                    "name": "emi_noise",
                    "description": "충전기 EMI 노이즈 환경 단일 터치 (40 프레임)",
                    "evaluation_purpose": "노이즈 내성, SNR"
                },
                {
                    "name": "water_blob",
                    "description": "물방울 false touch 시나리오 (40 프레임)",
                    "evaluation_purpose": "False positive 제거 (물방울 vs 실제 터치)"
                }
            ]
        }

    def _generate_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        지정한 시나리오 데이터 생성 및 pipeline_data에 저장

        numpy 배열은 pipeline_data에 저장하고,
        Claude에게는 통계 정보만 반환 (JSON 직렬화 가능한 형태)
        """
        generators = {
            "single_touch_drag": self.simulator.single_touch_drag,
            "multi_touch_pinch": self.simulator.multi_touch_pinch,
            "emi_noise": self.simulator.emi_noise,
            "water_blob": self.simulator.water_blob,
        }

        if scenario_name not in generators:
            return {"error": f"알 수 없는 시나리오: {scenario_name}"}

        # 실제 데이터 생성
        scenario = generators[scenario_name]()

        # 에이전트 간 공유 저장소에 보관 (numpy 배열 포함)
        self.data["scenarios"][scenario_name] = scenario

        # Claude에게는 요약 통계만 반환
        return {
            "status": "success",
            "scenario_name": scenario_name,
            "description": scenario.description,
            "n_frames": scenario.n_frames,
            "panel_size": f"{scenario.panel_shape[0]}H × {scenario.panel_shape[1]}W nodes",
            "frame_stats": {
                "mean_signal": round(float(scenario.frames.mean()), 1),
                "max_signal": round(float(scenario.frames.max()), 1),
                "background_noise_std": round(
                    float(scenario.frames[:, :2, :].std()), 1
                ),
            },
            "touches_per_frame": {
                "min": min(len(g) for g in scenario.ground_truth),
                "max": max(len(g) for g in scenario.ground_truth),
            }
        }

    # ── 메인 실행 ───────────────────────────────────────────────────────────

    def run(self) -> str:
        tools = [
            {
                "name": "list_scenarios",
                "description": "사용 가능한 테스트 시나리오 목록과 평가 목적을 반환합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "generate_scenario",
                "description": "지정한 이름의 테스트 시나리오 프레임 데이터를 생성합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_name": {
                            "type": "string",
                            "description": "생성할 시나리오 이름",
                            "enum": [
                                "single_touch_drag",
                                "multi_touch_pinch",
                                "emi_noise",
                                "water_blob"
                            ]
                        }
                    },
                    "required": ["scenario_name"]
                }
            }
        ]

        result = self._tool_loop(
            system=SYSTEM_PROMPT,
            user_message="Touch IC 알고리즘 평가를 위한 모든 테스트 시나리오를 생성해주세요.",
            tools=tools,
            tool_handlers={
                "list_scenarios": self._list_scenarios,
                "generate_scenario": self._generate_scenario,
            }
        )

        print(f"  → {len(self.data['scenarios'])}개 시나리오 생성 완료: "
              f"{list(self.data['scenarios'].keys())}")
        return result
