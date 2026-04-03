"""
ReportAgent: 최종 평가 리포트 생성 에이전트
============================================
역할: 모든 에이전트의 결과를 종합하여 전문적인 평가 리포트를 작성합니다.

Claude가 담당하는 부분:
    - 전체 성능 종합 판단
    - 강점/약점 도출
    - 개선 방향 및 권고사항 작성
    - 자연어 리포트 생성

Python이 담당하는 부분:
    - 저장된 모든 데이터 취합 및 전달
"""
from typing import Dict, Any
import anthropic

from .base import BaseAgent


SYSTEM_PROMPT = """당신은 Touch IC 알고리즘 평가 전문가입니다.
종합적인 성능 평가 리포트를 작성합니다.

리포트 구성:
1. 📊 전체 평가 요약
   - 평가 개요 (시나리오 수, 프레임 수)
   - 전반적 성능 수준 (종합 판정)

2. 📋 시나리오별 성능 분석
   - 각 시나리오의 주요 지표와 해석
   - 이상 또는 특이 사항

3. ✅ 강점
   - 알고리즘이 잘 동작하는 부분

4. ⚠️ 약점 및 이슈
   - 개선이 필요한 부분
   - 특정 조건에서의 성능 저하

5. 🔧 개선 권고사항 (우선순위 순)
   - 구체적이고 실행 가능한 개선 방안
   - 알고리즘 파라미터 튜닝 포인트

6. 📌 결론

합격 기준:
    위치 오차  : mean_error_px < 2.0
    지터       : jitter_rms_px < 1.5
    SNR        : snr_db > 20 dB
    검출 F1    : > 0.90

get_all_results 도구로 데이터를 가져온 후 전문적인 리포트를 작성해주세요."""


class ReportAgent(BaseAgent):

    # ── Tool 구현 메서드 ────────────────────────────────────────────────────

    def _get_all_results(self) -> Dict[str, Any]:
        """파이프라인 전체 데이터 취합 및 반환"""
        return {
            "evaluation_overview": {
                "total_scenarios": len(self.data["scenarios"]),
                "scenarios": {
                    name: {
                        "description": s.description,
                        "n_frames": s.n_frames,
                        "panel_size": f"{s.panel_shape[0]}H × {s.panel_shape[1]}W",
                    }
                    for name, s in self.data["scenarios"].items()
                }
            },
            "algorithm_parameters": self.data.get("params", {}),
            "performance_metrics": self.data["metrics"],
            "pass_criteria": {
                "mean_error_px": {"threshold": 2.0, "unit": "sensor nodes", "direction": "lower_is_better"},
                "jitter_rms_px": {"threshold": 1.5, "unit": "sensor nodes", "direction": "lower_is_better"},
                "snr_db": {"threshold": 20.0, "unit": "dB", "direction": "higher_is_better"},
                "f1": {"threshold": 0.90, "unit": "score", "direction": "higher_is_better"},
            }
        }

    # ── 메인 실행 ───────────────────────────────────────────────────────────

    def run(self) -> str:
        tools = [
            {
                "name": "get_all_results",
                "description": (
                    "평가 파이프라인의 전체 결과를 반환합니다. "
                    "시나리오 정보, 알고리즘 파라미터, 모든 성능 지표를 포함합니다."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

        result = self._tool_loop(
            system=SYSTEM_PROMPT,
            user_message=(
                "Touch IC 알고리즘 평가 리포트를 작성해주세요. "
                "모든 시나리오를 포함한 종합적인 분석과 구체적인 개선 권고사항을 제시해주세요."
            ),
            tools=tools,
            tool_handlers={
                "get_all_results": self._get_all_results,
            }
        )

        return result
