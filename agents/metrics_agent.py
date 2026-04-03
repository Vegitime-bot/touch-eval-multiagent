"""
MetricsAgent: 성능 지표 계산 및 해석 전문 에이전트
===================================================
역할: 알고리즘 실행 결과를 정량적 지표로 변환하고 의미를 해석합니다.

Claude가 담당하는 부분:
    - 수치의 의미 해석 (어떤 지표가 좋은지/나쁜지)
    - 시나리오 간 성능 차이 분석
    - 문제 있는 시나리오 식별

Python이 담당하는 부분:
    - 실제 accuracy / jitter / SNR 계산 (numpy 연산)
    - 계산 결과를 pipeline_data에 저장
"""
from typing import Dict, Any
from .base import BaseAgent
from core import metrics as metrics_module


SYSTEM_PROMPT = """당신은 Touch IC 알고리즘 성능 지표 분석 전문가입니다.

당신의 역할:
- 각 시나리오의 성능 지표를 계산하고 결과를 해석합니다
- 시나리오 간 성능 차이를 분석합니다

주요 성능 지표 해석 기준:
    mean_error_px  (위치 오차, 센서 노드 단위)
        우수 : < 1.0    / 양호 : 1.0~2.0    / 미흡 : > 2.0

    jitter_rms_px  (지터, 센서 노드 단위)
        우수 : < 0.8    / 양호 : 0.8~1.5    / 미흡 : > 1.5

    snr_db         (신호 대 잡음비, dB)
        우수 : > 25 dB  / 양호 : 20~25 dB   / 미흡 : < 20 dB

    f1             (검출 종합 지표)
        우수 : > 0.95   / 양호 : 0.90~0.95  / 미흡 : < 0.90

compute_metrics 도구로 각 시나리오를 측정하고,
get_all_metrics로 전체를 비교 분석한 후 해석 결과를 작성해주세요."""


class MetricsAgent(BaseAgent):

    # ── Tool 구현 메서드 ────────────────────────────────────────────────────

    def _compute_metrics(self, scenario_name: str) -> Dict[str, Any]:
        """
        특정 시나리오의 성능 지표 계산

        Python이 실제 numpy 연산을 수행하고,
        결과를 pipeline_data["metrics"]에 저장
        """
        if scenario_name not in self.data["scenarios"]:
            return {"error": f"시나리오 없음: {scenario_name}"}
        if scenario_name not in self.data["detections"]:
            return {"error": f"감지 결과 없음 (AlgorithmAgent 먼저 실행 필요): {scenario_name}"}

        scenario = self.data["scenarios"][scenario_name]
        detected_seq = self.data["detections"][scenario_name]

        # 실제 지표 계산 (core/metrics.py)
        result = metrics_module.compute_all(scenario, detected_seq)

        # 파이프라인 공유 저장소에 저장
        self.data["metrics"][scenario_name] = result

        return {"status": "success", "scenario": scenario_name, "metrics": result}

    def _get_all_metrics(self) -> Dict[str, Any]:
        """계산된 모든 시나리오의 지표를 한 번에 반환"""
        if not self.data["metrics"]:
            return {"error": "아직 계산된 지표가 없습니다"}

        # 핵심 지표만 추출하여 비교표 생성
        summary = {}
        for name, m in self.data["metrics"].items():
            summary[name] = {
                "mean_error_px": m.get("mean_error_px"),
                "jitter_rms_px": m.get("jitter_rms_px"),
                "snr_db": m.get("snr_db"),
                "f1": m.get("f1"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "fp_count": m.get("total_fp"),
                "fn_count": m.get("total_fn"),
            }

        return {
            "scenarios_measured": list(self.data["metrics"].keys()),
            "comparison_table": summary,
            "full_metrics": self.data["metrics"]
        }

    # ── 메인 실행 ───────────────────────────────────────────────────────────

    def run(self) -> str:
        tools = [
            {
                "name": "compute_metrics",
                "description": "특정 시나리오의 터치 성능 지표(정확도, 지터, SNR, F1)를 계산합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_name": {
                            "type": "string",
                            "description": "측정할 시나리오 이름"
                        }
                    },
                    "required": ["scenario_name"]
                }
            },
            {
                "name": "get_all_metrics",
                "description": "계산된 모든 시나리오의 성능 지표를 비교표 형식으로 반환합니다",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

        scenarios_str = ", ".join(self.data["detections"].keys())
        result = self._tool_loop(
            system=SYSTEM_PROMPT,
            user_message=(
                f"다음 시나리오들의 성능 지표를 계산하고 분석해주세요: {scenarios_str}. "
                "각 시나리오별로 compute_metrics를 실행한 후, "
                "get_all_metrics로 전체 비교 분석을 수행해주세요."
            ),
            tools=tools,
            tool_handlers={
                "compute_metrics": self._compute_metrics,
                "get_all_metrics": self._get_all_metrics,
            }
        )

        print(f"  → {len(self.data['metrics'])}개 시나리오 지표 계산 완료")
        return result
