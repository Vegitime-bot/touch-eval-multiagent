"""
Touch IC Evaluation - LLM-Powered Orchestrator
===============================================

기존 main.py와의 차이점:

  [main.py]  Python 코드가 순서를 하드코딩
  ─────────────────────────────────────────
  data_agent.run()
  algo_agent.run()
  metrics_agent.run()    ← 무조건 이 순서
  report_agent.run()

  [이 파일]  Claude가 에이전트 호출을 동적으로 결정
  ─────────────────────────────────────────────────
  Orchestrator Claude가:
    1. 상태를 확인하면서
    2. 어떤 에이전트를 호출할지 스스로 판단
    3. 결과가 나쁘면 다른 파라미터로 재실행 결정
    4. 완료 조건을 스스로 판단

핵심 패턴 비교:

  Python Orchestrator (main.py)
  ┌─────────────────────────────┐
  │ 실행 순서: 코드에 고정      │
  │ 재시도 여부: 없음           │
  │ 조건 분기: if/else로 수동   │
  └─────────────────────────────┘

  LLM Orchestrator (이 파일)
  ┌─────────────────────────────┐
  │ 실행 순서: Claude가 결정    │
  │ 재시도 여부: Claude가 판단  │
  │ 조건 분기: 자연어로 추론    │
  └─────────────────────────────┘

사용법:
    python3 main_llm_orchestrator.py
"""

import os
import sys
import json
import anthropic

from agents.data_agent import DataAgent
from agents.algorithm_agent import AlgorithmAgent
from agents.metrics_agent import MetricsAgent
from agents.report_agent import ReportAgent


# ── Orchestrator 시스템 프롬프트 ─────────────────────────────────────────────
ORCHESTRATOR_SYSTEM = """당신은 Touch IC 알고리즘 평가 파이프라인을 관리하는 오케스트레이터입니다.

사용 가능한 전문 에이전트:
  - run_data_agent      : 테스트 시나리오(프레임 데이터) 생성
  - run_algorithm_agent : 터치 감지 알고리즘 실행
  - run_metrics_agent   : 성능 지표 계산 (accuracy, jitter, SNR, F1)
  - run_report_agent    : 종합 평가 리포트 작성
  - get_pipeline_status : 현재 파이프라인 진행 상태 확인

당신의 책임:
1. 적절한 순서로 에이전트를 호출하여 평가를 완성하세요
2. 각 단계 완료 후 get_pipeline_status로 결과를 확인하세요
3. 성능 지표가 기준 미달(mean_error > 2.0 또는 f1 < 0.85)인 시나리오가 있으면
   run_algorithm_agent를 재실행하여 개선을 시도하세요 (최대 1회)
4. 모든 단계 완료 후 run_report_agent로 최종 리포트를 작성하세요

합격 기준:
  mean_error_px < 2.0  /  jitter_rms_px < 1.5  /  snr_db > 20  /  f1 > 0.90"""


class LLMOrchestrator:
    """
    Claude가 직접 에이전트 호출 순서와 재실행 여부를 판단하는 오케스트레이터

    Orchestrator 자체가 tool-use 루프를 가진 Claude 에이전트입니다.
    각 tool이 실제 전문 에이전트(DataAgent, AlgorithmAgent...)를 실행합니다.

    즉 2단계 중첩 구조:
        Orchestrator Claude
            └─ tool: run_data_agent()
                    └─ DataAgent Claude (내부 tool-use 루프)
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, client: anthropic.Anthropic):
        self.client = client

        # 파이프라인 공유 데이터 (서브 에이전트들이 공유)
        self.pipeline_data = {
            "scenarios":  {},
            "detections": {},
            "params":     {},
            "metrics":    {},
        }

        # 각 에이전트 인스턴스 (오케스트레이터가 호출)
        self._data_agent      = DataAgent(client, self.pipeline_data)
        self._algo_agent      = AlgorithmAgent(client, self.pipeline_data)
        self._metrics_agent   = MetricsAgent(client, self.pipeline_data)
        self._report_agent    = ReportAgent(client, self.pipeline_data)

        # 실행 이력 추적
        self._execution_log = []

    # ── Orchestrator가 사용할 Tool 함수들 ────────────────────────────────────

    def _get_pipeline_status(self) -> dict:
        """현재 파이프라인 진행 상태 반환 (Orchestrator Claude가 판단 근거로 사용)"""
        metrics_summary = {}
        for name, m in self.pipeline_data["metrics"].items():
            metrics_summary[name] = {
                "mean_error_px": m.get("mean_error_px"),
                "f1": m.get("f1"),
                "snr_db": m.get("snr_db"),
                "jitter_rms_px": m.get("jitter_rms_px"),
                # 기준 대비 pass/fail
                "pass": (
                    m.get("mean_error_px", 99) < 2.0 and
                    m.get("f1", 0) > 0.90 and
                    m.get("snr_db", 0) > 20.0
                )
            }

        return {
            "completed_steps": {
                "data_generation":  len(self.pipeline_data["scenarios"]) > 0,
                "algorithm_run":    len(self.pipeline_data["detections"]) > 0,
                "metrics_computed": len(self.pipeline_data["metrics"]) > 0,
            },
            "scenario_count":   len(self.pipeline_data["scenarios"]),
            "detection_count":  len(self.pipeline_data["detections"]),
            "metrics_count":    len(self.pipeline_data["metrics"]),
            "metrics_summary":  metrics_summary,
            "execution_history": self._execution_log,
        }

    def _run_data_agent(self) -> dict:
        """DataAgent 실행: 테스트 시나리오 생성"""
        print("  [Orchestrator → DataAgent 호출]")
        self._data_agent.run()
        self._execution_log.append("DataAgent 완료")
        return {
            "status": "completed",
            "generated_scenarios": list(self.pipeline_data["scenarios"].keys()),
        }

    def _run_algorithm_agent(self) -> dict:
        """AlgorithmAgent 실행: 터치 감지 알고리즘"""
        print("  [Orchestrator → AlgorithmAgent 호출]")
        self._algo_agent.run()
        self._execution_log.append("AlgorithmAgent 완료")
        return {
            "status": "completed",
            "processed_scenarios": list(self.pipeline_data["detections"].keys()),
            "params_used": self.pipeline_data.get("params", {}),
        }

    def _run_metrics_agent(self) -> dict:
        """MetricsAgent 실행: 성능 지표 계산"""
        print("  [Orchestrator → MetricsAgent 호출]")
        self.pipeline_data["metrics"].clear()  # 재실행 시 초기화
        self._metrics_agent.run()
        self._execution_log.append("MetricsAgent 완료")

        # 기준 미달 시나리오 식별 (Orchestrator 판단 근거)
        failed = [
            name for name, m in self.pipeline_data["metrics"].items()
            if m.get("mean_error_px", 99) >= 2.0 or m.get("f1", 0) < 0.90
        ]
        return {
            "status": "completed",
            "measured_scenarios": list(self.pipeline_data["metrics"].keys()),
            "below_threshold_scenarios": failed,
            "all_passed": len(failed) == 0,
        }

    def _run_report_agent(self) -> dict:
        """ReportAgent 실행: 최종 리포트 생성"""
        print("  [Orchestrator → ReportAgent 호출]")
        report = self._report_agent.run()
        self._execution_log.append("ReportAgent 완료")
        # 리포트 내용은 별도 출력
        self._final_report = report
        return {
            "status": "completed",
            "report_length": len(report),
            "message": "리포트 생성 완료. 파이프라인 종료 가능.",
        }

    # ── 오케스트레이터 메인 실행 ──────────────────────────────────────────────

    def run(self) -> str:
        """
        Orchestrator Claude 실행

        이 메서드 자체가 tool-use 루프입니다.
        Claude가 어떤 에이전트를 언제 호출할지 스스로 결정합니다.
        """
        tools = [
            {
                "name": "get_pipeline_status",
                "description": "현재 파이프라인 진행 상태와 각 단계 완료 여부, 성능 지표 요약을 반환합니다",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "run_data_agent",
                "description": "DataAgent를 실행합니다. 4가지 테스트 시나리오(프레임 데이터)를 생성합니다. 가장 먼저 실행해야 합니다.",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "run_algorithm_agent",
                "description": "AlgorithmAgent를 실행합니다. 시나리오별 최적 파라미터로 터치 감지 알고리즘을 실행합니다. DataAgent 완료 후 실행 가능.",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "run_metrics_agent",
                "description": "MetricsAgent를 실행합니다. 정확도/지터/SNR/F1 등 성능 지표를 계산합니다. AlgorithmAgent 완료 후 실행 가능.",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "run_report_agent",
                "description": "ReportAgent를 실행합니다. 모든 결과를 종합한 최종 평가 리포트를 생성합니다. MetricsAgent 완료 후 실행 가능.",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            },
        ]

        tool_handlers = {
            "get_pipeline_status": self._get_pipeline_status,
            "run_data_agent":      self._run_data_agent,
            "run_algorithm_agent": self._run_algorithm_agent,
            "run_metrics_agent":   self._run_metrics_agent,
            "run_report_agent":    self._run_report_agent,
        }

        # ── Orchestrator tool-use 루프 시작 ──────────────────────────────────
        messages = [{
            "role": "user",
            "content": (
                "Touch IC 알고리즘 평가 파이프라인을 실행해주세요. "
                "에이전트들을 적절한 순서로 호출하고, "
                "성능이 기준 미달인 경우 개선을 시도한 후 최종 리포트를 완성해주세요."
            )
        }]

        self._final_report = ""

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM,
                messages=messages,
                tools=tools
            )

            if response.stop_reason == "end_turn":
                # Orchestrator의 최종 판단/요약
                for block in response.content:
                    if hasattr(block, "text"):
                        print(f"\n[Orchestrator 최종 판단]\n{block.text}\n")
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = tool_handlers.get(block.name)
                        if handler:
                            result = handler(**block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(
                                    result, ensure_ascii=False, default=str
                                )
                            })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        return self._final_report


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY 환경변수를 설정해주세요")
        sys.exit(1)

    client = anthropic.Anthropic()

    print()
    print("=" * 62)
    print("   Touch IC Evaluation - LLM-Powered Orchestrator")
    print("   (Orchestrator 자체가 Claude 에이전트)")
    print("=" * 62)
    print()
    print("Orchestrator Claude가 에이전트 호출 순서를 스스로 결정합니다...")
    print()

    orchestrator = LLMOrchestrator(client)
    report = orchestrator.run()

    if report:
        print("=" * 62)
        print("   최종 평가 리포트")
        print("=" * 62)
        print(report)
        print("=" * 62)


if __name__ == "__main__":
    main()
