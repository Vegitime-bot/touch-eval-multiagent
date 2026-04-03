"""
Touch IC Algorithm Evaluation - Multi-Agent Pipeline
=====================================================

4개의 전문 에이전트가 협력하여 터치 IC 알고리즘의 성능을 자동으로 평가합니다.

Architecture:
                    pipeline_data (공유 저장소)
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
[DataAgent]        [AlgorithmAgent]         [MetricsAgent]     [ReportAgent]
    │                      │                      │                  │
시나리오 생성         알고리즘 실행           지표 계산          리포트 생성
(프레임 데이터)    (파라미터 최적화)    (accuracy/jitter/SNR)   (종합 분석)
    │                      │                      │                  │
    └──► scenarios    detections ◄──► metrics ◄───┘    ◄── 전체 데이터

각 에이전트는:
    - 자신만의 system_prompt (역할 정의)
    - 전용 tools (도구 집합)
    - Claude의 판단 + Python의 계산 = 지능형 처리

사용법:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="your-api-key"
    python main.py
"""

import os
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

from agents.data_agent import DataAgent
from agents.algorithm_agent import AlgorithmAgent
from agents.metrics_agent import MetricsAgent
from agents.report_agent import ReportAgent


def _build_client(backend: str):
    """
    백엔드에 맞는 LLM 클라이언트 생성

    anthropic  → ANTHROPIC_API_KEY 사용
    openai     → OPENAI_BASE_URL / OPENAI_API_KEY 사용
    """
    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            print("Error: openai 패키지가 필요합니다: pip install openai")
            sys.exit(1)

        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key  = os.environ.get("OPENAI_API_KEY", "none")  # Ollama 등은 키 불필요
        if not base_url:
            print("Error: .env 에 OPENAI_BASE_URL을 설정해주세요")
            sys.exit(1)
        return OpenAI(base_url=base_url, api_key=api_key)

    else:  # anthropic
        import anthropic as _anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: .env 에 ANTHROPIC_API_KEY를 설정해주세요")
            sys.exit(1)
        return _anthropic.Anthropic()


def run_pipeline(filter_type: str = None) -> tuple:
    """
    Multi-Agent 평가 파이프라인 실행

    Args:
        filter_type: 노드 필터 알고리즘 사전 지정 ("gaussian" | "median" | None).
                     None이면 AlgorithmAgent가 시나리오를 보고 자율 선택.

    Returns:
        (report_text, pipeline_data) 튜플

    에이전트 실행 순서:
        1. DataAgent       → 테스트 시나리오 생성
        2. AlgorithmAgent  → 필터 알고리즘 선택 + 알고리즘 실행
        3. MetricsAgent    → 지표 계산 (AlgorithmAgent 결과 필요)
        4. ReportAgent     → 리포트 생성 (모든 결과 필요)
    """
    # ── 백엔드 및 모델 결정 (.env 기준) ───────────────────────────────────
    backend = os.environ.get("LLM_BACKEND", "anthropic").lower()
    model   = os.environ.get("OPENAI_MODEL") if backend == "openai" else None
    client  = _build_client(backend)
    agent_kwargs = {"backend": backend, "model": model}

    # ── 에이전트 간 공유 데이터 저장소 ─────────────────────────────────────
    # 각 에이전트가 이 딕셔너리에 결과를 쓰고, 다음 에이전트가 읽어 사용합니다.
    pipeline_data = {
        "scenarios":   {},  # DataAgent       → {name: TestScenario}
        "detections":  {},  # AlgorithmAgent  → {name: List[List[DetectedTouch]]}
        "params":      {},  # AlgorithmAgent  → {name: param_dict}
        "metrics":     {},  # MetricsAgent    → {name: metrics_dict}
        "filter_type": filter_type or "gaussian",  # AlgorithmAgent가 확정
    }

    filter_label   = f"[필터: {filter_type.upper()}]" if filter_type else "[필터: 자율 선택]"
    backend_label  = f"{backend.upper()}  model={model or 'default'}"

    print()
    print("=" * 62)
    print(f"   Touch IC Algorithm Evaluation  {filter_label}")
    print(f"   Backend: {backend_label}")
    print("=" * 62)

    # ── Agent 1: 테스트 데이터 생성 ────────────────────────────────────────
    print()
    print("┌─ [Agent 1] Data Agent")
    print("│  역할: 테스트 시나리오 생성 (프레임 시뮬레이션)")
    print("│  ...")
    data_agent = DataAgent(client, pipeline_data, **agent_kwargs)
    data_agent.run()
    print("└─ 완료")

    # ── Agent 2: 알고리즘 실행 ─────────────────────────────────────────────
    print()
    print("┌─ [Agent 2] Algorithm Agent")
    print("│  역할: 노드 필터 선택 + 시나리오별 파라미터 최적화 및 알고리즘 실행")
    print("│  입력: pipeline_data['scenarios'] (DataAgent 결과)")
    if filter_type:
        print(f"│  필터: {filter_type} (사전 지정)")
    else:
        print("│  필터: LLM이 시나리오를 보고 자율 선택")
    print("│  ...")
    algo_agent = AlgorithmAgent(client, pipeline_data, **agent_kwargs)
    algo_agent.run(filter_type=filter_type)
    print(f"│  확정된 필터: {pipeline_data.get('filter_type', '?').upper()}")
    print("└─ 완료")

    # ── Agent 3: 성능 지표 계산 ────────────────────────────────────────────
    print()
    print("┌─ [Agent 3] Metrics Agent")
    print("│  역할: 성능 지표 계산 및 해석 (accuracy, jitter, SNR, F1)")
    print("│  입력: pipeline_data['detections'] (AlgorithmAgent 결과)")
    print("│  ...")
    metrics_agent = MetricsAgent(client, pipeline_data, **agent_kwargs)
    metrics_output = metrics_agent.run()
    print("└─ 완료")

    # 중간 결과 출력 (교육용: 각 에이전트의 독립적 출력 확인)
    print()
    print("─── Metrics Agent 분석 결과 (요약) " + "─" * 27)
    preview = metrics_output[:600]
    if len(metrics_output) > 600:
        preview += "..."
    print(preview)

    # ── Agent 4: 리포트 생성 ───────────────────────────────────────────────
    print()
    print("┌─ [Agent 4] Report Agent")
    print("│  역할: 종합 평가 리포트 작성 (강점/약점/개선 권고)")
    print("│  입력: pipeline_data 전체 (모든 에이전트 결과)")
    print("│  ...")
    report_agent = ReportAgent(client, pipeline_data, **agent_kwargs)
    report = report_agent.run()
    print("└─ 완료")

    # ── 최종 리포트 출력 ───────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"   최종 평가 리포트  {filter_label}")
    print("=" * 62)
    print(report)
    print("=" * 62)

    return report, pipeline_data


def run_comparison() -> None:
    """
    가우시안 vs 미디언 필터를 각각 실행하고 성능 지표를 비교합니다.

    교육 목적:
        두 필터에 따라 AlgorithmAgent의 파라미터 선택과
        MetricsAgent의 해석이 어떻게 달라지는지 한눈에 확인합니다.
    """
    results = {}

    for ft in ["gaussian", "median"]:
        _, pipeline_data = run_pipeline(filter_type=ft)
        results[ft] = pipeline_data["metrics"]

    # ── 비교 표 출력 ───────────────────────────────────────────────────────
    PASS_CRITERIA = {
        "mean_error_px": ("<", 2.0),
        "jitter_rms_px": ("<", 1.5),
        "snr_db":        (">", 20.0),
        "f1":            (">", 0.90),
    }

    def judge(key, value) -> str:
        op, threshold = PASS_CRITERIA[key]
        ok = (value < threshold) if op == "<" else (value > threshold)
        return "PASS" if ok else "FAIL"

    scenarios = list(results["gaussian"].keys())

    print()
    print("=" * 70)
    print("   필터 알고리즘 비교: GAUSSIAN  vs  MEDIAN")
    print("=" * 70)

    for scenario in scenarios:
        g = results["gaussian"].get(scenario, {})
        m = results["median"].get(scenario, {})

        print(f"\n  [{scenario}]")
        print(f"  {'지표':<18} {'GAUSSIAN':>12} {'MEDIAN':>12}  {'기준'}")
        print(f"  {'-'*18} {'-'*12} {'-'*12}  {'-'*20}")

        for key, (op, thr) in PASS_CRITERIA.items():
            gv = g.get(key, float("nan"))
            mv = m.get(key, float("nan"))
            gj = judge(key, gv) if gv == gv else "N/A"
            mj = judge(key, mv) if mv == mv else "N/A"
            print(f"  {key:<18} {gv:>8.3f} {gj:>4} {mv:>8.3f} {mj:>4}   {op} {thr}")

    print()
    print("=" * 70)
    print("  [교육 포인트]")
    print("  • gaussian: 연속 분포 노이즈(EMI) 억제에 유리")
    print("  • median  : 임펄스성 노이즈·물방울 스파이크 제거에 유리")
    print("  • AlgorithmAgent는 각 필터에 맞춰 다른 파라미터를 선택함")
    print("  • MetricsAgent는 동일 기준으로 평가하지만 해석이 달라짐")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Touch IC Algorithm Evaluation - Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--filter",
        choices=["gaussian", "median"],
        default=None,
        help="노드 필터 알고리즘 지정 (기본: Claude가 자율 선택)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="gaussian / median 두 필터를 모두 실행하고 결과를 비교합니다"
    )
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        run_pipeline(filter_type=args.filter)
