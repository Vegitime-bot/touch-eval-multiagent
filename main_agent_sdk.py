"""
Touch IC Evaluation - Claude CLI Agent SDK 방식
================================================

API 키 없이 Claude CLI 인증을 사용하여 실행합니다.

기존 방식과의 비교:
  [main.py]          Anthropic SDK + API 키 필요
  [main_llm_orchestrator.py]  Anthropic SDK + API 키 필요
  [이 파일]          Claude CLI (claude -p) + 로그인 계정 사용

아키텍처:
  main_agent_sdk.py
      └─ claude -p (Orchestrator)        ← subprocess
              ├─ Bash: python3 pipeline_runner.py show-status
              ├─ Agent: data_agent        ← --agents 플래그로 정의
              │       └─ Bash: python3 pipeline_runner.py generate-data
              ├─ Agent: algorithm_agent
              │       └─ Bash: python3 pipeline_runner.py run-algorithm ...
              ├─ Agent: metrics_agent
              │       └─ Bash: python3 pipeline_runner.py compute-metrics
              └─ Agent: report_agent
                      └─ Bash: python3 pipeline_runner.py show-status (결과 읽기)

에이전트 간 데이터 공유:
  pipeline_output/
      ├─ frames.npz           (DataAgent 생성)
      ├─ scenarios_meta.json  (DataAgent 생성)
      ├─ detections.json      (AlgorithmAgent 생성)
      ├─ params.json          (AlgorithmAgent 생성)
      └─ metrics.json         (MetricsAgent 생성)

사용법:
    python3 main_agent_sdk.py
"""

import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "pipeline_output")


# ── 서브에이전트 정의 ────────────────────────────────────────────────────────
# Orchestrator가 --agents 플래그로 호출할 수 있는 전문 에이전트들

AGENTS = {
    "data_agent": {
        "description": (
            "터치 IC 평가용 테스트 시나리오(프레임 데이터)를 생성하는 전문 에이전트. "
            "python3 pipeline_runner.py generate-data 명령을 실행하여 "
            "frames.npz와 scenarios_meta.json을 생성합니다."
        ),
        "prompt": f"""당신은 Touch IC 테스트 데이터 생성 전문가입니다.

다음 명령을 실행하여 테스트 시나리오를 생성하세요:
  python3 {PROJECT_DIR}/pipeline_runner.py generate-data

출력 JSON에서 각 시나리오의 통계(max_signal, bg_noise_std, panel_size)를 확인하고
간략히 요약하여 보고하세요.
""",
        "tools": ["Bash"],
    },

    "algorithm_agent": {
        "description": (
            "터치 감지 알고리즘을 실행하는 전문 에이전트. "
            "시나리오별 특성(EMI 노이즈, 멀티터치, 워터 블롭 등)에 맞는 "
            "최적 파라미터를 선택하여 run-algorithm 명령을 실행합니다. "
            "현재 파이프라인 상태를 확인한 후 알고리즘을 실행합니다."
        ),
        "prompt": f"""당신은 Touch IC 알고리즘 파라미터 전문가입니다.

먼저 현재 상태를 확인하세요:
  python3 {PROJECT_DIR}/pipeline_runner.py show-status

그 다음 scenarios_meta.json의 시나리오 특성을 파악하고
아래 파라미터 가이드를 참고하여 적절한 파라미터로 알고리즘을 실행하세요:

  python3 {PROJECT_DIR}/pipeline_runner.py run-algorithm \\
    --threshold <값> --sigma <값> --min-area <값> --max-area <값>

파라미터 가이드:
  - single_touch_drag : threshold=55, sigma=1.2, min-area=4, max-area=40
  - multi_touch_pinch : threshold=50, sigma=1.0, min-area=3, max-area=25
  - emi_noise         : threshold=65, sigma=1.5, min-area=5, max-area=35
  - water_blob        : threshold=45, sigma=0.8, min-area=6, max-area=80

시나리오가 여러 개이면 균형 잡힌 하나의 파라미터 세트로 실행합니다.
결과 JSON의 detection_stats를 분석하여 보고하세요.
""",
        "tools": ["Bash", "Read"],
    },

    "metrics_agent": {
        "description": (
            "터치 감지 성능 지표(accuracy, jitter, SNR, F1)를 계산하는 전문 에이전트. "
            "compute-metrics 명령을 실행하고 결과를 분석합니다."
        ),
        "prompt": f"""당신은 Touch IC 성능 평가 전문가입니다.

다음 명령으로 성능 지표를 계산하세요:
  python3 {PROJECT_DIR}/pipeline_runner.py compute-metrics

합격 기준:
  - mean_error_px < 2.0  (위치 정확도)
  - jitter_rms_px < 1.5  (지터)
  - snr_db > 20.0        (신호 대 잡음비)
  - f1 > 0.90            (검출 성능)

결과 JSON에서 각 시나리오의 PASS/FAIL 상태와 실패한 지표를 분석하여 보고하세요.
""",
        "tools": ["Bash"],
    },

    "report_agent": {
        "description": (
            "모든 평가 결과를 종합하여 최종 리포트를 작성하는 전문 에이전트. "
            "pipeline_output 디렉토리의 파일들을 읽어 종합적인 분석을 수행합니다."
        ),
        "prompt": f"""당신은 Touch IC 알고리즘 평가 전문가입니다.

다음 파일들을 읽어 종합 평가 리포트를 작성하세요:
  - {OUTPUT_DIR}/scenarios_meta.json  (시나리오 정보)
  - {OUTPUT_DIR}/params.json          (알고리즘 파라미터)
  - {OUTPUT_DIR}/metrics.json         (성능 지표 전체)

또는 다음 명령으로 전체 요약을 확인하세요:
  python3 {PROJECT_DIR}/pipeline_runner.py show-status

리포트 구성:
1. 전체 평가 요약 (시나리오 수, 종합 판정)
2. 시나리오별 성능 분석 (지표 수치 + 해석)
3. 강점 (잘 동작하는 부분)
4. 약점 및 이슈 (개선 필요 부분)
5. 개선 권고사항 (우선순위 순)
6. 결론

합격 기준: mean_error<2.0 / jitter<1.5 / snr>20dB / f1>0.90
""",
        "tools": ["Bash", "Read"],
    },
}


# ── Orchestrator 프롬프트 ────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = f"""당신은 Touch IC 알고리즘 평가 파이프라인을 관리하는 오케스트레이터입니다.

사용 가능한 전문 에이전트 (Agent 도구로 호출):
  - data_agent       : 테스트 시나리오(프레임 데이터) 생성
  - algorithm_agent  : 터치 감지 알고리즘 실행
  - metrics_agent    : 성능 지표 계산 (accuracy, jitter, SNR, F1)
  - report_agent     : 종합 평가 리포트 작성

파이프라인 상태 확인 (Bash 도구 사용):
  python3 {PROJECT_DIR}/pipeline_runner.py show-status

실행 순서:
1. show-status로 현재 상태 확인
2. data_agent 호출 → 테스트 시나리오 생성
3. algorithm_agent 호출 → 알고리즘 실행
4. metrics_agent 호출 → 성능 지표 계산
5. show-status로 결과 확인
6. 성능 미달 시나리오가 있으면 algorithm_agent를 다른 파라미터로 재호출 (최대 1회)
7. report_agent 호출 → 최종 리포트 작성

합격 기준: mean_error_px < 2.0 / jitter_rms_px < 1.5 / snr_db > 20 / f1 > 0.90

지금 바로 파이프라인을 실행해주세요."""


def run_pipeline():
    """claude -p subprocess로 파이프라인 실행"""

    # ── 사전 확인 ──────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Error: 'claude' CLI를 찾을 수 없습니다.")
            print("Claude Code CLI가 설치되어 있는지 확인하세요.")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: 'claude' CLI를 찾을 수 없습니다.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("=" * 62)
    print("   Touch IC Evaluation - Claude CLI Agent SDK 방식")
    print("   (API 키 없이 Claude CLI 인증 사용)")
    print("=" * 62)
    print()
    print("Orchestrator Claude가 전문 에이전트들을 동적으로 호출합니다...")
    print()

    # ── claude -p 명령 구성 ────────────────────────────────────────────────
    cmd = [
        "claude",
        "-p", ORCHESTRATOR_PROMPT,
        "--allowedTools", "Bash,Agent,Read",
        "--agents", json.dumps(AGENTS),
        "--permission-mode", "bypassPermissions",
        "--output-format", "text",
    ]

    print("[실행 명령 구조]")
    print("  claude -p <orchestrator_prompt>")
    print("    --allowedTools Bash,Agent,Read")
    print("    --agents {data_agent, algorithm_agent, metrics_agent, report_agent}")
    print("    --permission-mode bypassPermissions")
    print()
    print("-" * 62)
    print()

    # ── 실행 ──────────────────────────────────────────────────────────────
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_DIR,
            bufsize=1,
        )

        # 실시간 출력
        output_lines = []
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

        process.wait()

        print()
        print("-" * 62)

        if process.returncode == 0:
            print()
            print("=" * 62)
            print("   파이프라인 완료")
            print("=" * 62)

            # 최종 상태 출력
            status_result = subprocess.run(
                ["python3", os.path.join(PROJECT_DIR, "pipeline_runner.py"), "show-status"],
                capture_output=True, text=True, cwd=PROJECT_DIR
            )
            if status_result.returncode == 0:
                try:
                    status = json.loads(status_result.stdout)
                    print()
                    print("[파이프라인 최종 상태]")
                    for step, done in status.get("completed_steps", {}).items():
                        mark = "✓" if done else "✗"
                        print(f"  {mark} {step}")
                    if status.get("metrics_summary"):
                        print()
                        print("[성능 요약]")
                        for name, m in status["metrics_summary"].items():
                            f1   = m.get("f1", 0)
                            err  = m.get("mean_error_px", 99)
                            verdict = "PASS" if f1 > 0.90 and err < 2.0 else "FAIL"
                            print(f"  {name}: f1={f1:.3f}, error={err:.2f}px → {verdict}")
                except json.JSONDecodeError:
                    pass
        else:
            print(f"\nError: claude 프로세스가 {process.returncode}로 종료되었습니다.")

    except KeyboardInterrupt:
        print("\n\n[사용자에 의해 중단됨]")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
