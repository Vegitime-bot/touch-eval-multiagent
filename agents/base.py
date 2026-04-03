"""
BaseAgent: Claude tool-use 루프를 처리하는 기반 클래스
=======================================================

Multi-Agent 패턴의 핵심:
    각 에이전트는 자신만의 system_prompt(역할 정의)와 tools(사용 가능 도구)를 가집니다.
    Claude가 tool_use를 요청하면 → 실제 Python 함수 실행 → 결과를 다시 Claude에게 전달
    이 루프를 stop_reason이 'end_turn'이 될 때까지 반복합니다.

         [User Message]
               ↓
         [Claude API] ←──────────────────────┐
               ↓                             │
     stop_reason == "tool_use"?              │
          Yes ↓         No ↓                 │
    [Tool 실행]    [텍스트 응답 반환]          │
          ↓                                  │
    [tool_result 전달] ──────────────────────┘
"""
import json
from typing import List, Dict, Any, Callable
import anthropic


class BaseAgent:
    """
    Claude API tool-use 루프를 처리하는 기반 클래스

    상속 후 구현할 항목:
        - run(): 메인 실행 메서드 (tools, tool_handlers, system prompt 정의 후 _tool_loop 호출)

    에이전트 간 데이터 공유:
        - self.data: 파이프라인 전체가 공유하는 딕셔너리
        - 한 에이전트가 쓴 데이터를 다음 에이전트가 읽음
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, client: anthropic.Anthropic, pipeline_data: Dict[str, Any]):
        self.client = client
        self.data = pipeline_data  # 에이전트 간 공유 데이터 (참조 전달)

    def _tool_loop(
        self,
        system: str,
        user_message: str,
        tools: List[Dict],
        tool_handlers: Dict[str, Callable]
    ) -> str:
        """
        Claude tool-use 반복 루프

        Args:
            system       : 에이전트 역할을 정의하는 시스템 프롬프트
            user_message : 에이전트에게 전달하는 초기 요청
            tools        : Claude가 사용할 수 있는 도구 정의 목록 (JSON Schema 형식)
            tool_handlers: 도구 이름 → 실행 함수 매핑

        Returns:
            Claude의 최종 텍스트 응답
        """
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tools
            )

            # ── Case 1: 응답 완료 ────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # ── Case 2: 도구 사용 요청 ───────────────────────────────────────
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        handler = tool_handlers.get(block.name)
                        if handler:
                            try:
                                # 실제 Python 함수 실행
                                result = handler(**block.input)
                            except Exception as e:
                                result = {"error": f"실행 오류: {str(e)}"}

                            # 결과를 JSON 문자열로 직렬화 (numpy 등 처리)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(
                                    result, ensure_ascii=False, default=str
                                )
                            })

                # 대화 히스토리 업데이트: assistant 응답 + tool 결과
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # 예상치 못한 stop_reason (max_tokens 등)
                break

        return ""
