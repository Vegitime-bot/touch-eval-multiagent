"""
BaseAgent: LLM tool-use 루프를 처리하는 기반 클래스
======================================================

두 가지 백엔드를 지원합니다:
    anthropic  → Anthropic Python SDK  (claude-opus-4-6 등)
    openai     → OpenAI-compatible SDK (OpenAI / Ollama / vLLM / LM Studio 등)

백엔드 선택:
    .env 의 LLM_BACKEND 값으로 자동 선택됩니다.
    기본값은 "anthropic" 입니다.

         [User Message]
               ↓
         [LLM API] ←──────────────────────┐
               ↓                          │
     stop == "tool_use/tool_calls"?       │
          Yes ↓         No ↓              │
    [Tool 실행]    [텍스트 응답 반환]      │
          ↓                               │
    [tool_result 전달] ────────────────────┘

Anthropic 포맷과 OpenAI 포맷의 차이를 내부에서 정규화하므로
각 에이전트 코드는 백엔드와 무관하게 동일하게 동작합니다.
"""
import json
from typing import List, Dict, Any, Callable


class BaseAgent:

    ANTHROPIC_MODEL = "claude-opus-4-6"
    OPENAI_DEFAULT_MODEL = "gpt-4o"

    def __init__(self, client, pipeline_data: Dict[str, Any],
                 backend: str = "anthropic", model: str = None):
        self.client = client
        self.data = pipeline_data
        self.backend = backend
        self.model = model or (
            self.ANTHROPIC_MODEL if backend == "anthropic" else self.OPENAI_DEFAULT_MODEL
        )

    # ── OpenAI tool schema 변환 ─────────────────────────────────────────────

    @staticmethod
    def _to_openai_tools(tools: List[Dict]) -> List[Dict]:
        """
        Anthropic tool 정의 → OpenAI function 정의 변환

        Anthropic:  {"name": ..., "description": ..., "input_schema": {...}}
        OpenAI:     {"type": "function", "function": {"name": ..., "parameters": {...}}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    # ── Anthropic tool-use 루프 ─────────────────────────────────────────────

    def _tool_loop_anthropic(
        self,
        system: str,
        user_message: str,
        tools: List[Dict],
        tool_handlers: Dict[str, Callable],
    ) -> str:
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tools,
            )

            # ── 응답 완료 ────────────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # ── 도구 사용 요청 ───────────────────────────────────────────────
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = tool_handlers.get(block.name)
                        if handler:
                            try:
                                result = handler(**block.input)
                            except Exception as e:
                                result = {"error": f"실행 오류: {str(e)}"}
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     json.dumps(result, ensure_ascii=False, default=str),
                            })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user",      "content": tool_results})
            else:
                break

        return ""

    # ── OpenAI-compatible tool-use 루프 ────────────────────────────────────

    def _tool_loop_openai(
        self,
        system: str,
        user_message: str,
        tools: List[Dict],
        tool_handlers: Dict[str, Callable],
    ) -> str:
        oai_tools = self._to_openai_tools(tools) if tools else []
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]

        while True:
            kwargs: Dict[str, Any] = {
                "model":      self.model,
                "messages":   messages,
                "max_tokens": 4096,
            }
            if oai_tools:
                kwargs["tools"] = oai_tools

            response = self.client.chat.completions.create(**kwargs)
            choice   = response.choices[0]
            msg      = choice.message

            # ── 응답 완료 ────────────────────────────────────────────────────
            if choice.finish_reason in ("stop", "end_turn", "length"):
                return msg.content or ""

            # ── 도구 사용 요청 ───────────────────────────────────────────────
            if choice.finish_reason == "tool_calls":
                # assistant 메시지 (tool_calls 포함)
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content,
                    "tool_calls": [
                        {
                            "id":   tc.id,
                            "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # 각 tool 실행 결과를 tool 메시지로 추가
                for tc in msg.tool_calls:
                    handler = tool_handlers.get(tc.function.name)
                    if handler:
                        try:
                            args   = json.loads(tc.function.arguments)
                            result = handler(**args)
                        except Exception as e:
                            result = {"error": f"실행 오류: {str(e)}"}

                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      json.dumps(result, ensure_ascii=False, default=str),
                        })
            else:
                break

        return ""

    # ── 공용 진입점 (백엔드 자동 분기) ─────────────────────────────────────

    def _tool_loop(
        self,
        system: str,
        user_message: str,
        tools: List[Dict],
        tool_handlers: Dict[str, Callable],
    ) -> str:
        if self.backend == "openai":
            return self._tool_loop_openai(system, user_message, tools, tool_handlers)
        return self._tool_loop_anthropic(system, user_message, tools, tool_handlers)
