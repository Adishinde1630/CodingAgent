"""Thin wrapper around the Groq API (``groq`` SDK, JSON mode) with friendly errors."""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from app.config import get_api_key, get_model_name, redact

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Any problem talking to the LLM (message is safe to show to users)."""


class MissingAPIKeyError(LLMError):
    pass


class InvalidAPIKeyError(LLMError):
    pass


class RateLimitError(LLMError):
    pass


class ModelNotFoundError(LLMError):
    pass


class _BadJSON(LLMError):
    """The model (or Groq's JSON validator) produced invalid JSON; the caller may retry."""


class _ToolCallLeak(_BadJSON):
    """The model emitted a tool call although no tools were offered (Groq: 400 tool_use_failed)."""


NO_TOOLS_NOTE = (
    "\n\nIMPORTANT: You have NO tools, functions or file browsers available in this conversation. "
    "Never emit a tool/function call (for example repo_browser.* or functions.*). Everything you need is "
    "already in this message - answer directly with the requested JSON object only."
)
NO_TOOLS_RETRY_NOTE = (
    "\n\nYour previous reply tried to call a tool, but no tools exist. Do NOT call any tool or function. "
    "Reply with ONLY one valid JSON object that follows the requested schema - no markdown, no commentary."
)
MAX_TOOL_CALLS = 8
MAX_TOOL_ITERATIONS = 12


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output (tolerates code fences / surrounding prose)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return data


class GroqClient:
    """generate_json(stage, system, prompt) -> dict. ``stage`` is only used for error messages/testing."""

    def __init__(self, api_key: str | None = None, model: str | None = None, retries: int = 2):
        key = api_key if api_key is not None else get_api_key()
        self.api_key = (key or "").strip()
        if not self.api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Add it to your .env file (local) or to Streamlit Secrets "
                "(deployment), or set it in the environment."
            )
        self.model = (model or get_model_name()).strip()
        self.retries = retries
        from groq import Groq

        self._client = Groq(api_key=self.api_key, max_retries=0, timeout=90)

    # ------------------------------------------------------------------ public
    def generate_json(
        self,
        stage: str,
        system: str,
        prompt: str,
        max_output_tokens: int = 8000,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
    ) -> dict:
        tool_results: list[dict[str, Any]] = []
        tool_state = {
            "tool_call_count": 0,
            "executed_tools": {},
            "limit_reached": False,
            "tool_iterations": 0,
        }
        leaked_tool_call = False
        extra = ""
        for attempt in range(3):
            try:
                result = parse_json(self._generate(
                    stage,
                    system,
                    prompt + extra,
                    max_output_tokens,
                    tools=tools,
                    tool_handlers=tool_handlers,
                    tool_results=tool_results,
                    tool_state=tool_state,
                ))
                if tool_results:
                    result["_tool_results"] = tool_results
                if tools:
                    result["_tool_call_count"] = tool_state["tool_call_count"]
                    result["_tool_limit_reached"] = tool_state["limit_reached"]
                return result
            except (json.JSONDecodeError, _BadJSON) as exc:
                leaked = isinstance(exc, _ToolCallLeak)
                leaked_tool_call = leaked_tool_call or leaked
                extra = NO_TOOLS_RETRY_NOTE if leaked else (
                    "\n\nYour previous reply was not valid JSON. Reply again with ONLY one valid JSON object "
                    "that follows the requested schema - no markdown, no commentary.")
                continue
        if leaked_tool_call:
            raise LLMError(
                f"The model '{self.model}' kept trying to call tools instead of answering during '{stage}'. "
                "Switch GROQ_MODEL to a model that follows JSON mode reliably, e.g. llama-3.3-70b-versatile.")
        raise LLMError(f"The model did not return valid JSON during '{stage}'. Try again or switch GROQ_MODEL.")

    # ----------------------------------------------------------------- private
    def _generate(
        self,
        stage: str,
        system: str,
        prompt: str,
        max_output_tokens: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Callable[..., Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        tool_state: dict[str, Any] | None = None,
    ) -> str:
        if tools and tool_handlers is None:
            raise ValueError("Tool handlers are required when callable tools are provided.")
        messages = [
            {"role": "system", "content": system + ("" if tools else NO_TOOLS_NOTE)},
            {"role": "user", "content": prompt},
        ]
        remaining_tools = tools
        tool_state = tool_state if tool_state is not None else {
            "tool_call_count": 0,
            "executed_tools": {},
            "limit_reached": False,
            "tool_iterations": 0,
        }
        if tool_state["limit_reached"]:
            remaining_tools = None
        tool_handlers = tool_handlers or {}
        tool_results = tool_results if tool_results is not None else []
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                for _ in range(MAX_TOOL_ITERATIONS + 1):
                    request: dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "max_completion_tokens": max_output_tokens,
                        "temperature": 0.2,
                    }
                    if remaining_tools:
                        request["tools"] = remaining_tools
                        request["tool_choice"] = "auto"
                    else:
                        request["response_format"] = {"type": "json_object"}
                    if request.get("tools") and request.get("tool_choice") != "auto":
                        raise LLMError("Tool-enabled Groq requests must use tool_choice='auto'.")
                    tool_choice = request.get("tool_choice", "none (not sent; no tools)")
                    logger.warning(
                        "LLM CALL\nModel: %s\nStage: %s\nTools enabled: %s\n"
                        "Tool count: %d\nTool choice: %s",
                        self.model,
                        stage,
                        "yes" if remaining_tools else "no",
                        len(remaining_tools or []),
                        tool_choice,
                    )

                    response = self._client.chat.completions.create(**request)
                    choice = response.choices[0]
                    message = choice.message
                    calls = getattr(message, "tool_calls", None) or []
                    if calls:
                        tool_state["tool_iterations"] += 1
                        if tool_state["tool_iterations"] > MAX_TOOL_ITERATIONS:
                            tool_state["limit_reached"] = True
                            logger.warning(
                                "Tool-call iteration limit reached; continuing with cached results."
                            )
                            return json.dumps({
                                "relevant_files": [],
                                "_tool_limit_reached": True,
                            })
                        messages.append({
                            "role": "assistant",
                            "content": getattr(message, "content", None),
                            "tool_calls": [
                                call.model_dump(exclude_none=True)
                                if hasattr(call, "model_dump")
                                else {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments,
                                    },
                                }
                                for call in calls
                            ],
                        })
                        for call in calls:
                            name = call.function.name
                            arguments = call.function.arguments or "{}"
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except json.JSONDecodeError as exc:
                                    raise LLMError(f"Groq returned invalid arguments for tool {name}.") from exc
                            if not isinstance(arguments, dict):
                                raise LLMError(f"Groq returned invalid arguments for tool {name}.")
                            signature = json.dumps(
                                [name, arguments],
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            )
                            cached_result = tool_state["executed_tools"].get(signature)
                            if cached_result is not None:
                                logger.warning(
                                    "Duplicate tool call detected — using cached result."
                                )
                                result_text = cached_result
                            elif (not remaining_tools
                                  or tool_state["tool_call_count"] >= MAX_TOOL_CALLS):
                                tool_state["limit_reached"] = True
                                result_text = (
                                    "Tool-call limit reached. Continuing with the information already collected."
                                )
                            else:
                                handler_name = name.replace(".", "_")
                                handler = tool_handlers.get(name) or tool_handlers.get(handler_name)
                                if handler is None:
                                    raise LLMError(f"Groq requested an unavailable tool: {name}")
                                tool_state["tool_call_count"] += 1
                                logger.info(
                                    "Tool: %s\nArguments: %s\nTool call number: %d/%d",
                                    name,
                                    redact(json.dumps(arguments, sort_keys=True, ensure_ascii=True)),
                                    tool_state["tool_call_count"],
                                    MAX_TOOL_CALLS,
                                )
                                result = handler(**arguments)
                                tool_results.append({"name": name, "result": result})
                                result_text = json.dumps(result, ensure_ascii=True, default=str)
                                if len(result_text) > 12_000:
                                    result_text = result_text[:12_000] + "...[truncated]"
                                tool_state["executed_tools"][signature] = result_text
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": result_text,
                            })
                        if tool_state["limit_reached"] or (
                            tool_state["tool_call_count"] >= MAX_TOOL_CALLS
                        ):
                            tool_state["limit_reached"] = True
                            remaining_tools = None
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You have reached the tool-call limit. Do not request any more tools. "
                                    "Use the information already provided and return the requested JSON now."
                                ),
                            })
                        continue
                    if choice.finish_reason == "length":
                        raise LLMError("The model's reply was cut off (too long). Try a smaller task or a "
                                       "model with a larger output limit.")
                    text = message.content
                    if not text:
                        raise LLMError("Groq returned an empty response. Try rephrasing the task.")
                    return text
                tool_state["limit_reached"] = True
                logger.warning(
                    "Tool-call iteration limit reached; continuing with cached results."
                )
                return json.dumps({
                    "relevant_files": [],
                    "_tool_limit_reached": True,
                })
            except LLMError:
                raise
            except Exception as exc:
                mapped = self._map_error(exc)
                if isinstance(mapped, _BadJSON):
                    raise mapped
                if isinstance(mapped, RateLimitError):
                    raise mapped from None
                if self._status(exc) in (500, 502, 503, 504):
                    last = mapped
                    if attempt < self.retries:
                        time.sleep(4 * (attempt + 1))
                        continue
                raise mapped from None
        raise last or LLMError("Unknown Groq error.")

    @staticmethod
    def _status(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            return int(status) if status is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _retry_after(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        if not headers:
            return ""
        for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            value = headers.get(name)
            if value:
                text = str(value).strip()
                if name == "retry-after":
                    try:
                        return f"{float(text):g} seconds"
                    except ValueError:
                        return text
                return text
        return ""

    def _map_error(self, exc: Exception) -> LLMError:
        code = self._status(exc)
        raw = getattr(exc, "message", None) or str(exc)
        message = redact(raw)
        low = message.lower()
        if "tool_use_failed" in low or "tool choice is none" in low:
            return _ToolCallLeak("model called a tool although none were provided")
        if "json_validate_failed" in low or "failed to generate json" in low:
            return _BadJSON("invalid JSON")
        if code == 429 or "rate limit" in low:
            detail = message.strip() or "The Groq API rate limit has been reached."
            retry_after = self._retry_after(exc)
            retry = f" Retry after {retry_after}." if retry_after else ""
            return RateLimitError(
                f"Groq rate limit reached: {detail}.{retry} Please wait and try again."
            )
        if code == 401 or "invalid api key" in low or "invalid_api_key" in low:
            return InvalidAPIKeyError("Groq rejected the API key (invalid or expired). Check GROQ_API_KEY.")
        if code == 413 or "request too large" in low or "request_too_large" in low:
            return LLMError("The request is too large for this Groq model/plan (tokens-per-minute limit). "
                            "Use a smaller task or set GROQ_MODEL to a model with a higher limit.")
        if code == 404 or "decommissioned" in low or "does not exist" in low or "model_not_found" in low:
            return ModelNotFoundError(f"Model '{self.model}' is not available on Groq. Set GROQ_MODEL to a "
                                      "current model (e.g. llama-3.3-70b-versatile).")
        if code in (500, 502, 503, 504):
            return LLMError("Groq is temporarily unavailable. Please retry shortly.")
        if code in (400, 403):
            return LLMError(f"Groq rejected the request ({code}): {message[:300]}")
        if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            return LLMError("Could not reach the Groq API (network problem or timeout).")
        return LLMError(f"Could not get a response from Groq: {message[:300]}")
