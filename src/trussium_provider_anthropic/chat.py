"""Anthropic Messages API chat adapter."""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from trussium.capabilities.chat import (
    ChatCapability,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    ChatStreamDeltaEvent,
    ChatStreamEndEvent,
    ChatStreamErrorEvent,
    ChatStreamEvent,
    ChatStreamStartEvent,
    FinishReason,
    TokenUsage,
)
from trussium.errors import ProviderError


class AnthropicProviderError(ProviderError):
    """Raised when an Anthropic response cannot be normalized."""


class AnthropicChatCapability(ChatCapability):
    """Normalize Anthropic's Messages API into Trussium chat contracts."""

    provider_name = "anthropic"
    provider_display_name = "Anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"))
        self._owns_client = client is None
        self._api_key = api_key

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = await self._client.post(
                "/messages",
                headers=self._headers(),
                json=self._payload(request, stream=False),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise self._status_error(error.response.status_code) from error
        except httpx.RequestError as error:
            raise AnthropicProviderError(
                "Anthropic connection failed", code="anthropic_connection"
            ) from error
        return self._normalize_response(response.json())

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatStreamEvent]:
        response_id: str | None = None
        model = request.model
        input_tokens = 0
        try:
            async with self._client.stream(
                "POST",
                "/messages",
                headers=self._headers(),
                json=self._payload(request, stream=True),
            ) as response:
                if response.is_error:
                    yield self._stream_error(self._status_error(response.status_code))
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                        event_type = event.get("type")
                        if event_type == "message_start":
                            message = event["message"]
                            response_id = str(message["id"])
                            model = str(message.get("model", model))
                            input_tokens = int(message.get("usage", {}).get("input_tokens", 0))
                            yield ChatStreamStartEvent(
                                id=response_id, provider=self.provider_name, model=model
                            )
                        elif event_type == "content_block_delta" and event.get("delta", {}).get(
                            "text"
                        ):
                            if response_id is None:
                                yield ChatStreamErrorEvent(
                                    id=None,
                                    code="anthropic_invalid_stream",
                                    message="Anthropic emitted content before message start.",
                                )
                                return
                            yield ChatStreamDeltaEvent(
                                id=response_id, content=str(event["delta"]["text"])
                            )
                        elif event_type == "message_delta":
                            if response_id is None:
                                yield ChatStreamErrorEvent(
                                    id=None,
                                    code="anthropic_invalid_stream",
                                    message="Anthropic emitted completion before message start.",
                                )
                                return
                            output_tokens = int(event.get("usage", {}).get("output_tokens", 0))
                            yield ChatStreamEndEvent(
                                id=response_id,
                                finish_reason=self._finish_reason(
                                    event.get("delta", {}).get("stop_reason")
                                ),
                                usage=TokenUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    total_tokens=input_tokens + output_tokens,
                                ),
                            )
                            return
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        yield ChatStreamErrorEvent(
                            id=response_id,
                            code="anthropic_invalid_stream",
                            message="Anthropic returned an invalid streaming event.",
                        )
                        return
        except httpx.RequestError:
            yield ChatStreamErrorEvent(
                id=response_id,
                code="anthropic_connection",
                message="Anthropic connection failed",
            )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    @staticmethod
    def _payload(request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        system = [
            message.content for message in request.messages if message.role == ChatRole.SYSTEM
        ]
        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
            if message.role != ChatRole.SYSTEM
        ]
        return {
            "model": request.model,
            "max_tokens": request.max_output_tokens or 1024,
            "messages": messages,
            "system": "\n\n".join(system) if system else None,
            "temperature": request.temperature,
            "stream": stream,
        }

    def _normalize_response(self, payload: Mapping[str, Any]) -> ChatCompletionResponse:
        try:
            text = "".join(
                str(block["text"]) for block in payload["content"] if block.get("type") == "text"
            )
            if not text:
                raise ValueError("missing text content")
            usage = payload["usage"]
            input_tokens = int(usage["input_tokens"])
            output_tokens = int(usage["output_tokens"])
            return ChatCompletionResponse(
                id=str(payload["id"]),
                provider=self.provider_name,
                model=str(payload["model"]),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role=ChatRole.ASSISTANT, content=text),
                        finish_reason=self._finish_reason(payload.get("stop_reason")),
                    )
                ],
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AnthropicProviderError(
                "Anthropic returned an invalid message response",
                code="anthropic_invalid_response",
            ) from error

    @staticmethod
    def _finish_reason(value: Any) -> FinishReason:
        return {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALL,
        }.get(value, FinishReason.ERROR)

    @staticmethod
    def _status_error(status_code: int) -> AnthropicProviderError:
        code = "anthropic_rate_limited" if status_code == 429 else "anthropic_http_error"
        return AnthropicProviderError("Anthropic request failed", code=code)

    @staticmethod
    def _stream_error(error: AnthropicProviderError) -> ChatStreamErrorEvent:
        return ChatStreamErrorEvent(id=None, code=error.code, message=error.message)
