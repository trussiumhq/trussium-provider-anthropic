"""Offline contract tests for the Anthropic provider plugin."""

import asyncio

import httpx
from trussium.capabilities.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatRole,
    ChatStreamErrorEvent,
    ChatStreamEvent,
)

from trussium_provider_anthropic import AnthropicChatCapability


def request(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet",
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content="Be concise."),
            ChatMessage(role=ChatRole.USER, content="hello"),
        ],
        stream=stream,
    )


def test_complete_normalizes_message_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={
                "id": "msg-1",
                "model": "claude-sonnet",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://anthropic")
    capability = AnthropicChatCapability("test-key", client=client)
    response = asyncio.run(capability.complete(request()))
    assert response.provider == "anthropic"
    assert response.choices[0].message.content == "hi"


def test_stream_normalizes_sse_lifecycle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = 'data: {"type":"message_start","message":{"id":"msg-1","model":"claude-sonnet","usage":{"input_tokens":2}}}\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\ndata: {"type":"message_stop"}'
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://anthropic")
    capability = AnthropicChatCapability("test-key", client=client)

    async def collect() -> list[ChatStreamEvent]:
        return [event async for event in capability.stream(request(True))]

    events = asyncio.run(collect())
    assert [event.type for event in events] == ["start", "delta", "end"]


def test_http_errors_are_bounded() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(429)),
        base_url="http://anthropic",
    )
    capability = AnthropicChatCapability("test-key", client=client)

    async def collect() -> list[ChatStreamEvent]:
        return [event async for event in capability.stream(request(True))]

    events = asyncio.run(collect())
    assert isinstance(events[0], ChatStreamErrorEvent)
    assert events[0].code == "anthropic_rate_limited"
