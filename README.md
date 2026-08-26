# Trussium Anthropic provider plugin

Standalone Python adapter for Anthropic's Messages API. The application owner
installs this package, supplies an API key through its secret-management
boundary, and explicitly registers `AnthropicChatCapability`.

```python
from trussium.capabilities import CapabilityRegistry
from trussium_provider_anthropic import AnthropicChatCapability

registry = CapabilityRegistry()
registry.register("chat.completions", AnthropicChatCapability(api_key="secret"))
registry.seal()
```

The adapter normalizes Messages API JSON and SSE responses, maps bounded HTTP
failures, and does not log credentials or payloads. Tool-use and multimodal
extensions are intentionally deferred from this initial adapter. Dynamic
loading remains outside the package; follow Trussium ADR-0008.

Run offline tests with `uv run pytest`.
