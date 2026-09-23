"""Tool definition, registration, and schema export."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Handler = Callable[..., Any | Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    # Spoken filler the agent should say while this tool runs.
    #
    # MEASURED, not assumed: the server serializes the whole tools array into a
    # JSON string and injects it into the system prompt's <AVAILABLE_TOOLS>
    # block verbatim -- confirmed by grepping the container's own "Sending
    # system prompt" log line for a tool carrying this field. So on_hold is NOT
    # a structured server feature with machinery behind it; it is prompt text
    # the model happens to read, and any key added here rides along the same
    # way. Treat honouring it as emergent behaviour to be measured, and put
    # anything you actually depend on in AgentSpec.instructions instead.
    on_hold: str | None = None
    # Artificial delay in seconds, injected before the handler runs. This is the
    # primary independent variable for the evaluation: real tools take anywhere
    # from 50 ms (cache hit) to 10 s (cold backend query), and the interesting
    # question is where the model's conversational behaviour falls apart.
    latency: float = 0.0

    def schema(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.on_hold is not None:
            spec["on_hold_message"] = self.on_hold
        return spec

    async def invoke(self, arguments: dict[str, Any]) -> str:
        if self.latency:
            await asyncio.sleep(self.latency)

        result = self.handler(**arguments)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, str):
            payload = result
        else:
            payload = json.dumps(result, separators=(",", ":"))

        # The model card is explicit: system prompts and tool responses must be
        # ASCII-only. Non-ASCII here fails somewhere deep in the server rather
        # than at the callsite, so normalize loudly at the boundary instead.
        return payload.encode("ascii", errors="backslashreplace").decode("ascii")


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def tool(
        self,
        description: str,
        parameters: dict[str, Any],
        *,
        name: str | None = None,
        on_hold: str | None = None,
        latency: float = 0.0,
    ) -> Callable[[Handler], Handler]:
        """Decorator registering a function as a callable tool."""

        def decorate(fn: Handler) -> Handler:
            tool_name = name or fn.__name__
            self.tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                parameters=parameters,
                handler=fn,
                on_hold=on_hold,
                latency=latency,
            )
            return fn

        return decorate

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)


def obj(**properties: dict[str, Any]) -> dict[str, Any]:
    """Shorthand for a JSON-Schema object with all properties required.

    Required-by-default is deliberate: the model card says "if a required
    argument is missing, ask the user; never guess", and testing that claim
    needs arguments that are actually marked required.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}
