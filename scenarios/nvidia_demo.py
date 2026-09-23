"""NVIDIA's own demo tools, lifted verbatim from the container's warmup prompt.

tool_call.wav is NVIDIA's demo recording, so it almost certainly exercises this
tool set rather than a support-desk one. Schemas here match what the server
sends itself during warmup -- including their "type": "dict" spelling, which
differs from the "type": "object" in the published model card.

Latencies are injected to make the duplex behaviour observable; NVIDIA's own
demo presumably answers instantly.
"""

from __future__ import annotations

from playground import AgentSpec, Registry

registry = Registry()


@registry.tool(
    description="Convert a certain amount of currency from one currency to another",
    parameters={
        "type": "dict",
        "properties": {
            "amount": {"type": "number", "description": "The amount of currency to convert"},
            "from_currency": {"type": "string", "description": "The currency to convert from"},
            "to_currency": {"type": "string", "description": "The currency to convert to"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    on_hold="Let me convert that for you.",
    latency=1.5,
)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, object]:
    rate = 0.92 if to_currency.upper() in {"EUR", "EUROS"} else 1.27
    return {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted": round(amount * rate, 2),
        "rate": rate,
    }


@registry.tool(
    description="Get the latest news headlines",
    parameters={
        "type": "dict",
        "properties": {
            "country": {"type": "string", "description": "The country for which news headlines should be retrieved"},
            "category": {"type": "string", "description": "The category of news, e.g. business, sports, entertainment"},
        },
        "required": [],
    },
    on_hold="Pulling up the headlines now.",
    latency=3.0,
)
def get_news_headlines(country: str = "us", category: str = "general") -> dict[str, object]:
    return {
        "country": country,
        "category": category,
        "headlines": [
            "Markets close higher on tech gains",
            "Storm system moves across the midwest",
            "City council approves transit expansion",
        ],
    }


@registry.tool(
    description="Calculate the Body Mass Index (BMI)",
    parameters={
        "type": "dict",
        "properties": {
            "weight": {"type": "number", "description": "The weight in kilograms"},
            "height": {"type": "number", "description": "The height in meters"},
        },
        "required": ["weight", "height"],
    },
    on_hold="Calculating that now.",
    latency=0.2,
)
def calculate_bmi(weight: float, height: float) -> dict[str, object]:
    return {"bmi": round(weight / (height**2), 1), "weight": weight, "height": height}


@registry.tool(
    description="Generate a random number within specified range",
    parameters={
        "type": "dict",
        "properties": {
            "min": {"type": "number", "description": "The minimum value of the range"},
            "max": {"type": "number", "description": "The maximum value of the range"},
        },
        "required": ["min", "max"],
    },
    on_hold="Picking a number.",
    latency=0.1,
)
def generate_random_number(min: float, max: float) -> dict[str, object]:
    # Fixed, not random: a reproducible harness must not introduce variance the
    # run-to-run comparison cannot account for.
    return {"number": int((min + max) // 2), "min": min, "max": max}


SPEC = AgentSpec(
    name="nvidia_demo",
    registry=registry,
    instructions=(
        "You are a helpful voice assistant. Keep replies short and natural. "
        "Use only values the user actually spoke as tool arguments. "
        "If a required argument is missing, ask for it rather than guessing."
    ),
)
