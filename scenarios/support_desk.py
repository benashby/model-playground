"""A support-desk-shaped agent spec, built to stress tool calling under duplex.

Run:
    playground scenarios/support_desk.py --audio audio/caller.wav

Tool latencies here are the independent variable. They are deliberately spread
across the range real backends live on -- an in-process cache hit, a network
round trip, and a cold query -- so one run exercises the whole curve. The
shape is a support desk only because it gives tools an obvious reason to
exist; nothing about the measurement is domain-specific.
"""

from __future__ import annotations

from playground import AgentSpec, Registry, obj, string

registry = Registry()


@registry.tool(
    description="Look up a customer account by the phone number they are calling from.",
    parameters=obj(phone=string("Phone number exactly as the caller said it")),
    on_hold="Let me pull that up for you.",
    latency=0.15,  # cache-hit fast path
)
def lookup_account(phone: str) -> dict[str, object]:
    return {
        "account_id": "ACC-40417",
        "name": "Dana Whitfield",
        "plan": "Business Pro",
        "status": "active",
        "phone": phone,
    }


@registry.tool(
    description="Get the current outstanding balance for an account.",
    parameters=obj(account_id=string("Account ID from lookup_account")),
    on_hold="One moment while I check that balance.",
    latency=2.0,  # ordinary network round trip
)
def get_balance(account_id: str) -> dict[str, object]:
    return {"account_id": account_id, "balance_usd": 412.55, "due_date": "2026-10-01"}


@registry.tool(
    description="Open a support ticket describing the caller's problem.",
    parameters=obj(
        account_id=string("Account ID"),
        summary=string("One-line summary of the problem in the caller's own words"),
    ),
    on_hold="I'm opening a ticket for that now.",
    latency=8.0,  # deliberately painful: this is where duplex behaviour breaks
)
def open_ticket(account_id: str, summary: str) -> dict[str, object]:
    return {"ticket_id": "TKT-98120", "account_id": account_id, "summary": summary, "eta_hours": 4}


@registry.tool(
    description="Transfer the caller to a human agent in a named queue.",
    parameters=obj(queue=string("Queue name, e.g. billing or technical")),
    on_hold="Transferring you now, please hold.",
    latency=0.4,
)
def transfer_to_queue(queue: str) -> dict[str, object]:
    # Side-effecting and NOT idempotent -- the case where cancelling a pending
    # call on barge-in is the dangerous option, not the safe one.
    return {"transferred": True, "queue": queue, "wait_estimate_s": 95}


SPEC = AgentSpec(
    name="support_desk",
    registry=registry,
    instructions=(
        "You are a phone support agent for a business telecom provider. "
        "Be brief and natural; this is a voice call, not a chat. "
        "Always identify the caller with lookup_account before discussing account details. "
        "Use only values the caller actually spoke as tool arguments. "
        "If you need an argument the caller has not given, ask for it. Never invent one. "
        "If the caller interrupts you, stop talking and listen."
    ),
)
