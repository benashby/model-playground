"""Interruption policy: what happens to an in-flight tool when the user barges in.

THIS IS THE DESIGN DECISION AT THE CENTRE OF THE EVALUATION.

Full-duplex means the user can start speaking while a tool call is still
executing. The model keeps the conversation alive during that window; the
client has to decide what the pending tool call is now worth. There is no
correct default -- which is exactly why it is worth studying. The three
policies below are the experiment: each one makes the model behave differently
under interruption, and the point of the harness is to watch how.

The three defensible policies:

  CANCEL
      Abandon the in-flight tool, never send a function_call_output.
      + Lowest latency, no stale context reaches the model.
      - You burned the lookup. And if the user's interruption was "yes, that
        one" -- an affirmation, not a topic change -- you discarded the exact
        result they were confirming.

  COMPLETE_AND_DELIVER
      Let it finish, send the result whenever it lands.
      + Never wastes work; simplest to reason about.
      - The result arrives mid-new-turn. The model may splice "your balance is
        four hundred dollars" into a conversation that moved on two turns ago.

  COMPLETE_AND_HOLD
      Let it finish, cache the result, deliver only if still relevant.
      + Most faithful to what a human agent does.
      - Needs a pending-call table keyed by call_id and a relevance test, and
        "still relevant" is itself a judgement call the model cannot make for
        you.

`decide()` is called from the receive loop the moment
`input_audio_buffer.speech_started` arrives while at least one tool call is
still pending. It must not block -- return a decision, do not await work.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class Decision(enum.Enum):
    CANCEL = "cancel"
    COMPLETE_AND_DELIVER = "complete_and_deliver"
    COMPLETE_AND_HOLD = "complete_and_hold"


@dataclass(frozen=True)
class PendingCall:
    """A tool call the client started but has not yet answered."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    started_at: float          # time.monotonic() when dispatch began
    elapsed: float             # seconds in flight at the moment of interruption
    expected_latency: float    # the tool's configured latency, if any


# Tools whose effects survive the call. Cancelling one of these does not undo
# it -- it just throws away our knowledge that it happened, which is strictly
# worse than delivering a late result. Kept as a name set rather than a Tool
# field only because decide() receives a PendingCall, not the Tool; promote it
# to a `side_effecting` flag on Tool if this list outgrows a glance.
SIDE_EFFECTING = {"transfer_to_queue", "open_ticket"}

# Interruptions that are agreement, not redirection. A caller saying "mm hm"
# over the agent is not asking it to stop working -- treating backchannel as a
# topic change is the single most common way duplex clients feel broken.
BACKCHANNEL = {
    "yeah", "yea", "yep", "yes", "uh huh", "mm hm", "mhm", "right",
    "ok", "okay", "sure", "got it", "i see", "exactly",
}


def decide(pending: PendingCall, *, user_transcript_so_far: str) -> Decision:
    """Choose what to do with `pending` now that the user has started speaking.

    `user_transcript_so_far` is whatever ASR has emitted for the interrupting
    utterance at this instant -- usually only a word or two, often an empty
    string, since speech_started fires on the first token. Treat it as a weak
    hint, not a reliable classification input.

    TODO(ben): implement the policy.

    Things worth weighing:
      - Time already invested. A call 90% done is cheap to finish and expensive
        to redo; one that just started is nearly free to drop.
      - Whether the interruption looks like an affirmation ("yeah", "uh huh",
        "right") versus a redirection ("no wait", "actually", "cancel").
        Backchannel noise is NOT a topic change and probably should not kill a
        query -- this is where full-duplex models most often get it wrong.
      - Whether the tool is idempotent. Re-running a lookup is free; re-running
        a transfer or a charge is not, so CANCEL on a side-effecting tool may be
        the dangerous option rather than the safe one.
      - You may return different decisions for different tools; `pending.name`
        is available.
    """
    # A side effect already in motion cannot be un-fired. Cancelling would only
    # discard the receipt, leaving the caller transferred while the agent
    # believes it never happened -- the worst of both outcomes.
    if pending.name in SIDE_EFFECTING:
        return Decision.COMPLETE_AND_DELIVER

    # Agreement is not redirection. If the caller is backchannelling, they are
    # still on the same topic and still want the answer.
    words = user_transcript_so_far.strip().lower().strip(".,!?")
    if words and words in BACKCHANNEL:
        return Decision.COMPLETE_AND_DELIVER

    # Nearly finished: pay the remaining milliseconds, but do not barge the
    # result into a turn that has moved on. Hold it and let relevance decide.
    if pending.expected_latency and pending.elapsed >= 0.7 * pending.expected_latency:
        return Decision.COMPLETE_AND_HOLD

    # Idempotent, barely started, and the caller wants something else: drop it.
    return Decision.CANCEL
