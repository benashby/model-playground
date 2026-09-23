---
name: realtime-protocol
description: VoiceChat-specific — one model, one container, one wire format; nothing here generalises to the other models in models/. Use when touching src/playground/protocol.py, debugging WebSocket events, wiring tool calls, or answering any question about the wire format that NVIDIA's NIM VoiceChat container speaks — the OpenAI Realtime dialect at /v1/realtime. Covers the full client-to-server and server-to-client event vocabulary, the 24kHz/80ms/3840-byte wire constants, client-side tool execution semantics, and the measured findings where the server contradicts its own documentation (on_hold_message is prompt text not a feature, "type":"dict" vs "object", tools echoed as a string, tool calls firing before transcript finalization).
---

# The realtime protocol

**Scope: NemotronLabs VoiceChat 11B as served by NVIDIA's NIM container.** This
is one model's wire format, not a general protocol reference. No other model
written up in `models/` speaks it.

Endpoint: `ws://<host>:9000/v1/realtime`

Everything here was verified against a running server. Where the docs and the
server disagree, **the server wins** and it is flagged.

Upstream reference: `NVIDIA-NeMo/Speech` @ `nemotron-labs-voicechat`,
`voicechat_realtime_instructions/api-reference.md`.

## It is the OpenAI Realtime API

Event names are lifted directly from OpenAI's Realtime API. Anything already
speaking that dialect is close to drop-in, and the mental model transfers.

## Wire constants

| Property | Value |
|---|---|
| Sample rate | **24 000 Hz both directions** |
| Encoding | 16-bit signed PCM, little-endian, mono |
| Chunk | ~80 ms = 1920 samples = **3840 bytes** |
| Transport | base64 inside JSON |

The model natively wants 16 kHz in and emits 22.05 kHz out; **the server
resamples internally**. Do not send 16 kHz.

> The model card's 16 kHz mono instruction applies to
> `offline_voicechat_infer.py`, **not** this path. Easy to conflate, and the
> failure is silent-ish rather than an error.

## Client → server

```jsonc
// Configure. Send immediately after connect.
{"type":"session.update","event_id":"<uuid>","session":{
  "audio":{"input":{"format":{"type":"audio/pcm","rate":24000}},
           "output":{"format":{"type":"audio/pcm","rate":24000}}},
  "instructions":"<system prompt or null>",
  "tools":[ /* function definitions */ ]}}

{"type":"input_audio_buffer.append","event_id":"<uuid>","audio":"<base64>"}

{"type":"conversation.item.create","item":{
  "type":"function_call_output","call_id":"<call-id>","output":"<string>"}}

{"type":"session.close","event_id":"<uuid>"}
```

Constraints that bite:

- **`instructions` applies only to the first inference call.** It is not a
  persistent system prompt re-injected each turn. Anything that must hold for
  the whole conversation has to survive on its own.
- **`tools` must be set before any turn that could trigger one**, else
  `tools_not_set`.
- **`output` must be a string.** Serialize dicts yourself.
- **System prompts and tool responses must be ASCII-only.** Non-ASCII fails deep
  in the server, not at your callsite — clamp at the boundary.

## Server → client

| Event | Meaning |
|---|---|
| `session.created` / `.updated` / `session.end` | lifecycle |
| `input_audio_buffer.speech_started` | first ASR token — **the barge-in signal** |
| `input_audio_buffer.speech_stopped` | end of speech |
| `conversation.item.input_audio_transcription.delta` / `.completed` | caller transcript |
| `response.output_audio.delta` / `.done` | synthesized audio, base64 |
| `response.output_audio_transcript.delta` / `.done` | agent transcript |
| `response.function_call_arguments.done` | **tool call** → `{call_id, name, arguments}` |
| `response.created` / `response.done` | turn boundaries; `.done` carries usage |

Errors: `inference_timeout`, `inference_error`, `session_timeout`,
`tools_not_set`.

## Tool execution is client-side

The server never runs a tool. It emits `response.function_call_arguments.done`
and **keeps talking**; you execute and reply with `conversation.item.create`.

Consequence worth internalising: *how long the tool gap is* and *what happens to
an in-flight call when the user interrupts* are properties of **your client**,
not the model. Good for evaluation — you control the independent variable. Bad
if you attribute your own client's latency to the model.

## Measured findings the docs do not state

### `on_hold_message` is not a server feature

The model card advertises per-tool "on-hold" spoken filler. The API reference
documents no field for it.

**Measured**: the server serializes the entire `tools` array into a JSON string
and injects it verbatim into the system prompt's `<AVAILABLE_TOOLS>` block.
Confirmed by sending a tool carrying `on_hold_message` and grepping the
container's own log:

```
sudo docker logs voicechat 2>&1 | grep "Sending system prompt" | tail -1
# ...on_hold_message\": \"One moment while I check that balance.\"}]</AVAILABLE_TOOLS>
```

So there is **no machinery** firing that message at the right moment. It is
prompt text the model may or may not act on, and *any* key added to a tool dict
reaches the model the same way. Put anything load-bearing in `instructions`.

### The server's schema dialect differs from its docs

The container's warmup prompt uses `"parameters": {"type": "dict", ...}`; the
model card uses `"type": "object"`. Both appear to work since it is all prompt
text — but if tool-call accuracy looks marginal, flip this first.

### `tools` echoes back as a string

`session.updated` returns `tools` as a JSON **string**, not a list — consistent
with it being destined for prompt interpolation rather than schema parsing.

### Tool calls precede transcript finalization

```
11.24s  tool.call    generate_random_number({"min":1,"max":50})
12.36s  caller.said  "can you generate a random number between one and fifty?"
```

**1.1 s ahead.** The model commits before ASR finalizes — architecturally
consistent with the dedicated `function_head` parallel to the LM head. Gating
tool handling on `...transcription.completed` throws away over a second of the
latency advantage.

### `speech_started` fires with almost no transcript

It fires on the **first ASR token**, so accumulated deltas are usually a word or
two, often nothing. Barge-in logic keyed on *what* was said must treat that text
as a weak hint, never a classification.

## The prompt the server builds

From its warmup log — useful when reasoning about why the model did something:

```
You can use the following tools to assist the user if required:
<AVAILABLE_TOOLS>[ ...tool JSON... ]</AVAILABLE_TOOLS>

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>
```

## Debugging

The container's own log is the highest-value source and shows the *actual*
prompt, which no client-side logging can:

```bash
sudo docker logs voicechat 2>&1 | grep "Sending system prompt" | tail -1
sudo docker logs voicechat 2>&1 | tail -40
```

Client-side, every unrecognised event is recorded as `kind:"unhandled"` with its
type — check that before assuming an event does not exist:

```bash
jq -r 'select(.kind=="unhandled") | .type' logs/session.jsonl | sort | uniq -c
```
