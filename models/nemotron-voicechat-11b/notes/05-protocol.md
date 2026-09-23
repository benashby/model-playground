# The protocol

> Part of the [NemotronLabs VoiceChat 11B](../README.md) investigation. See also [all model notes](../../README.md).


Endpoint: `ws://<host>:9000/v1/realtime`

Everything here was checked against the running server. Where the docs and the
server disagree, the server wins and the disagreement is noted.

## It is the OpenAI Realtime API

The event names come straight from OpenAI's Realtime API, so anything that
already speaks that dialect is close to drop-in and the mental model transfers.

## Wire format

| Property | Value |
|---|---|
| Sample rate | 24 000 Hz both directions |
| Encoding | 16-bit signed PCM, little-endian, mono |
| Chunk size | ~80 ms = 1920 samples = 3840 bytes |
| Transport | base64 inside JSON |

The model natively takes 16 kHz in and produces 22.05 kHz out, and the server
resamples internally. Do not "helpfully" send 16 kHz; the wire is 24 kHz.

> The model card's offline instructions say 16 kHz mono. That applies to
> `offline_voicechat_infer.py` and not to the WebSocket path, which is easy to
> miss.

## Client to server

```jsonc
// Configure. Send immediately after connect.
{"type":"session.update","event_id":"<uuid>","session":{
  "audio":{"input":{"format":{"type":"audio/pcm","rate":24000}},
           "output":{"format":{"type":"audio/pcm","rate":24000}}},
  "instructions":"<system prompt or null>",
  "tools":[ /* function definitions */ ]}}

// Audio, ~80ms per event
{"type":"input_audio_buffer.append","event_id":"<uuid>","audio":"<base64>"}

// Return a tool result
{"type":"conversation.item.create","item":{
  "type":"function_call_output","call_id":"<call-id>","output":"<string>"}}

{"type":"session.close","event_id":"<uuid>"}
```

- `instructions` is applied only to the first inference call. It is not a
  persistent system prompt re-injected each turn.
- `tools` must be set before any turn that could trigger one, or the server
  answers `tools_not_set`.
- `output` must be a string, so serialize dicts yourself.
- System prompts and tool responses must be ASCII-only.

## Server to client

| Event | Meaning |
|---|---|
| `session.created` / `session.updated` / `session.end` | lifecycle |
| `input_audio_buffer.speech_started` | first ASR token; this is the barge-in signal |
| `input_audio_buffer.speech_stopped` | end of speech |
| `conversation.item.input_audio_transcription.delta` / `.completed` | caller transcript |
| `response.output_audio.delta` / `.done` | synthesized audio (base64) |
| `response.output_audio_transcript.delta` / `.done` | agent transcript |
| `response.function_call_arguments.done` | tool invocation, carrying `{call_id, name, arguments}` |
| `response.created` / `response.done` | turn boundaries; `response.done` carries usage |

Error codes: `inference_timeout`, `inference_error`, `session_timeout`,
`tools_not_set`.

## Tool execution is client-side

The server never runs a tool. It emits `response.function_call_arguments.done`
and keeps talking; you execute the tool and return `conversation.item.create`.

So how long the tool gap lasts, and what happens to an in-flight call when the
user interrupts, are properties of your client. For evaluation that is good
news, because you control the independent variable. It is bad news if you
assumed the model owned that behaviour.

## Measured findings the docs do not state

### `on_hold_message` is not a server feature

The model card advertises per-tool "on-hold" messages, spoken filler while a
tool runs. The API reference documents no field for it.

Measured: the server serializes the entire `tools` array into a JSON string
and injects it verbatim into the system prompt's `<AVAILABLE_TOOLS>` block.
This was confirmed by sending a tool carrying `on_hold_message` and grepping the
container's own `Sending system prompt` log line:

```
on_hold_message\": \"One moment while I check that balance.\"}]</AVAILABLE_TOOLS>
```

Nothing on the server fires that message at the right moment. It is prompt
text the model may or may not act on, and any key added to a tool dict reaches
the model the same way. Anything that matters belongs in `instructions`.

### The server's own schema dialect differs from the docs

The container's warmup prompt uses:

```json
"parameters": {"type": "dict", "properties": {...}}
```

The model card's example uses `"type": "object"`. Since all of it is
serialized into prompt text, both appear to work, but NVIDIA's internal
convention differs from its published documentation. This is the first thing
to flip if tool calling misbehaves.

### `tools` echoes back as a string

`session.updated` returns `tools` as a JSON string rather than a structured
list, which fits with it being destined for prompt interpolation instead of
being parsed into a schema object.

### Tool calls precede transcript finalization

```
11.24s   tool.call    generate_random_number({"min":1,"max":50})
12.36s   caller.said  "can you generate a random number between one and fifty?"
```

The gap is 1.1 s: the model commits to a tool call before ASR finalizes the
utterance. If tool handling waits on `...transcription.completed`, you throw
away over a second of the latency advantage.

### `speech_started` fires with almost no transcript

It fires on the first ASR token, so `...transcription.delta` has usually
produced a word or two at most, and often nothing. Barge-in logic keyed on
what the user said can only treat that text as a weak hint.

---

Previous: [Setting it up](04-deployment.md) | [Contents](../README.md#contents) | Next: [The harness and its fixtures](06-harness.md)
