# Failures worth recording

> Part of the [K2-Horizon-32B](../README.md) investigation. See also [all model notes](../../README.md).

## The container image was broken, and the error pointed elsewhere

The obvious image choice, the `cu129` build that matches the variant the
neighbouring model uses, dies at import before reading any argument:

```
RuntimeError: operator torchvision::nms does not exist
```

This is an upstream packaging bug, reproduced on 2026-09-23 with both images
still on the node (`results/image-import-check.log`). The `cu129` image pairs
torch 2.14.0 with torchvision 0.28.0+cu129, and importing vLLM's API server
fails with exactly the error above. The plain `v0.30.0` image pairs torch
2.13.0+cu130 with torchvision 0.28.0+cu130 and imports cleanly. So the break is
a torch/torchvision version mismatch inside the image. (An earlier version of
this note described it as "torch 2.14.0+cu130 against cu129-built
torchvision", which does not match what the image reports.)

Nothing in the traceback names the image. It reads like a model or
configuration problem, and it fails the same way whatever you pass on the
command line, so changing flags, which is the natural first move, gets
nowhere. The plain tag is internally consistent and works.

When a container fails during import rather than during model load, suspect
the image before your configuration.

## Seventy-three gigabytes held by an invisible process

vLLM then failed with the error below. It was seen during the original
deployment and not saved, so this copy is not backed by a committed log; the
numbers in it are at least consistent (0.9 × 79.18 = 71.26).

```
ValueError: Free memory on device cuda:0 (5.95/79.18 GiB) on startup is less
than desired GPU memory utilization (0.9, 71.26 GiB).
```

The GPUs were held by a speech-model container running under a second
container runtime. The "stop everything" command only covered the other
runtime, so it succeeded and the GPUs stayed full.
`nvidia-smi --query-compute-apps=pid,used_memory,name` lists what is actually
on the GPU, and it was the only thing that showed a second runtime was
involved.

## The chat template silently ate every prompt

With the server healthy and `/v1/models` listing the model, every chat
completion rendered a 10-token prompt regardless of input. Responses came back
HTTP 200 with `finish_reason: "stop"`, and the replies were fluent and well
formed. There was no error or warning, and nothing in the log.

One of those responses is committed as `results/k2resp.json`
(`prompt_tokens: 10`). The model's reasoning invented a request that was
never sent, "the user wrote just 'Analyze'", and the reply answered it: *"Sure!
What would you like me to analyze? Please provide the text, data, image,
topic, or any specific details you'd like me to examine."* The empty prompt
produced a confident answer to a question the model made up.

`/v1/completions` worked throughout (`"The capital city of France is"` gave
`" Paris."`; seen at the time, not saved), so the weights and engine were
fine. Only the chat path was broken, which narrowed the problem to the chat
template.

The tokenizer confirmed it directly. A message containing a distinctive
canary string, rendered through the tokenize endpoint, gave the lists below.
Only the second is committed (`results/tok2.json`); the first was seen at the
time, and its count of 10 matches the `prompt_tokens` in `results/k2resp.json`.

```
before:  {"count":10, "tokens":[0,250018,2672,200,250019,250018,142036,200,250029,200]}
after :  {"count":16, "tokens":[0,250018,2672,200,72534,8541,163823,7169,331,8465,...]}
```

The broken render is ten tokens of scaffolding with no canary in it. The cause
is in the model's own `chat_template.jinja`, lines 932 to 937
(`results/chat-template-excerpts.txt`):

```jinja
{%- for message in messages -%}
    {%- if message.content is string -%}
        {%- set content = message.content -%}
    {%- else -%}
        {%- set content = '' -%}
    {%- endif -%}
```

Any message content that is not a plain string becomes an empty string.

Why the content was not a string turned out to be worse than this note first
said. The earlier version blamed vLLM's default for clients that send content
as a list of parts. On 2026-09-23 IFM's published recipe was run exactly as
written, with only enough GPUs added for it to start
(`probes/recipe_launch.sh`). vLLM's startup log said `Detected the chat
template content format to be 'openai'`, which means every message is
converted to a parts list before rendering, plain strings included. The result,
in `results/recipe-tp4-empty-prompt.log`:

- A canary sent as a plain string and as a parts list: both rendered to the
  same 10 tokens, and the canary was absent from both.
- Asked "What is the capital of Australia? One word.", the model reported 10
  prompt tokens, reasoned that "the user just sent a blank message", and
  replied "Hello! How can I assist you today?" with HTTP 200.

So a server started from IFM's own instructions answers an empty prompt on
every request, whatever the client sends. Adding
`--chat-template-content-format string` fixes it: the same four-GPU server
then renders the canary in both forms (`results/k2-bench-tp4.log`). The
mechanism is traced through the template in [how it works](03-how-it-works.md).

This was the most important of the three failures in the original deployment,
because it is the failure class the repository is built to be paranoid
about: a 200 response with fluent output, no crash or warning, and a
completely inert input path. A benchmark suite pointed at that server would have run to
completion and produced clean, publishable numbers that meant nothing. Every
quality score would have measured the model's prior instead of its response
to the prompt.

The same defence works for any model. Before measuring anything, put a
distinctive canary value in the input and confirm it appears in the rendered
prompt. Proving the round trip first costs one request.

## IFM's recipe does not start on two 80 GB GPUs

The recipe on the model card runs with two GPUs and sets no context limit, so
vLLM tries to reserve room for one request of the full 524,288 tokens. On two
H100s that needs 64.0 GiB of KV cache per GPU and only 36.25 GiB is free, so
the server refuses to start and suggests a limit of 296,928 tokens
(`results/recipe-as-published.log`). This failure is at least loud. With four
GPUs the same recipe starts, and then runs into the empty-prompt problem above.

## The reply after a tool call goes missing

In a tool-calling conversation, the turn after a tool result came back with an
empty `content` field on every low- and medium-effort request tried: 16 of 16
(`results/multiturn-tags.log`). Where the model had written a reply, the parser
filed it under `reasoning`; in one inspected case it wrote none. Nothing
reported an error. A client that reads `content` gets a blank reply to a
question it just paid a tool call for.
At high effort the answer arrived in `content` in 7 of 7. The cause is a
mismatch between the thinking tags the template uses for the conversation
history and for the new turn ([how it works](03-how-it-works.md)).

## Long, repetitive input turns the reply into word salad

Asked to find a code in a prompt of about 254,000 tokens or more, made of ten
words repeated at random, the model neither answered nor said it could not
find the code. It produced unrelated words ("Village of lot of window village
Hills economic Military Housing Area...") until it reached the 2,000-token
limit, with HTTP 200 and `finish_reason: "length"`. The same test on natural
prose found the code up to 380,000 tokens. Details are in
[results](04-results.md).

## Tool calls that come back as text

A tool call the parser cannot read is returned as plain text in `content`, with
`finish_reason: "stop"` and HTTP 200. That happened in 4 of 10 `json`-format
calls at low effort, where the model closed a correct JSON call with an XML
tag, and in 7 of 8 `xml_typed` calls at high effort. In two more cases the
model said it had no suitable tool when one was offered. Rates and advice are
in [using it](02-using-it.md) and [results](04-results.md).

In every one of these the server returned success and the output looked
reasonable. The only way to see the failure was to check the specific field or
token that should have been there.

---

Previous: [The reasoning trace](05-reasoning-trace.md) | [Contents](../README.md#contents) | Next: [Method and open questions](07-method.md)
