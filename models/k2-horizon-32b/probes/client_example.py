"""The usage guide's client example, run as-is so the guide can't drift from
what actually works. Uses the OpenAI Python SDK (tested with 3.19.0).

    PLAYGROUND_HOST=<host> uv run --no-project --with openai \
        python models/k2-horizon-32b/probes/client_example.py

Note the last line of the first block: `reasoning_content` is None. That is the
field an OpenAI-style client looks for, and reading it is how an earlier
version of this note concluded the trace was never returned.
"""
import os
from openai import OpenAI

client = OpenAI(base_url=f"http://{os.environ['PLAYGROUND_HOST']}:8080/v1", api_key="EMPTY")

r = client.chat.completions.create(
    model="k2-horizon-32b",
    messages=[{"role": "user", "content": "What is 17 multiplied by 23? Give the number only."}],
    temperature=1.0, top_p=0.95, max_tokens=4000,
    reasoning_effort="medium",
)
msg = r.choices[0].message
print("answer   :", msg.content.strip())
print("reasoning:", (getattr(msg, "reasoning", None) or "")[:80].replace("\n", " "), "...")
print("reasoning_content attr:", repr(getattr(msg, "reasoning_content", None)))

stream = client.chat.completions.create(
    model="k2-horizon-32b", stream=True, reasoning_effort="medium", max_tokens=4000,
    messages=[{"role": "user", "content": "Name the capital of Australia in one word."}],
)
thinking, answer = [], []
for chunk in stream:
    d = chunk.choices[0].delta if chunk.choices else None
    if d is None:
        continue
    thinking.append(getattr(d, "reasoning", None) or "")
    answer.append(d.content or "")
print("stream answer:", "".join(answer).strip(), "| reasoning chars:", len("".join(thinking)))
