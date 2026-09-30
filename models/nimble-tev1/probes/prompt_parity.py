"""Check that systemone.py builds byte-identical user messages to its two sources.

    OLLAMA_SRC=<ollama/ollama checkout at 1abe35e> NIMBLE_SRC=<bespokelabsai/nimble at 62076b4> \
        python models/nimble-tev1/probes/prompt_parity.py > models/nimble-tev1/results/prompt-parity.log

Ollama side: runs Ollama's own decision.Compile (Go, needs a Go toolchain) on
fixtures/prompt-edge-cases.json through parity/main.go.

Nimble side: runs Bespoke's prepare_prompts with a stand-in tokenizer. It
encodes one token per character and renders a template with sentinel markers,
so the exact system and user strings can be read back out. prepare_prompts
needs no ML libraries.

Then prints, per fixture and question, whether the messages match, and where
the Ollama and Nimble formats differ from each other (the point of the
"nimble" format is that they do).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import systemone  # noqa: E402

FIXTURES = HERE / "fixtures" / "prompt-edge-cases.json"
cases = json.loads(FIXTURES.read_text())


def ollama_messages():
    src = Path(os.environ["OLLAMA_SRC"])
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory(dir=src) as tmp:
        shutil.copy(HERE / "parity" / "main.go", Path(tmp) / "main.go")
        out = subprocess.run(["go", "run", "./" + Path(tmp).name, str(FIXTURES)], cwd=src,
                             capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr)
    return head, [json.loads(line) for line in out.stdout.splitlines()]


class CharTokenizer:
    all_special_ids = []

    def apply_chat_template(self, messages, **_):
        return "".join(f"\x00{m['role']}\x00{m['content']}" for m in messages) + "\x00end"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def nimble_messages():
    src = Path(os.environ["NIMBLE_SRC"])
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    sys.path.insert(0, str(src))
    from nimble.evaluation.evaluate_pilot import adapt_input
    from nimble.scoring.parallel_schema import SYSTEM_PROMPT, prepare_prompts
    out = []
    for case in cases:
        try:
            context, schema = adapt_input({"state": case["state"], "questions": case["questions"]})
            prepared = prepare_prompts(CharTokenizer(), context, schema, 10**6)
        except (ValueError, KeyError, TypeError) as exc:
            out.append(f"refused: {exc}")
            continue
        users = []
        for ids in prepared.full_ids:
            parts = "".join(map(chr, ids)).split("\x00")
            users.append(parts[parts.index("user") + 1])
            assert parts[parts.index("system") + 1] == SYSTEM_PROMPT
        out.append(users)
    return head, out, SYSTEM_PROMPT


def first_difference(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at char {i}: {a[max(0, i - 25):i + 25]!r} vs {b[max(0, i - 25):i + 25]!r}"


o_head, o_all = ollama_messages()
n_head, n_all, n_system = nimble_messages()
print(f"ollama/ollama {o_head}")
print(f"bespokelabsai/nimble {n_head}")
print(f"nimble SYSTEM_PROMPT == systemone['nimble']: {n_system == systemone.SYSTEM_PROMPTS['nimble']}")
print()
bad = 0
for case, o_users, n_users in zip(cases, o_all, n_all):
    print(f"== {case['name']}")
    ours_o, _ = systemone.user_messages("ollama", case["state"], case["questions"])
    same = ours_o == o_users
    bad += not same
    print(f"  systemone ollama format == Ollama decision.Compile: {same}")
    if not same:
        for a, b in zip(ours_o, o_users):
            if a != b:
                print("   ", first_difference(a, b))
    if isinstance(n_users, str):
        print(f"  Nimble upstream: {n_users}")
        continue
    ours_n, _ = systemone.user_messages("nimble", case["state"], case["questions"])
    same = ours_n == n_users
    bad += not same
    print(f"  systemone nimble format == Nimble prepare_prompts: {same}")
    if not same:
        for a, b in zip(ours_n, n_users):
            if a != b:
                print("   ", first_difference(a, b))
    for name, a, b in zip(case["questions"], o_users, n_users):
        if a != b:
            print(f"  Ollama vs Nimble upstream, {name!r}: {len(a)} vs {len(b)} chars, first "
                  + first_difference(a, b))
print()
print("all systemone formats match their sources" if not bad else f"{bad} MISMATCHES")
sys.exit(bool(bad))
