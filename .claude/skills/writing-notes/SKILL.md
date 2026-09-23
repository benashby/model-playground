---
name: writing-notes
description: Use whenever writing or editing any human-facing markdown in model-playground (a model note under models/, its articles under models/<slug>/notes/, models/README.md, or the root README.md). Every such document must pass the humanizer before it is finished. Covers invoking the humanizer skill, the condensed rules to apply when it is not installed, what a rewrite must never change in a measurement note (code, numbers, tags, correction records), how long notes are split into navigable articles for the GitHub web UI, and the bundled check_rewrite.py integrity check.
---

# Writing notes

Every human-facing document in this repository goes through the humanizer
before it counts as done. That covers model notes, their articles, the rubric
in `models/README.md`, and the root `README.md`. It does not cover agent-facing
files (`CLAUDE.md`, `.claude/skills/*`), code, docstrings, or anything under
`results/`.

The reason is practical. These notes are read by people in the GitHub web UI,
and text full of AI habits (staged contrasts, one-line closers, dashes
everywhere, bold on every label) is slower to read and harder to trust. A
reader who spots those habits starts to doubt the numbers too, and the numbers
are the point of the repo.

## The procedure

1. Write the note, including its numbers, tables and probes.
2. Snapshot the files you are about to rewrite, keeping repo-relative paths:
   `mkdir -p $SNAP && cp --parents <files> $SNAP/`
3. Run the humanizer skill over them in file mode. It is installed as a plugin
   skill named `humanizer` (the namespace prefix varies by install). If it is
   not available, apply the condensed rules below yourself.
4. Run the integrity check and fix every hard failure:
   `python .claude/skills/writing-notes/check_rewrite.py $SNAP`
   Review each "numbers lost" warning by hand. A number may move to another
   sentence; it may not disappear.

For more than a few files, give each agent a disjoint set of files and the
hard rules below, and run the check once all of them have finished.

## What a rewrite must never change

A measurement note is prose wrapped around evidence. The humanizer edits the
prose and leaves the evidence alone:

- Fenced code blocks, byte for byte, including `#` comment lines inside them.
  (A heading regex that ignores fences will delete those comments. That
  happened once in this repo, and the output looked fine.)
- Inline code spans and link targets.
- Every number and unit, exactly as written. Never round, re-express, add or
  drop one.
- The `[MEASURED]` and `[CLAIM]` tags, and "Open:" labels. The rubric in
  `models/README.md` depends on them.
- Correction records. A note that says an earlier version got something wrong,
  and how, is keeping the most useful thing in the repo. The humanizer's rule
  against "writing about the previous version" does not apply to these.
- Anything stating the repo is personal and not tied to an employer.

## Condensed rules, for when the humanizer is not installed

Strongest first. Act on the first five at a single sighting.

1. No "not X but Y" contrasts, in any form ("isn't X, it's Y", "X rather than
   Y", split across two sentences, or a clipped ", no X" tail), unless the
   negative half corrects something a reader actually believes.
2. No one-line closing paragraphs that restate the paragraph before, and no
   rows of dramatic fragments.
3. No aphorisms or "the real question is" framing. State the specific claim.
4. No staged run-ups ("Here's the thing", "Let's look at").
5. No answering objections nobody raised.
6. No em or en dashes in prose. Use a comma, colon, period or parentheses.
   Ranges become "0.467 to 0.492" or "0.467-0.492".
7. No triads for rhythm. Three items only when there are three things.
8. No inflation ("pivotal", "crucial", "a testament to"), no sales language,
   and "is"/"has" instead of "serves as"/"boasts".
9. Bold only where a reader needs it; no bold label on every list item.
   Sentence-case headings, no decorative arrows or emoji, no horizontal rule
   between every section.
10. No chat residue ("I hope this helps", "let me know").

Technical notes stay neutral and plain. Keep the author's dry asides and
first-person choices; they are part of the voice.

## Splitting long notes

A note much over ~400 lines is too long to read in a browser. Split it into
articles under `models/<slug>/notes/NN-topic.md`, and make
`models/<slug>/README.md` the hub: what the model is, what problem it solves,
the main findings with links, and a contents table. GitHub renders that README
under the directory listing, so it is what a visitor sees first.

Each article opens with its title and a one-line breadcrumb back to the hub,
and ends with a `Previous | Contents | Next` line. Cross-references link to the
article, not to a section number, because section numbers stop meaning
anything once the monolith is gone. `models/nemotron-voicechat-11b/` is the
worked example.
