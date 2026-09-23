# model-playground

A personal project where I pick AI models apart to understand them.

The loop: find a model worth a look, get it actually running on hardware I can
name, learn what it really is, stress it until it shows its edges, then write it
up. Each write-up is a practical guide to *using* the model plus a deeper
account of *what it is and why it behaves that way*.

This is not a product or a library, and it is not meant to be cloned and run.
There is no support, stability promise, roadmap or install path. It's public
because several of these things were annoying to find out and the write-ups
might save someone an afternoon. Nothing here is connected to any job or
company; it's curiosity.

The write-ups are what matter. The code is just how I got there.

## What's in here

| Path | |
|---|---|
| `models/<slug>/README.md` | Overview of the write-up: what the model is, what was found, and the contents |
| `models/<slug>/notes/` | The write-up itself, split into linked articles. This is the deliverable. |
| `models/<slug>/probes/` | The code behind every number in it, re-runnable |
| `models/<slug>/results/` | Raw probe output, committed as evidence |
| `logs/` | Scratch run logs; anything worth keeping moves to `results/` |

## License

MIT. Take anything useful. No weights or vendor code are vendored here; each
model carries its own terms, recorded in its note.
