# Evidence Directory

This directory stores exported evidence snapshots used in summaries and writeups.

## Conventions

- Each run bundle is under `docs/evidence/<run_family>_<run_id>/`.
- JSON/MD files here are copied from remote run outputs and intended to be versioned.
- Some markdown summaries include repo-relative pointers to full run outputs under `results/experiments/...`.

## Important Note

Not every referenced `results/experiments/...` artifact is checked into git. Some were generated on remote compute and only summarized/captured here.

When a referenced result file is missing locally, treat the corresponding exported file in this folder as the canonical evidence snapshot for that run.
