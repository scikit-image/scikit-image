# Cursor tooling

## validate-contribution.sh

Local pre-PR gate: **contribution heuristics** (tests paired with `src/` changes; `TODO.txt` when adding deprecation helpers), **pre-commit** on changed files, then **`spin test --test-modified`**.

```bash
./tools/cursor/validate-contribution.sh
./tools/cursor/validate-contribution.sh --allow-no-tests   # rare; document in PR
./tools/cursor/validate-contribution.sh --base-ref main --module tests/skimage2/filters/
```

Agent workflow: `.cursor/skills/pre-pr-gate/SKILL.md`. Human/agent routing: [AGENTS.md](../../AGENTS.md) § Verification.
