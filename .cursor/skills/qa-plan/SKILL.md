---
name: qa-plan
description: >-
  Write a read-only QA test plan for one pull request when the user types /qa-plan.
disable-model-invocation: true
---

# QA plan

`/qa-plan <pull request number>` is for QA. Read-only.

## Procedure

1. **Fork** — Run `git remote get-url origin`. `<fork>` is `owner/repo` from that URL (drop a trailing `.git`; keep the last two path parts). Never use `upstream`.

2. **Read the pull request** — Run both:
   - `gh pr view <n> --repo <fork> --json title,body,files`
   - `gh pr diff <n> --repo <fork>`

3. **Notion** — If the pull request links a Notion task, ask the human before reading it. Use its Acceptance text only if the human says so.

4. **Write the test plan** with these parts:
   - What changed, in two plain sentences.
   - The cases to check: normal, edge, and regression.
   - Which existing tests under `tests/` already cover this area. Search the repo. If none, say "none found".
   - What is NOT covered.
   - How to run the relevant tests. Use a command from `doc/source/development/testing.md` (`spin test -- <path>`, `spin test -- -k <expression>`, or `spin test --test-modified`). State that nothing was run.

## Explicit don'ts

- Never edit files.
- Never comment on the pull request.
- Never write to Notion unless the human names a specific page.
