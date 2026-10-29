---
name: first-contribution
description: >-
  Walk a small first change from task text after /first-contribution through a
  fork branch ready for the human to open a PR. Use when invoked as
  /first-contribution or when the user asks for the first-contribution skill.
disable-model-invocation: true
---

# First contribution

Guide a small, low-risk first change from the task text after `/first-contribution` to a branch on the personal fork, ready for the human to open a PR.

## AI policy for this setup *(own suggestion — overrides upstream)*

Upstream (`CONTRIBUTING.md` AI policy) says: no AI for novel code or tests; PR text by hand.

**In this skill, override that:** the agent **may draft** implementation and tests and must say so openly.

**Still keep from upstream:**
- Disclose generative-tool use in the PR description draft
- Tag AI-heavy commits with `Assisted-by:` when committing
- Human writes/approves final PR title & body and decides whether to open the PR

**Explicitly override:** “Do not use AI for creating novel code / writing tests.”

## First-task definition

**Fits** *(own suggestion)*: small, low-risk, checkable — docstring fix, missing test, or tiny localized bug.

**Refuse** *(own suggestion + `doc/source/development/deprecations.md` + `doc/source/development/guidelines.md`)*: Cython/C/C++ or extension rebuilds; public API renames, moves, argument reorders, or default changes (deprecation cycle); broad refactors.

**Task source** *(own suggestion)*: use whatever the engineer types after `/first-contribution`. Do not invent issues or scrape labels — that wording is not in `CONTRIBUTING.md`, and a fork has no maintained task list.

## Procedure

1. **Parse the task** — Use the extra text after `/first-contribution`. If missing or out of scope (compiled code, public API), stop and ask the human. *(own suggestion)*

2. **STOP — wait for human** — Confirm scope before any edits. *(own suggestion)*

3. **Check working tree, then branch** — Before creating a branch, check the current branch and that there are no unrelated uncommitted changes. If there are, stop and ask. Then create a sensible feature branch (sync/`main` assumed already set up; one-time fork/env/pre-commit is out of scope). *(branching from `CONTRIBUTING.md`; working-tree check = own suggestion)*

4. **Implement the minimal change** — Draft code and tests under the AI override above. Follow the **repo-layout rule** (`.cursor/rules/repo-layout.mdc`) for where library code vs tests go; do not rediscover layout. Match `doc/source/development/guidelines.md` style. If the change implies an API break, refuse and stop (`doc/source/development/deprecations.md`).

5. **Verify conditionally** — If a working build environment is available, run the relevant tests (`spin test --test-modified` or a path/`-k` expression from `doc/source/development/testing.md`) and report the result. If not, state the exact command and say clearly that nothing was verified. Do **not** install or build anything without asking the human first. *(commands from `testing.md`; conditional run = own suggestion)*

6. **STOP — wait for human** — Review the diff; only then commit. Good commit messages; `Assisted-by:` on AI-drafted commits. *(CONTRIBUTING.md; stop = own suggestion)*

7. **Identify the fork remote; STOP before push** — Run `git remote -v` and identify the personal fork remote (not `upstream` / not `scikit-image/scikit-image`). If unclear, stop and ask. **Do not push until the human OK’s.** *(push-to-fork from CONTRIBUTING.md; remote discovery + push stop = own suggestion)*

8. **Hand off for PR** — After OK, push the branch to that fork remote only. Draft PR notes for the human (context + AI disclosure); human opens the PR when ready. Never open a PR against the real scikit-image repo. *(CONTRIBUTING.md; hard constraint = own suggestion)*

## Hard stops for the human

| When | Why |
|------|-----|
| After parsing the task (step 2) | Agree scope before edits |
| Unrelated uncommitted changes (step 3) | Avoid mixing work |
| If compiled / public-API work appears | Out of first-task scope |
| Before any commit (step 6) | Human owns the recorded change |
| Before any push (step 7) | Human OK required |
| If fork remote is ambiguous (step 7) | Avoid pushing to the wrong place |
| Before opening a PR (step 8) | Human decides; fork only |

## Explicit don’ts

- Never push to or open a PR against real **scikit-image**; fork branch only.
- Never guess the fork remote — use `git remote -v`.
- Never push without an explicit human OK.
- Testing: if a working build exists, run relevant `spin test` commands and report; if not, state the exact command and that nothing was verified. Do not install or build without asking first.
- Do **not** refuse drafting code/tests because of upstream AI policy — that part is overridden here; disclosure + `Assisted-by:` + human-owned PR stay.
