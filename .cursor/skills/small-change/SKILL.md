---
name: small-change
description: >-
  Guide one small, low-risk change from a typed task to a pushed fork branch,
  not only a first contribution. Use when invoked as /small-change or when
  the user asks for the small-change skill.
disable-model-invocation: true
---

# Small change

Guide one small, low-risk change from a typed task to a pushed fork branch, not only a first contribution.

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

**Over the limit** *(own suggestion)*: if the change would touch more than 3 files or roughly 100 changed lines, or needs a design decision, stop at step 1 and tell the human to use Plan mode or a cloud agent with a reviewer instead.

**Task source** *(own suggestion)*: use whatever the engineer types after `/small-change`. The task may be a Notion row the human names. In that case read it with the Notion tool, use its Acceptance text as the definition of done, never invent acceptance criteria, and after the human approves the push update only that row (Status "In review", the PR link, a short summary of the test cases). Work only in the workspace the human names, and never create or delete pages. Do not invent issues or scrape labels — that wording is not in `CONTRIBUTING.md`, and a fork has no maintained task list.

## Procedure

1. **Parse the task** — Use the extra text after `/small-change`. If missing or out of scope (compiled code, public API), stop and ask the human. *(own suggestion)*

2. **STOP — wait for human** — Confirm scope before any edits. *(own suggestion)*

3. **Fetch origin, branch from `origin/main`, then check the tree** — Run `git fetch origin`. Create a sensible feature branch from `origin/main` with `--no-track` (`git switch -c <branch> --no-track origin/main`). Then check the working tree is clean. If there are unrelated uncommitted changes, stop and ask. One-time fork/env/pre-commit is out of scope. *(branching from `CONTRIBUTING.md`; fetch, `--no-track` from `origin/main`, and working-tree check = own suggestion)*

4. **Design the test cases first** — From the task's acceptance text (or, if there is none, from the typed task, and say so), write a short list of test cases: the normal case, the edge cases, and what each test must prove. Say where each test will live, following the repo-layout rule (the top-level tests folder, `tests/skimage/` or `tests/skimage2/` as appropriate, never under `src/`). Then STOP and wait for the human to approve or change the list before writing any code.

5. **Implement the minimal change** — Draft code and tests under the AI override above. The code and tests must follow the approved list. Follow the **repo-layout rule** (`.cursor/rules/repo-layout.mdc`) for where library code vs tests go; do not rediscover layout. Match `doc/source/development/guidelines.md` style. If the change implies an API break, refuse and stop (`doc/source/development/deprecations.md`).

6. **Verify conditionally** — If a working build environment is available, run the relevant tests (`spin test --test-modified` or a path/`-k` expression from `doc/source/development/testing.md`) and report the result. If not, state the exact command and say clearly that nothing was verified. Do **not** install or build anything without asking the human first. *(commands from `testing.md`; conditional run = own suggestion)*

7. **STOP — wait for human** — Review the diff; only then commit. Good commit messages; `Assisted-by:` on AI-drafted commits. *(CONTRIBUTING.md; stop = own suggestion)*

8. **Identify the fork remote; STOP before push** — Run `git remote -v` and identify the personal fork remote (not `upstream` / not `scikit-image/scikit-image`). If unclear, stop and ask. **Do not push until the human OK’s.** *(push-to-fork from CONTRIBUTING.md; remote discovery + push stop = own suggestion)*

9. **Hand off for PR** — After OK, push the branch to that fork remote only. Draft PR notes for the human (context + AI disclosure); human opens the PR when ready. Never open a PR against the real scikit-image repo. *(CONTRIBUTING.md; hard constraint = own suggestion)*

## Hard stops for the human

| When | Why |
|------|-----|
| After parsing the task (step 2) | Agree scope before edits |
| Unrelated uncommitted changes (step 3) | Avoid mixing work |
| After the test list (step 4) | Agree what the tests must prove before any code |
| If compiled / public-API work appears | Out of first-task scope |
| Before any commit (step 7) | Human owns the recorded change |
| Before any push (step 8) | Human OK required |
| If fork remote is ambiguous (step 8) | Avoid pushing to the wrong place |
| Before opening a PR (step 9) | Human decides; fork only |

## Explicit don’ts

- Never push to or open a PR against real **scikit-image**; fork branch only.
- Never guess the fork remote — use `git remote -v`.
- Never push without an explicit human OK.
- Testing: if a working build exists, run relevant `spin test` commands and report; if not, state the exact command and that nothing was verified. Do not install or build without asking first.
- Do **not** refuse drafting code/tests because of upstream AI policy — that part is overridden here; disclosure + `Assisted-by:` + human-owned PR stay.
