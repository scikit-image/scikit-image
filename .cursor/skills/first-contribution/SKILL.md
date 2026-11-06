---
name: first-contribution
description: Guides a scoped first library contribution on this scikit-image fork. Use when working an Acme first-contribution ticket, tracing wrappers to implementations, proposing a plan and tests, or assisting after human approval.
---

# First contribution

Follow this workflow for a ticketed first contribution. Stop for human review before implementation. Do not treat Cursor rules as a sandbox of file or network access.

## Policy sources

- **Synthesized Acme policy:** current ticket; `.cursor/rules/acme-scope.mdc`. Consult the ticket for acceptance criteria and approved files. Do not copy a prior ticket's file list into later work.
- **Upstream conventions:** `CONTRIBUTING.md` (AI policy), `doc/source/development/guidelines.md`, `doc/source/development/testing.md`, `doc/source/development/deprecations.md`, `doc/source/skips/4-transition-to-v2.rst`.
- The **human engineer owns test scenarios and expected results.**

Respect the upstream AI policy: explore and iterate; do not author novel algorithms or tests; AI may help with scaffolding after the human specifies cases; disclose AI use; keep diffs trivial to review.

## Workflow

1. **Read the ticket and clarify acceptance criteria.** Restate observed vs intended behavior, in-scope vs out-of-scope changes, and anything ambiguous. Do not implement yet.

2. **Trace public wrappers to the implementation.** Follow `skimage` / `skimage2` into `_skimage2`. Note shared code objects vs distinct wrapper objects. Inspect the current ticket's named sources, not a hardcoded accelerator file list.

3. **Load relevant conventions and inspect neighboring tests.** Read the documents above and existing tests in the matching `tests/skimage/` and `tests/skimage2/` directories.

4. **Propose a scoped implementation plan and test scenarios.** Include the smallest code change, files you believe need edits, and suggested scenarios. Mark suggestions as proposals. Escalate if the plan would leave the ticket's boundaries.

5. **Stop for human review before implementation.** Wait for explicit approval of the plan, scenarios, and expected results.

6. **After approval, assist with the minimal change and test scaffolding.** Implement only the approved behavior. Help structure tests the human specified. Do not add dependencies, CI, or unrelated refactors.

7. **Run appropriate local checks.** Prefer the ticket's documented command when present; otherwise a focused `spin test -- <paths>` for the touched test modules (`doc/source/development/testing.md`).

8. **Report** changed files, actual commands and results, remaining review/CI/maintainer steps, and anything mocked or unverified. Local green is not CI, Grok Bot QA, merge, or deployment.
