---
name: pre-pr-gate
description: >-
  Run local pre-PR checks before opening a pull request: pre-commit and
  spin test --test-modified via validate-contribution.sh. Use when the user
  is ready for a PR, wants to validate a branch, or asks for pre-PR / CI-local
  verification.
---

# Pre-PR gate

Run **local checks that mirror PR CI** before handoff. Do not open or push a PR unless the user explicitly asks.

**Script (canonical commands):** [tools/cursor/validate-contribution.sh](../../../tools/cursor/validate-contribution.sh)

Norms: **AGENTS.md** § Verification and § Pull requests. Test quality: [scaffold-test](../scaffold-test/SKILL.md).

## When to use

- User says they are ready for a PR or want branch validation
- End of **Developer** implementation work (after code + tests)
- **First contribution** § Verify — run this skill instead of ad hoc pre-commit/spin commands

## Workflow

```
Pre-PR gate:
- [ ] Scope matches issue / request (no drive-by changes)
- [ ] Weak-test checklist (if src/ or tests/ changed) — [scaffold-test](../scaffold-test/SKILL.md)
- [ ] ./tools/cursor/validate-contribution.sh
- [ ] PR metadata checklist (below)
- [ ] Report commands and outcomes to the user
```

### 1. Scope

Confirm changes match the stated issue or task. If unrelated edits exist, revert or split before validating.

### 2. Test quality (if applicable)

If `src/` or `tests/` changed, complete the weak-test checklist in [scaffold-test](../scaffold-test/SKILL.md) before running the script.

### 3. Run validation

From the repository root:

```bash
chmod +x tools/cursor/validate-contribution.sh   # once per clone, if needed
./tools/cursor/validate-contribution.sh
```

Optional targeted tests (same flags as the script):

```bash
./tools/cursor/validate-contribution.sh --module tests/skimage2/<subpackage>/
```

If Cython or `meson.build` changed and tests fail to collect, run `spin build --clean` per **AGENTS.md**, then re-run the script.

On failure: fix issues, re-run until the script exits 0. Do **not** claim success without a passing run.

**Script heuristics (no manual symbol lists):**

- `src/` behavior changes (`.py` / Cython / `meson.build` under `src/_skimage2/` or `src/skimage/`) must include changes under `tests/skimage/` or `tests/skimage2/`.
- New `deprecate_func`, `deprecate_parameter`, or `warn_external` lines under `src/` must include an update to `TODO.txt` (see CONTRIBUTING deprecation cycle).

If the change truly has no test impact (e.g. comment-only — rare for `src/`), the user may re-run with:

```bash
./tools/cursor/validate-contribution.sh --allow-no-tests
```

Use `--allow-no-tests` sparingly; note the reason in the PR handoff. There is no flag to skip the `TODO.txt` check when adding deprecations.

### 4. PR metadata checklist

The script does not check these — confirm with the user:

- [ ] Generative tools disclosed in the PR description ([AI policy](../../../CONTRIBUTING.md))
- [ ] `Fixes #N` / `Closes #N` when appropriate
- [ ] Optional `release-note` block for non-trivial user-facing changes ([PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md))
- [ ] Reminder: new PRs need a **category label** from a maintainer or CI may fail

### 5. Handoff

Summarize: files changed, intent, validate-contribution result, and suggested PR title/body bullets.

Do not commit, push, merge, or create a PR unless the user explicitly requests it.

## Out of scope

- Full suite (`spin test` without `--test-modified`) unless the user asks or changes affect test infrastructure
- `--skip-pre-commit` except when the user explicitly needs a debug iteration (note it in the handoff)
- `--allow-no-tests` unless the user confirms the src/ change does not need tests (note it in the handoff)
- Editing protected paths (see `.cursor/rules/security.mdc`)
