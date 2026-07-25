---
name: first-contribution
description: >-
  Guides a newcomer's first scikit-image pull request, preferring GitHub issues
  labeled ":beginner: Good first issue". Use when the user asks for a first
  contribution, first PR, starter issue, onboarding as a new contributor, or
  help picking an easy issue to work on.
---

# First contribution

Help someone land a **first** scikit-image PR. Stay inside this playbook; do not invent a parallel onboarding path.

Norms for layout, `spin`, pre-commit, and PR policy: **AGENTS.md** (always applied). Deep reference: **CONTRIBUTING.md** — link, do not paste.

## Issue selection

**Default:** open issues labeled `:beginner: Good first issue` (list with `gh` below). Prefer these strongly for a first PR.

If the user names an issue **without** that label (or with different labels):

1. Say clearly that labeled issues are the recommended first-contribution path and why (scoped, curated for newcomers).
2. Show a short list of `:beginner: Good first issue` alternatives.
3. Ask whether they want to switch to a labeled issue **or** proceed with their choice anyway.
4. **Do not edit code** until they explicitly confirm. Silence or ambiguity is not confirmation.
5. If they confirm proceeding off-label, continue this playbook on that issue, and note in the PR handoff that it was not a `:beginner: Good first issue`.

Do not expand scope beyond the chosen issue. No drive-by refactors, renames, or unrelated cleanups.

## Workflow

Copy and track:

```
First contribution:
- [ ] List / pick an issue (prefer :beginner: Good first issue)
- [ ] If off-label: strongly suggest labeled alternatives; get explicit confirmation
- [ ] Dev setup (if needed) — AGENTS.md § Build and test
- [ ] Branch from up-to-date main
- [ ] Implement only what the issue asks
- [ ] Add or update tests — read and follow [scaffold-test](../scaffold-test/SKILL.md)
- [ ] Verify — [pre-pr-gate skill](../pre-pr-gate/SKILL.md) / `./tools/cursor/validate-contribution.sh`
- [ ] Summarize for PR (link issue, AI disclosure)
```

### 1. List and pick

```bash
gh issue list --label ":beginner: Good first issue" --state open --limit 20
```

Show the user the list (number, title, URL). Help them choose; do not start coding until they confirm an issue number.

To inspect one issue:

```bash
gh issue view <number>
```

If labels include `:beginner: Good first issue`, proceed after the user confirms the issue number.

If not, follow **Issue selection** above (suggest labeled alternatives → wait for explicit confirmation before coding).

If the issue asks contributors to comment before starting, remind the user to comment on GitHub. Do not post as the user unless they explicitly ask you to run `gh issue comment`.

### 2. Setup (skip steps already done)

**Remotes (first-time clone)** — typical layout after forking:

```bash
git clone --origin upstream git@github.com:scikit-image/scikit-image
cd scikit-image
git remote add <your-github-username> git@github.com:<your-github-username>/scikit-image
git fetch <your-github-username>
```

- `upstream` — scikit-image/scikit-image
- `<your-github-username>` — personal fork (push target)

Many clones use `origin` as the fork and add `upstream` separately. Inspect with `git remote -v` and adapt commands; never run `git config`.

SSH help: https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

**Install, hooks, and spin commands:** follow **AGENTS.md** § Build and test (e.g. `spin install -v`, `pre-commit install`, `spin test -- …`, `spin build --clean` when adding Cython/source files).

### 3. Branch and implement

```bash
git switch main
git fetch upstream main
git merge upstream/main
git switch -c first-contribution-<issue-number>
```

If there is no `upstream` remote, use the project's primary remote for `main` (often `origin` on a fresh fork clone). Match remotes to how this clone is set up; do not rewrite git config.

- Read the issue and the files it points at before editing.
- Follow **AGENTS.md** § Package layout and **Read before edit**.
- Match surrounding style. Change only what the issue requires.

### Tests

If the issue changes code under `src/`, **read and follow** **[scaffold-test](../scaffold-test/SKILL.md)** (scaffold from peer tests, weak-test checklist). While editing test files, **skimage-tests.mdc** applies automatically.

Doc-only issues: skip unless the issue requires tests.

### 4. Verify

Follow **[pre-pr-gate](../pre-pr-gate/SKILL.md)**:

1. Complete the **weak-test checklist** in [scaffold-test](../scaffold-test/SKILL.md) when tests were added or changed.
2. Run from the repository root:

   ```bash
   ./tools/cursor/validate-contribution.sh
   ```

3. Complete the PR metadata checklist in the pre-pr-gate skill before handoff.

Do not claim checks passed unless the script (or equivalent steps) succeeded. Report commands and outcomes.

### 5. Hand off for PR

Do **not** commit, push, or open a PR unless the user explicitly asks.

When summarizing for the user, include:

- Issue link (`Fixes #N` or `Closes #N` when appropriate)
- What changed (files + intent)
- Commands run and results
- Concise PR title; disclose generative tools per **AGENTS.md** § Pull requests / CONTRIBUTING AI policy
- Reminder: CI fails on new PRs until a maintainer adds a category label (expected for newcomers)
- Do not merge; core team handles review/merge

## Getting unstuck

Point the user at:

- Zulip: https://skimage.zulipchat.com/
- Developer forum: https://discuss.scientific-python.org/c/contributor/skimage

## Out of scope for this skill

- Creating/relabeling GitHub issues
- Proactively hunting unlabeled "looks easy" work (only consider off-label issues when the user asks for a specific one)
- Cython/API design/deprecation work unless the chosen issue explicitly requires it
- Editing protected paths (see `.cursor/rules/security.mdc`)
