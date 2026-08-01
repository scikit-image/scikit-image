# Reinstate Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `asv` benchmark suite run automatically on merge to `main`, file/update a GitHub issue when a merge regresses performance, and target current numpy (2.x) instead of the stale `1.24` pin.

**Architecture:** All changes are configuration/CI, no application code. `.github/workflows/benchmarks.yaml` gains a `push`-to-`main` trigger, a commit-resolution step that computes the right baseline/contender SHAs per event type (PR label vs. push vs. manual dispatch), and a failure-reporting step that reuses the repo's existing `JasonEtco/create-an-issue` + `MAIN_FAIL_TEMPLATE.md` convention (already used by `nightly-wheel-build.yaml` / `test-nightlies-on-main.yaml`). `asv.conf.json` drops its numpy pin. `benchmarks/README_CI.md` gets factual corrections.

**Tech Stack:** GitHub Actions (YAML), `asv` (airspeed velocity benchmarking), bash, JSON.

## Global Constraints

- Every `actions/checkout` step must include `persist-credentials: false` (already present; do not remove it).
- Any third-party action pinned to a SHA must have that SHA verified against its tag comment before use. `JasonEtco/create-an-issue@1b14a70e4d8dc185e5cc76d3bec9eab20257b2c5 # v2.9.2` was verified via `gh api repos/JasonEtco/create-an-issue/git/refs/tags/v2.9.2` — the tag's commit SHA matches exactly. Reuse this exact pin; do not introduce a new unverified one.
- Untrusted/expression-derived values (PR labels, branch names, SHAs) must be passed into `run:` steps via an `env:` block, never interpolated directly into the shell script text via `${{ }}` — this is the existing mitigation in the `Run benchmarks` step (see the "Escape user controlled variables" comment) and must be preserved for any new step that consumes such values.
- The existing PR-label guard must remain intact: benchmarks only run on a PR when a label whose name contains `benchmark` is added (`contains(github.event.label.name, 'benchmark')`).
- `asv.conf.json` must remain valid JSON; `.github/workflows/benchmarks.yaml` must remain valid YAML and pass the repo's `zizmor` pre-commit hook (`.pre-commit-config.yaml:46-49`).
- This repo has no `justfile` — do not run `just typing` / `just pre-commit`. Use `pre-commit run --files <paths>` directly for validation instead.

---

### Task 1: Unpin numpy in `asv.conf.json`

**Files:**
- Modify: `asv.conf.json:19`

**Interfaces:**
- None (standalone config file, no other task depends on its content).

- [ ] **Step 1: Change the numpy matrix pin**

In `asv.conf.json`, change:

```json
         "numpy": ["1.24"],
```

to:

```json
         "numpy": [],
```

- [ ] **Step 2: Validate JSON syntax**

Run: `python -m json.tool asv.conf.json`
Expected: pretty-printed JSON output, no error, and the output shows `"numpy": []` under `matrix`.

- [ ] **Step 3: Commit**

```bash
git add asv.conf.json
git commit -m "Unpin numpy in asv benchmark matrix to target numpy 2.x"
```

---

### Task 2: Add push-to-main trigger and extend job guard

**Files:**
- Modify: `.github/workflows/benchmarks.yaml:1-19`

**Interfaces:**
- Produces: the job-level `if:` condition that Task 3's and Task 4's steps run inside (their steps live in the same `benchmark` job, so this task's guard governs whether they execute at all).

- [ ] **Step 1: Add the `push` trigger**

In `.github/workflows/benchmarks.yaml`, change the `on:` block from:

```yaml
on:
  pull_request:
    types: [labeled]
  workflow_dispatch:
```

to:

```yaml
on:
  pull_request:
    types: [labeled]
  push:
    branches:
      - main
  workflow_dispatch:
```

- [ ] **Step 2: Add `issues: write` permission**

Change:

```yaml
permissions:
  contents: read
```

to:

```yaml
permissions:
  contents: read
  issues: write
```

- [ ] **Step 3: Extend the job guard**

Change:

```yaml
  benchmark:
    if: contains(github.event.label.name, 'benchmark') || github.event_name == 'workflow_dispatch'
```

to:

```yaml
  benchmark:
    if: >-
      contains(github.event.label.name, 'benchmark') ||
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'push' && github.ref == 'refs/heads/main')
```

- [ ] **Step 4: Validate YAML syntax**

Run: `pre-commit run check-yaml --files .github/workflows/benchmarks.yaml`
Expected: `Passed` (exit 0). The ambient environment has no `pyyaml` installed, so use this hook — which runs in its own isolated venv — rather than `python -c "import yaml"`.

- [ ] **Step 5: Run zizmor lint**

Run: `pre-commit run zizmor --files .github/workflows/benchmarks.yaml`
Expected: passes (exit 0). If zizmor flags the new `if:` or triggers, fix per its message before continuing — do not suppress the finding.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/benchmarks.yaml
git commit -m "Trigger benchmark workflow on push to main"
```

---

### Task 3: Resolve baseline/contender commits per event type

**Files:**
- Modify: `.github/workflows/benchmarks.yaml` (insert a new step before `Run benchmarks`, and modify the `Run benchmarks` step's `env:` block)

**Interfaces:**
- Consumes: nothing from other tasks (reads directly from `github.event.*` context and git history).
- Produces: step outputs `steps.commits.outputs.baseline_sha`, `steps.commits.outputs.baseline_label`, `steps.commits.outputs.contender_label`, consumed by the `Run benchmarks` step's `env:` block in this same task.

- [ ] **Step 1: Insert the "Determine comparison commits" step**

In `.github/workflows/benchmarks.yaml`, insert this new step immediately after the `"Restore ccache"` step and before the `"Run benchmarks"` step:

```yaml
      - name: Determine comparison commits
        id: commits
        shell: bash
        env:
          EVENT_NAME: ${{ github.event_name }}
          PR_BASE_SHA: ${{ github.event.pull_request.base.sha }}
          PR_BASE_LABEL: ${{ github.event.pull_request.base.label }}
          PR_HEAD_LABEL: ${{ github.event.pull_request.head.label }}
          PUSH_BEFORE_SHA: ${{ github.event.before }}
        run: |
          set -e
          if [ "$EVENT_NAME" = "pull_request" ]; then
            baseline_sha="$PR_BASE_SHA"
            baseline_label="$PR_BASE_LABEL"
            contender_label="$PR_HEAD_LABEL"
          elif [ "$EVENT_NAME" = "push" ]; then
            before="$PUSH_BEFORE_SHA"
            if [ -z "$before" ] || [ "$before" = "0000000000000000000000000000000000000000" ]; then
              before=$(git rev-parse "$GITHUB_SHA~1")
            fi
            baseline_sha="$before"
            baseline_label="$before"
            contender_label="$GITHUB_SHA"
          else
            baseline_sha=$(git rev-parse "$GITHUB_SHA~1")
            baseline_label="$baseline_sha"
            contender_label="$GITHUB_SHA"
          fi
          echo "baseline_sha=$baseline_sha" >> "$GITHUB_OUTPUT"
          echo "baseline_label=$baseline_label" >> "$GITHUB_OUTPUT"
          echo "contender_label=$contender_label" >> "$GITHUB_OUTPUT"
```

Note: `GITHUB_SHA` and `GITHUB_OUTPUT` are default environment variables GitHub Actions provides to every step; they don't need to be declared in `env:`.

- [ ] **Step 2: Rewire `Run benchmarks` to use the new step's outputs**

In the same file, change the `Run benchmarks` step's `env:` block from:

```yaml
        env:
          OPENBLAS_NUM_THREADS: 1
          MKL_NUM_THREADS: 1
          OMP_NUM_THREADS: 1
          ASV_FACTOR: 1.5
          ASV_SKIP_SLOW: 1
          # Escape user controlled variables
          BASELINE_SHA: "${{ github.event.pull_request.base.sha }}"
          BASELINE_LABEL: "${{ github.event.pull_request.base.label }}"
          CONTENDER_LABEL: "${{ github.event.pull_request.head.label }}"
```

to:

```yaml
        env:
          OPENBLAS_NUM_THREADS: 1
          MKL_NUM_THREADS: 1
          OMP_NUM_THREADS: 1
          ASV_FACTOR: 1.5
          ASV_SKIP_SLOW: 1
          # Escape user controlled variables
          BASELINE_SHA: "${{ steps.commits.outputs.baseline_sha }}"
          BASELINE_LABEL: "${{ steps.commits.outputs.baseline_label }}"
          CONTENDER_LABEL: "${{ steps.commits.outputs.contender_label }}"
```

Do not change anything else in this step (the `run:` script body stays as-is).

- [ ] **Step 3: Validate YAML syntax**

Run: `pre-commit run check-yaml --files .github/workflows/benchmarks.yaml`
Expected: `Passed` (exit 0). The ambient environment has no `pyyaml` installed, so use this hook — which runs in its own isolated venv — rather than `python -c "import yaml"`.

- [ ] **Step 4: Run zizmor lint**

Run: `pre-commit run zizmor --files .github/workflows/benchmarks.yaml`
Expected: passes (exit 0). This specifically checks that the new step doesn't reintroduce script injection — confirm no `${{ github.event.pull_request.* }}` or `${{ github.event.before }}` expression appears directly inside any `run:` block's script text (they must only appear inside `env:` blocks).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/benchmarks.yaml
git commit -m "Resolve benchmark baseline/contender commits for push and dispatch events"
```

---

### Task 4: Report benchmark regressions on main as a GitHub issue

**Files:**
- Modify: `.github/workflows/benchmarks.yaml` (append a new step at the end of the `benchmark` job)

**Interfaces:**
- Consumes: the job's overall `failure()` status (set when the `Run benchmarks` step `exit 1`s on a detected regression or error) and `github.ref`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Append the failure-reporting step**

In `.github/workflows/benchmarks.yaml`, add this as the last step in the `benchmark` job, after the `actions/upload-artifact@v4` step:

```yaml
      - name: "Job has failed: reporting"
        if: ${{ failure() && github.ref == 'refs/heads/main' }}
        uses: JasonEtco/create-an-issue@1b14a70e4d8dc185e5cc76d3bec9eab20257b2c5 # v2.9.2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          BUILD_TYPE: "benchmark"
        with:
          filename: .github/MAIN_FAIL_TEMPLATE.md
          update_existing: true
```

- [ ] **Step 2: Validate YAML syntax**

Run: `pre-commit run check-yaml --files .github/workflows/benchmarks.yaml`
Expected: `Passed` (exit 0). The ambient environment has no `pyyaml` installed, so use this hook — which runs in its own isolated venv — rather than `python -c "import yaml"`.

- [ ] **Step 3: Run zizmor lint**

Run: `pre-commit run zizmor --files .github/workflows/benchmarks.yaml`
Expected: passes (exit 0).

- [ ] **Step 4: Confirm the SHA pin is unchanged from the verified value**

Run: `grep -o "JasonEtco/create-an-issue@[a-f0-9]*" .github/workflows/benchmarks.yaml`
Expected output: `JasonEtco/create-an-issue@1b14a70e4d8dc185e5cc76d3bec9eab20257b2c5` — this must match the SHA already used in `.github/workflows/test-nightlies-on-main.yaml`. Confirm with:
`grep -o "JasonEtco/create-an-issue@[a-f0-9]*" .github/workflows/test-nightlies-on-main.yaml`
Both commands must print the same SHA.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/benchmarks.yaml
git commit -m "File a GitHub issue when a benchmark run fails on main"
```

---

### Task 5: Fix stale documentation in `benchmarks/README_CI.md`

**Files:**
- Modify: `benchmarks/README_CI.md:1-27` (header comment and "How it works" / "Running the benchmarks" sections)

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Fix the workflow filename reference**

Change:

```markdown
The `asv` suite can be run for any PR on GitHub Actions (check workflow `.github/workflows/benchmarks.yml`) by adding a `run-benchmark` label to said PR. This will trigger a job that will run the benchmarking suite for the current PR head (merged commit) against the PR base (usually `main`).
```

to:

```markdown
The `asv` suite can be run for any PR on GitHub Actions (check workflow `.github/workflows/benchmarks.yaml`) by adding a label containing `benchmark` (e.g. `run-benchmark`) to said PR. This will trigger a job that will run the benchmarking suite for the current PR head (merged commit) against the PR base (usually `main`).

The suite also runs automatically on every merge to `main` (comparing the new commit against the commit `main` pointed to immediately before the merge), and can be triggered manually via `workflow_dispatch` from the `Actions` tab. If a run on `main` detects a regression, it opens (or updates, if one is already open) a `CI failure`-labeled GitHub issue using the same convention as the repo's other main-branch CI checks.
```

- [ ] **Step 2: Fix the label instructions**

Change:

```markdown
## Running the benchmarks on GitHub Actions

1. On a PR, add the label `run-benchmark`.
```

to:

```markdown
## Running the benchmarks on GitHub Actions

1. On a PR, add a label whose name contains `benchmark` (e.g. `run-benchmark`).
```

- [ ] **Step 3: Update the "Last updated" comment**

Change:

```markdown
<!-- Last updated: 2021.07.06 -->
```

to:

```markdown
<!-- Last updated: 2026.08.01 -->
```

- [ ] **Step 4: Review the rendered diff**

Run: `git diff benchmarks/README_CI.md`
Expected: only the three edits above; no other content changed (the "artifacts", "re-running the analysis", and "skipping slow tests" sections are untouched).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/README_CI.md
git commit -m "Fix stale benchmark CI docs (filename, label, merge trigger)"
```

---

### Task 6: End-to-end validation

**Files:**
- None modified; this task only runs verification commands across the files touched in Tasks 1-5.

**Interfaces:**
- Consumes: the final state of `asv.conf.json`, `.github/workflows/benchmarks.yaml`, `benchmarks/README_CI.md` from Tasks 1-5.

- [ ] **Step 1: Run the full pre-commit suite on changed files**

```bash
pre-commit run --files asv.conf.json .github/workflows/benchmarks.yaml benchmarks/README_CI.md
```

Expected: all hooks pass. If a hook reformats a file, re-stage it (`git add <file>`) and re-run this command until clean, then amend or add a follow-up commit.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin reinstate-benchmarks
```

- [ ] **Step 3: Manually trigger the workflow to test the non-PR path**

```bash
gh workflow run benchmarks.yaml --ref reinstate-benchmarks
```

Wait ~30 seconds, then find the run:

```bash
gh run list --workflow=benchmarks.yaml --branch=reinstate-benchmarks --limit=1
```

- [ ] **Step 4: Watch the run and confirm the commit-resolution logic works**

```bash
gh run watch $(gh run list --workflow=benchmarks.yaml --branch=reinstate-benchmarks --limit=1 --json databaseId --jq '.[0].databaseId')
```

Expected: the job starts, and its log for the "Determine comparison commits" step shows non-empty `baseline_sha` and `contender_label` values (visible via `gh run view <id> --log | grep -A3 "Determine comparison commits"` if you want to inspect after completion). Since this is a `workflow_dispatch` run, it takes the `else` branch in Task 3's script, so `baseline_sha` should equal the parent commit of the branch tip.

Note: this run exercises the job but not the Task 4 issue-filing step (`github.ref` won't be `refs/heads/main` on a feature branch), and not the `push` trigger from Task 2 (that only fires on an actual push to `main`, which happens once this branch is merged). Both are covered by code review of the diff plus the zizmor/YAML validation already done in Tasks 2-4, since they can't be exercised pre-merge.

- [ ] **Step 5: Report the outcome**

If the run in Step 4 fails for a reason unrelated to this plan's changes (e.g. a flaky benchmark, an unrelated environment issue), note it — that's expected/acceptable per the spec's "Testing" section, which only requires confirming the job *runs* and produces sane baseline/contender values, not that every benchmark passes. If it fails because of a workflow syntax or logic error introduced in Tasks 1-5, fix the relevant task's file and re-run from Step 3.
