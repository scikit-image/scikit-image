# Reinstate benchmarks

Issue: https://github.com/scikit-image/scikit-image/issues/8199

## Problem

`asv` benchmark infrastructure already exists (`benchmarks/`, `asv.conf.json`,
`.github/workflows/benchmarks.yaml`) but is incomplete per #8199:

1. Benchmarks only run when a PR gets a `benchmark`-ish label, or via manual
   `workflow_dispatch`. They never run automatically on merge to `main`.
2. There's no reinstated notification when a merge to `main` regresses
   performance beyond the existing `ASV_FACTOR` threshold — today a
   PR-triggered run just fails its own check and uploads an artifact, which
   is invisible once the PR is merged and the run is gone.
3. `asv.conf.json` pins `matrix.numpy` to `["1.24"]`, inconsistent with the
   rest of the project's `numpy>=2.0` requirement.

## Non-goals

- Not changing the PR-label-triggered flow's comparison logic (base vs head
  SHA) — it already works.
- Not adding Slack/email notifications — the repo has no such integration
  anywhere, and issue #8199 doesn't ask for one.
- Not rewriting `benchmarks/README_CI.md` wholesale — only fixing factual
  drift (filename, label matching) and documenting the new merge trigger.

## Design

### 1. Trigger benchmarks on merge to `main`

Add a `push: branches: [main]` trigger to
`.github/workflows/benchmarks.yaml`, alongside the existing
`pull_request: types: [labeled]` and `workflow_dispatch` triggers.

The job-level `if:` gains one more OR'd condition, leaving the existing
PR-label guard (`contains(github.event.label.name, 'benchmark')`) and
`workflow_dispatch` condition untouched:

```yaml
if: >-
  contains(github.event.label.name, 'benchmark') ||
  github.event_name == 'workflow_dispatch' ||
  (github.event_name == 'push' && github.ref == 'refs/heads/main')
```

### 2. Resolve baseline/contender commits per event type

Today, `BASELINE_SHA` / `BASELINE_LABEL` / `CONTENDER_LABEL` are read
directly from `github.event.pull_request.*`, which is empty for `push` and
`workflow_dispatch` events. Replace this with a step that computes the
right values for each event type:

- `pull_request`: unchanged — PR base SHA vs PR head SHA/label.
- `push` (merge to `main`): baseline is `github.event.before` (the commit
  `main` pointed to immediately before this push), contender is
  `github.sha`. If `github.event.before` is the null SHA (e.g. branch
  creation — shouldn't normally happen on `main`, but guard anyway), fall
  back to `git rev-parse "${{ github.sha }}~1"`.
- `workflow_dispatch`: baseline is `git rev-parse "${{ github.sha }}~1"`,
  contender is `github.sha` (this path currently has no working baseline at
  all; this makes manual dispatch usable instead of leaving it broken).

This requires `fetch-depth: 0` on checkout, which is already set.

### 3. Report regressions on `main` as a GitHub issue

Reuse the existing repo convention for main-branch CI failures (used by
`nightly-wheel-build.yaml` and `test-nightlies-on-main.yaml`): on failure,
auto-file/update a GitHub issue via `JasonEtco/create-an-issue` and the
shared `.github/MAIN_FAIL_TEMPLATE.md` template.

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

SHA `1b14a70e4d8dc185e5cc76d3bec9eab20257b2c5` was verified against GitHub's
API to correspond exactly to tag `v2.9.2`.

`github.ref == 'refs/heads/main'` is false for `pull_request`-triggered
runs (their ref is `refs/pull/N/merge`), so this step is a no-op there —
only push-to-main and main-branch `workflow_dispatch` runs can file an
issue. This reuses the existing dedup behavior (`update_existing: true`):
repeated regressions update the same open issue rather than spamming new
ones.

Add `issues: write` to the workflow's `permissions:` block (currently
`contents: read` only), matching `test-nightlies-on-main.yaml`.

The existing pass/fail detection (`grep` over `benchmarks.log` for
`Traceback`, `failed`, or `PERFORMANCE DECREASED`, `exit 1` if found) is
unchanged — that's what `failure()` above keys off.

### 4. Target numpy 2.0

Change `asv.conf.json`'s `matrix.numpy` from `["1.24"]` to `[]` (no numpy
pin), so `asv` installs whatever numpy is otherwise resolved (currently
numpy 2.x, per the project's `numpy>=2.0` runtime requirement) instead of
forcing an outdated 1.x install. This avoids needing to bump this value
again on every future numpy release.

### 5. Fix stale docs in `benchmarks/README_CI.md`

Minimal factual corrections, no structural rewrite:

- Fix the referenced workflow filename (`benchmarks.yaml`, not
  `benchmarks.yml`).
- Fix the label description to match the actual `if:` check (any label
  containing `benchmark`, not literally `run-benchmark`).
- Add a short paragraph noting benchmarks also run automatically on merge
  to `main` (comparing against the prior commit on `main`), and that a
  regression files/updates a `CI failure`-labeled issue.

## Testing

- `workflow_dispatch` on this branch to confirm the job runs and the new
  commit-resolution step produces sane baseline/contender values (won't
  exercise the issue-filing step since `github.ref` won't be
  `refs/heads/main` on a feature branch).
- Manual review of the YAML (`actionlint`/`zizmor` if configured in
  pre-commit) for syntax correctness, since the `push`-to-`main` and
  issue-filing paths can't be fully exercised until this actually merges.
