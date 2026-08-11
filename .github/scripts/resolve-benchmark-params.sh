#!/usr/bin/env bash
# Determine the baseline/contender commits to compare, and the asv run
# parameters (comparison factor, process count, skip-slow, full-params,
# and an optional benchmark-module filter) for the current trigger event.
#
# Required env vars:
#   EVENT_NAME:        github.event_name
#   GITHUB_SHA:        the current commit (provided by default by Actions)
#   GITHUB_REPOSITORY: "owner/repo" (provided by default by Actions; used
#                       for the nightly-baseline gh run list lookup)
#   GH_TOKEN:          token for gh CLI (PR label lookup, nightly baseline
#                       lookup - needs `actions: read` for the latter)
#   PR_NUMBER:         pull request number (empty for non-PR events)
#   PR_BASE_SHA:       github.event.pull_request.base.sha
#   PR_BASE_LABEL:     github.event.pull_request.base.label
#   PR_HEAD_SHA:       github.event.pull_request.head.sha
#   PR_HEAD_LABEL:     github.event.pull_request.head.label
#   DISPATCH_BASELINE: github.event.inputs.baseline (workflow_dispatch
#                       only; "parent-commit" or "previous-nightly")
#
# Writes should_run, baseline_sha, baseline_label, contender_label,
# asv_factor, asv_processes, asv_skip_slow, asv_full_params,
# bench_filter, and python_version to $GITHUB_OUTPUT.

set -euo pipefail

# Map changed src/skimage/<subpackage>/ paths to the benchmark module(s)
# that cover them. Subpackages with no corresponding benchmark file, or
# changes outside src/skimage/ entirely (docs, CI config, etc.), simply
# don't contribute a module - they neither force nor block a run.
bench_modules_for_path_changes() {
  local changed="$1"
  local modules=""
  local extra
  for pkg in exposure feature filters graph measure metrics morphology registration restoration segmentation transform util; do
    if ! echo "$changed" | grep -q "^src/skimage/$pkg/"; then
      continue
    fi
    case "$pkg" in
      feature) extra="benchmark_feature|benchmark_peak_local_max" ;;
      filters) extra="benchmark_filters|benchmark_rank" ;;
      transform) extra="benchmark_transform|benchmark_transform_warp|benchmark_interpolation" ;;
      *) extra="benchmark_$pkg" ;;
    esac
    if [ -z "$modules" ]; then
      modules="$extra"
    else
      modules="$modules|$extra"
    fi
  done
  echo "$modules"
}

# The commit compared by the last successful nightly (schedule) run of
# this workflow, or empty if none exists yet.
last_nightly_sha() {
  gh run list --repo "$GITHUB_REPOSITORY" --workflow=benchmarks.yaml \
    --event=schedule --status=success --limit=1 --json headSha --jq '.[0].headSha // empty'
}

if [ "$EVENT_NAME" = "pull_request" ]; then
  baseline_sha="$PR_BASE_SHA"
  baseline_label="$PR_BASE_LABEL"
  contender_label="$PR_HEAD_LABEL"
  asv_factor="1.5"
  asv_processes="2"
  asv_skip_slow="1"
  asv_full_params="0"

  # Fetch label state dynamically (not from the workflow trigger
  # payload) so that re-running the workflow after adding the
  # 'benchmark' label picks up the new label state correctly.
  has_benchmark_label=$(gh pr view "$PR_NUMBER" --json labels \
    --jq '[.labels[].name] | map(test("benchmark"; "i")) | any')

  if [ "$has_benchmark_label" = "true" ]; then
    # Explicit override: always run the full suite, regardless of
    # which paths changed.
    should_run="true"
    bench_filter=""
  else
    changed=$(git diff --name-only "$PR_BASE_SHA" "$PR_HEAD_SHA" -- src/skimage/)
    modules=$(bench_modules_for_path_changes "$changed")
    if [ -z "$modules" ]; then
      should_run="false"
      bench_filter=""
    else
      should_run="true"
      bench_filter="^($modules)\."
    fi
  fi
elif [ "$EVENT_NAME" = "schedule" ]; then
  # Nightly: full, untrimmed suite against the commit compared by the
  # last successful nightly run, not just main's immediate parent - this
  # catches cumulative drift across however many PRs merged since then,
  # not just the single most recent one. Falls back to the immediate
  # parent if no prior successful nightly run exists yet (e.g. the very
  # first time this schedule fires).
  prev_sha=$(last_nightly_sha)
  if [ -z "$prev_sha" ] || ! git cat-file -e "$prev_sha^{commit}" 2>/dev/null; then
    prev_sha=$(git rev-parse "$GITHUB_SHA~1")
  fi
  baseline_sha="$prev_sha"
  baseline_label="$prev_sha"
  contender_label="$GITHUB_SHA"
  asv_factor="1.5"
  asv_processes="2"
  asv_skip_slow="0"
  asv_full_params="1"
  should_run="true"
  bench_filter=""
else
  # workflow_dispatch: compare against either the immediate parent
  # commit (default) or the commit compared by the last successful
  # nightly run, per the "baseline" dispatch input. Falls back to the
  # immediate parent if "previous-nightly" is requested but no prior
  # successful nightly run exists yet.
  if [ "${DISPATCH_BASELINE:-parent-commit}" = "previous-nightly" ]; then
    baseline_sha=$(last_nightly_sha)
    if [ -z "$baseline_sha" ] || ! git cat-file -e "$baseline_sha^{commit}" 2>/dev/null; then
      baseline_sha=$(git rev-parse "$GITHUB_SHA~1")
    fi
  else
    baseline_sha=$(git rev-parse "$GITHUB_SHA~1")
  fi
  baseline_label="$baseline_sha"
  contender_label="$GITHUB_SHA"
  asv_factor="1.5"
  asv_processes="2"
  asv_skip_slow="1"
  asv_full_params="0"
  should_run="true"
  bench_filter=""
fi

echo "should_run=$should_run" >> "$GITHUB_OUTPUT"
echo "baseline_sha=$baseline_sha" >> "$GITHUB_OUTPUT"
echo "baseline_label=$baseline_label" >> "$GITHUB_OUTPUT"
echo "contender_label=$contender_label" >> "$GITHUB_OUTPUT"
echo "asv_factor=$asv_factor" >> "$GITHUB_OUTPUT"
echo "asv_processes=$asv_processes" >> "$GITHUB_OUTPUT"
echo "asv_skip_slow=$asv_skip_slow" >> "$GITHUB_OUTPUT"
echo "asv_full_params=$asv_full_params" >> "$GITHUB_OUTPUT"
echo "bench_filter=$bench_filter" >> "$GITHUB_OUTPUT"

# Single source of truth for the Python version used to build and run
# the benchmarks, so it can't drift out of sync with asv's own config.
python_version=$(jq -r '.pythons[0]' asv.conf.json)
echo "python_version=$python_version" >> "$GITHUB_OUTPUT"
