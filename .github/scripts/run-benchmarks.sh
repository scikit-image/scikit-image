#!/usr/bin/env bash
# Run asv benchmarks: either record results only, or compare a baseline
# against a contender with a regression factor.
#
# Required env vars:
#   FACTOR: Regression factor passed to `asv continuous` (e.g. 1.1 or 1.5).
#
# Optional env vars:
#   BASELINE_SHA: SHA to compare against. If unset, a previous clean run's
#     SHA is read from `last_contender_sha` (nightly mode). If neither is
#     available, results are recorded only and no comparison is made.
#   CONTENDER_SHA: SHA to benchmark. Defaults to `$GITHUB_SHA`.
#
# Writes `has_baseline=true/false` to `$GITHUB_OUTPUT`. On a comparison
# failure it also writes `dependency_changes`, listing core dependencies
# whose version specifiers changed between the baseline and the contender.

set -eo pipefail

asv machine --yes

CONTENDER_SHA="${CONTENDER_SHA:-$GITHUB_SHA}"
BASELINE_SHA="${BASELINE_SHA:-}"

if [[ -z "$BASELINE_SHA" && -f last_contender_sha ]]; then
    BASELINE_SHA="$(cat last_contender_sha)"
fi

echo "Contender: $CONTENDER_SHA"
echo "Baseline: ${BASELINE_SHA:-<none>}"

record_dependency_changes() {
    [[ -n "$BASELINE_SHA" ]] || return 0
    {
        echo "dependency_changes<<EOF"
        python tools/generate_requirements.py --compare "$BASELINE_SHA" "$CONTENDER_SHA" || true
        echo "EOF"
    } >> "$GITHUB_OUTPUT"
}

if [[ -z "$BASELINE_SHA" ]]; then
    # First run or cache evicted: record only, no comparison.
    set +e
    asv run "$CONTENDER_SHA" --show-stderr | tee benchmarks.log
    status=${PIPESTATUS[0]}
    set -e
    if [[ $status -ne 0 ]] || grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null; then
        exit 1
    fi
    echo "has_baseline=false" >> "$GITHUB_OUTPUT"
    exit 0
fi

echo "has_baseline=true" >> "$GITHUB_OUTPUT"

set +e
asv continuous --split --show-stderr --factor "$FACTOR" \
    "$BASELINE_SHA" "$CONTENDER_SHA" \
    | sed "/Traceback \|failed$\|PERFORMANCE DECREASED/ s/^/::error::/" \
    | tee benchmarks.log
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]] || grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null; then
    record_dependency_changes
    exit 1
fi
