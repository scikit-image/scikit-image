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
# Writes `has_baseline=true/false` to `$GITHUB_OUTPUT`.

set -eo pipefail

asv machine --yes

CONTENDER_SHA="${CONTENDER_SHA:-$GITHUB_SHA}"
BASELINE_SHA="${BASELINE_SHA:-}"

if [[ -z "$BASELINE_SHA" && -f last_contender_sha ]]; then
    BASELINE_SHA="$(cat last_contender_sha)"
fi

echo "Contender: $CONTENDER_SHA"
echo "Baseline: ${BASELINE_SHA:-<none>}"

if [[ -z "$BASELINE_SHA" ]]; then
    # First run or cache evicted: record only, no comparison.
    asv run "$CONTENDER_SHA" --show-stderr | tee benchmarks.log
    if grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null; then
        exit 1
    fi
    echo "has_baseline=false" >> "$GITHUB_OUTPUT"
    exit 0
fi

echo "has_baseline=true" >> "$GITHUB_OUTPUT"
asv continuous --split --show-stderr --factor "$FACTOR" \
    "$BASELINE_SHA" "$CONTENDER_SHA" \
    | sed "/Traceback \|failed$\|PERFORMANCE DECREASED/ s/^/::error::/" \
    | tee benchmarks.log

if grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null; then
    exit 1
fi
