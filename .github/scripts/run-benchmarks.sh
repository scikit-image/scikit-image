#!/usr/bin/env bash
# Run the asv benchmark comparison for the current commit against the
# resolved baseline, and fail the step if a regression or error is
# detected.
#
# Expects prepare-benchmarks.sh to have run first; it puts these into
# the job environment from benchmark-params.json:
#   ASV_FACTOR:      the asv --factor threshold
#   ASV_PROCESSES:   the asv processes/rounds count
#   BASELINE_SHA:    baseline commit SHA
#   BASELINE_LABEL:  baseline display label (for the printed summary)
#   CONTENDER_LABEL: contender display label (for the printed summary)
#   BENCH_FILTER:    optional asv -b regex filter (empty = run everything)
#
# ASV_SKIP_SLOW and ASV_FULL_PARAMS come from the same file but are read
# by benchmarks/__init__.py in the benchmark processes, not here.
#
#   GITHUB_SHA:      contender commit SHA (provided by default by Actions)

set -euo pipefail
set -x

# Stage the log and this suite's README next to asv's own results for
# the artifact upload. On an EXIT trap because the log is most useful
# on the runs that failed, including the ones that fail before
# benchmarking starts and never write a log at all.
collect_results() {
    status=$?
    mkdir -p .asv/results
    cp benchmarks/README_CI.md .asv/results/
    if [ -f benchmarks.log ]; then
        cp benchmarks.log .asv/results/
    fi
    return $status
}
trap collect_results EXIT

echo "Baseline: $BASELINE_SHA ($BASELINE_LABEL)"
echo "Contender: $GITHUB_SHA ($CONTENDER_LABEL)"

# --verbose (DEBUG-level logging) makes env-creation/pip-install progress
# visible instead of one large silent gap. `sed -u` (unbuffered) streams
# that output in real time; buffered, it was stalling the whole pipeline,
# not just delaying output.
ASV_OPTIONS="--verbose --split --show-stderr --factor $ASV_FACTOR -a processes=$ASV_PROCESSES"
if [ -n "$BENCH_FILTER" ]; then
    ASV_OPTIONS="$ASV_OPTIONS -b $BENCH_FILTER"
fi

set +e
asv continuous $ASV_OPTIONS $BASELINE_SHA $GITHUB_SHA \
    | sed -u "/Traceback \|failed$\|PERFORMANCE DECREASED/ s/^/::error::/" \
    | tee benchmarks.log
asv_status=${PIPESTATUS[0]}
set -e

if [ "$asv_status" -ne 0 ] || grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null ; then
    exit 1
fi
