#!/usr/bin/env bash
# Run the asv benchmark comparison for the current commit against the
# resolved baseline, and fail the step if a regression or error is
# detected.
#
# Required env vars:
#   ASV_FACTOR:      the asv --factor threshold
#   ASV_PROCESSES:   the asv processes/rounds count
#   BASELINE_SHA:    baseline commit SHA
#   BASELINE_LABEL:  baseline display label (for the printed summary)
#   CONTENDER_LABEL: contender display label (for the printed summary)
#   GITHUB_SHA:      contender commit SHA (provided by default by Actions)
#   BENCH_FILTER:    optional asv -b regex filter (empty = run everything)

set -euo pipefail
set -x

python -m pip install virtualenv

# ID this runner
asv machine --yes

echo "Baseline: $BASELINE_SHA ($BASELINE_LABEL)"
echo "Contender: $GITHUB_SHA ($CONTENDER_LABEL)"

# Run benchmarks for current commit against base.
# --verbose (DEBUG-level logging) so env-creation/pip-install progress is
# actually visible instead of one large silent gap; `sed -u` (unbuffered)
# so that output streams through the pipeline in real time rather than
# waiting on sed's default block buffering when its stdout isn't a
# terminal - which was previously stalling the whole pipeline, not just
# hiding output.
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

# Report and export results for subsequent steps
if [ "$asv_status" -ne 0 ] || grep "Traceback \|failed\|PERFORMANCE DECREASED" benchmarks.log > /dev/null ; then
    exit 1
fi
