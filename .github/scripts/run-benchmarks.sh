#!/usr/bin/env bash
# Run the asv benchmark comparison for the current commit against the
# resolved baseline, and fail the step if a regression or error is
# detected.
#
# Required env vars, exported into the job environment from
# benchmark-params.json by export-benchmark-params.py:
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

# Keep the numeric libraries single-threaded so timings reflect the code
# under test rather than however many cores the runner happened to give
# us.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# asv installs each matrix requirement with its own separate pip
# invocation; prefer an existing wheel over a much slower source build if
# a package's very latest release doesn't have full wheel coverage yet.
export PIP_PREFER_BINARY=1

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
