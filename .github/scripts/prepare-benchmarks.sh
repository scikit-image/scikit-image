#!/usr/bin/env bash
# Get the runner ready for run-benchmarks.sh: export the resolved
# parameters, point asv's build step at the already-downloaded wheels, and
# install/register what asv needs.
#
# Reads benchmark-params.json (written by resolve-benchmark-params.py,
# downloaded here as an artifact) into $GITHUB_ENV, for run-benchmarks.sh
# and the benchmark processes. CI-only: $GITHUB_ENV doesn't exist outside
# Actions.

set -euo pipefail
set -x

# Some parameter values (the PR base/head labels) derive from
# user-controlled branch names, so use the heredoc form with an
# unguessable delimiter rather than KEY=VALUE: a value can't then
# terminate its own block and inject further variables.
delimiter="__BENCHMARK_PARAM_$(openssl rand -hex 16)__"
jq -r --arg d "$delimiter" 'to_entries[] | "\(.key)<<\($d)\n\(.value)\n\($d)"' \
    benchmark-params.json >> "$GITHUB_ENV"

# Keep the numeric libraries single-threaded so timings reflect the code
# under test rather than however many cores the runner happened to give us.
{
    echo "OPENBLAS_NUM_THREADS=1"
    echo "MKL_NUM_THREADS=1"
    echo "OMP_NUM_THREADS=1"
    echo "PYTHONUNBUFFERED=1"
    # asv installs each matrix requirement with its own separate pip
    # invocation; prefer an existing wheel over a much slower source build
    # if a package's very latest release doesn't have full wheel coverage
    # yet.
    echo "PIP_PREFER_BINARY=1"
} >> "$GITHUB_ENV"

# asv.conf.json's checked-in build_command does a real build, so local runs
# (spin asv -- continuous ...) work out of the box. Both commits' wheels are
# already downloaded here, so swap in a copy-only build_command instead of
# rebuilding from source a second time.
jq --arg cmd "/bin/bash -c 'cp {conf_dir}/.benchmark-wheels/{commit}/*.whl {build_cache_dir}/'" \
    '.build_command = [$cmd]' asv.conf.json > asv.conf.json.tmp
mv asv.conf.json.tmp asv.conf.json

python -m pip install virtualenv

# ID this runner
asv machine --yes
