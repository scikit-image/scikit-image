#/usr/bin/env bash
set -ex

# It is really annoying if flake8 causes all your tests to fail
# but in the event that you have a typo, flake8 is pretty good at
# catching it.
# Therefore, it is best to run flake8 first, on a single build
# with the oldest version of python we support
section "Flake8.Test"
if [[ $RUN_FLAKE8 == "1" ]]; then
    pip install flake8
    flake8
fi
section_end "Flake8.Test"


set +ex
