#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

# This causes way too many internal warnings within python.
# export PYTHONWARNINGS="d,all:::skimage"

retry () {
    # https://gist.github.com/fungusakafungus/1026804
    local retry_max=3
    local count=$retry_max
    while [ $count -gt 0 ]; do
        "$@" && break
        count=$(($count - 1))
        sleep 1
    done

    [ $count -eq 0 ] && {
        echo "Retry failed [$retry_max]: $@" >&2
        return 1
    }
    return 0
}

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    for filename in requirements/*.txt; do
        sed -i 's/>=/==/g' $filename
    done
fi

python -m pip install --upgrade pip wheel "setuptools<=59.4"

# Install build time requirements
python -m pip install $PIP_FLAGS -r requirements/build.txt
# Default requirements are necessary to build because of lazy importing
# They can be moved after the build step if #3158 is accepted
python -m pip install $PIP_FLAGS -r requirements/default.txt

# Show what's installed
python -m pip list

section () {
    tools/header.py $1
}

export -f section
export -f retry

set +ex
