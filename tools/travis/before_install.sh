#!/usr/bin/env bash
set -ev

export PIP_DEFAULT_TIMEOUT=60

export DISPLAY=:99.0
# This causes way too many internal warnings within python.
# export PYTHONWARNINGS="d,all:::skimage"
export TEST_ARGS="--doctest-modules --cov=skimage"
WHEELBINARIES="matplotlib scipy pillow cython"

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

python -m pip install --upgrade pip
pip install --retries 3 -q $PIP_FLAGS -r requirements/build.txt
# The line below isn't necessary if #3158 is accepted.
# That said, `pip install .` also install runtime requirements before
# the build process starts. This line helps with the strange lazy loading
# necessary to build the sdist packages.
pip install --retries 3 -q $PIP_FLAGS -r requirements/default.txt

# Show what's installed
pip list

section () {
    echo -en "travis_fold:start:$1\r"
    tools/header.py $1
}

section_end () {
    echo -en "travis_fold:end:$1\r"
}

export -f section
export -f section_end
export -f retry

set +ex
