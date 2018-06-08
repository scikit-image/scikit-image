#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

export DISPLAY=:99.0
export PYTHONWARNINGS="d,all:::skimage"
export TEST_ARGS="-v --doctest-modules"
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
    sed -i 's/>=/==/g' requirements/build.txt
    sed -i 's/>=/==/g' requirements/default.txt
    sed -i 's/>=/==/g' requirements/docs.txt
    sed -i 's/>=/==/g' requirements/optional.txt
    sed -i 's/>=/==/g' requirements/test.txt
fi

python -m pip install --upgrade pip

pip install --retries 3 -q $PIP_FLAGS -r requirements/build.txt

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
