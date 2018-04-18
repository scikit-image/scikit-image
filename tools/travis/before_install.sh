#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    sh -e /etc/init.d/xvfb start
    # This one is for wheels we can only build on the travis precise container.
    # As of 14 Jan 2017, this is only pyside.  Also on Rackspace, see above.
    # To build new wheels for this container, consider using:
    # https://github.com/matthew-brett/travis-wheel-builder . The wheels from
    # that building repo upload to the container "travis-wheels" available at
    # https://8167b5c3a2af93a0a9fb-13c6eee0d707a05fa610c311eec04c66.ssl.cf2.rackcdn.com
    # You then need to transfer them to the container pointed to by the URL
    # below (called "precise-wheels" on the Rackspace interface).
    PRECISE_WHEELS="https://7d8d0debcc2964ae0517-cec8b1780d3c0de237cc726d565607b4.ssl.cf2.rackcdn.com"
    WHEELHOUSE="--find-links=$PRECISE_WHEELS $WHEELHOUSE"
fi
export WHEELHOUSE

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
    sed -i 's/>=/==/g' requirements/default.txt
fi

python -m pip install --upgrade pip
pip install --retries 3 -q flake8
pip install --retries 3 -q $PIP_FLAGS -r requirements.txt

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
