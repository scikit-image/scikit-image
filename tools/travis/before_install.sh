#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

# This URL is for any extra wheels that are not available on pypi.  As of 14
# Jan 2017, the major packages such as numpy and matplotlib are up for all
# platforms.  The URL points to a Rackspace CDN belonging to the scikit-learn
# team.  Please contact Olivier Grisel or Matthew Brett if you need
# permissions for this folder.
EXTRA_WHEELS="https://5cf40426d9f06eb7461d-6fe47d9331aba7cd62fc36c7196769e4.ssl.cf2.rackcdn.com"
WHEELHOUSE="--find-links=$EXTRA_WHEELS"

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
# This causes way too many internal warnings within python.
# export PYTHONWARNINGS="d,all:::skimage"
export TEST_ARGS="--doctest-modules"

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
pip install --retries 3 -q wheel

# install specific wheels from wheelhouse
for requirement in matplotlib scipy pillow; do
    WHEELS="$WHEELS $(grep $requirement requirements/default.txt)"
done
# cython is not in the default.txt requirements
WHEELS="$WHEELS $(grep -i cython requirements/build.txt)"
pip install --retries 3 -q $PIP_FLAGS $WHEELHOUSE $WHEELS

# Install build time requirements
pip install --retries 3 -q $PIP_FLAGS -r requirements/build.txt
# Default requirements are necessary to build because of lazy importing
# They can be moved after the build step if #3158 is accepted
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
