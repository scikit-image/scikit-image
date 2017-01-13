#!/usr/bin/env bash
set -ex


export COVERALLS_REPO_TOKEN=7LdFN9232ZbSY3oaXHbQIzLazrSf6w2pQ
export PIP_DEFAULT_TIMEOUT=60

# This URL is for any extra wheels that are not available on pypi.  As of 14
# Jan 2017, the major packages such as numpy and matplotlib are up for all
# platforms.  The URL comes points ot a Rackspace CDN belonging to the
# scikit-learn team.  Please contact Olivier Grisel or Matthew Brett if you
# need permissions for this folder.
EXTRA_WHEELS="https://5cf40426d9f06eb7461d-6fe47d9331aba7cd62fc36c7196769e4.ssl.cf2.rackcdn.com"
export WHEELHOUSE="--find-links=$EXTRA_WHEELS"

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    sh -e /etc/init.d/xvfb start
fi

export DISPLAY=:99.0
export PYTHONWARNINGS="d,all:::skimage"
export TEST_ARGS="--exe --ignore-files=^_test -v --with-doctest \
                  --ignore-files=^setup.py$"
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

# add build dependencies
echo "cython>=0.23.4" >> requirements.txt
echo "numpydoc>=0.6" >> requirements.txt

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    sed -i 's/>=/==/g' requirements.txt
fi

# create new empty venv
virtualenv -p python ~/venv
source ~/venv/bin/activate

python -m pip install --upgrade pip
pip install --retries 3 -q wheel flake8 codecov nose
# install numpy from PyPI instead of our wheelhouse
pip install --retries 3 -q wheel numpy

# install wheels
for requirement in $WHEELBINARIES; do
    WHEELS="$WHEELS $(grep $requirement requirements.txt)"
done
pip install --retries 3 -q $PIP_FLAGS $WHEELHOUSE $WHEELS

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
