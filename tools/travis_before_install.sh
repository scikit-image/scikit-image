#!/usr/bin/env bash
set -ex


export COVERALLS_REPO_TOKEN=7LdFN9232ZbSY3oaXHbQIzLazrSf6w2pQ
export PIP_DEFAULT_TIMEOUT=60

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then
    sh -e /etc/init.d/xvfb start
    export WHEELHOUSE="--no-index --trusted-host travis-wheels.scikit-image.org \
                       --find-links=http://travis-wheels.scikit-image.org/"
fi

export DISPLAY=:99.0
export PYTHONWARNINGS="d,all:::skimage"
export TEST_ARGS="--exe --ignore-files=^_test -v --with-doctest \
                  --ignore-files=^setup.py$"
WHEELBINARIES="matplotlib numpy scipy pillow cython"

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

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    sed -i 's/>=/==/g' requirements.txt
fi

# create new empty venv
virtualenv -p python ~/venv
source ~/venv/bin/activate

pip install --upgrade pip
pip install --retries 3 -q wheel flake8 codecov nose

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
