#!/usr/bin/env bash
set -ex

export WHEELHOUSE="--no-index --trusted-host travis-wheels.scikit-image.org \
                   --find-links=http://travis-wheels.scikit-image.org/"
export COVERALLS_REPO_TOKEN=7LdFN9232ZbSY3oaXHbQIzLazrSf6w2pQ
export PIP_DEFAULT_TIMEOUT=60
sh -e /etc/init.d/xvfb start
export DISPLAY=:99.0
export PYTHONWARNINGS="all"
export TEST_ARGS="--exe --ignore-files=^_test -v --with-doctest \
                  --ignore-files=^setup.py$"
export WHEELBINARIES="matplotlib numpy scipy pillow cython"


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


# test with minimum requirements on 2.7
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sed -i 's/>=/==/g' requirements.txt
    # PIL instead of Pillow
    sed -i '/pillow/d' requirements.txt
    export WHEELBINARIES=${WHEELBINARIES/pillow/pil}
fi

# create new empty venv
virtualenv -p python ~/venv
source ~/venv/bin/activate

retry pip install wheel flake8 coveralls nose

# install system tk for matplotlib
sudo apt-get install python-tk


# on Python 3.2, use matplotlib 1.3.1
if [[ $TRAVIS_PYTHON_VERSION == 3.2 ]]; then
    sed -i 's/matplotlib>=.*/matplotlib==1.3.1/g' requirements.txt
fi

retry pip install $WHEELHOUSE $WHEELBINARIES
retry pip install -r requirements.txt

# clean up disk space
sudo apt-get clean
sudo rm -rf /tmp/*


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
