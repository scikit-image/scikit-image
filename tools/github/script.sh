#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -evx

python -m pip install $PIP_FLAGS -r requirements/test.txt
export MPL_DIR=`python -c 'import matplotlib; print(matplotlib.get_configdir())'`
mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc

TEST_ARGS="--doctest-modules --cov=skimage --showlocals"

if [[ ${WITHOUT_POOCH} == "1" ]]; then
  # remove pooch (previously installed via requirements/test.txt)
  python -m pip uninstall pooch -y
fi
if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
    python -m pip install $PIP_FLAGS -r ./requirements/optional.txt
fi

python -m pip list

(cd .. && pytest $TEST_ARGS --pyargs skimage)


if [[ "${BUILD_DOCS}" == "1" ]] || [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo Build or run examples
  python -m pip install $PIP_FLAGS -r ./requirements/docs.txt
  python -m pip list
  echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi
if [[ "${BUILD_DOCS}" == "1" ]]; then
  echo Build docs
  export SPHINXCACHE=${HOME}/.cache/sphinx; make -C doc html
elif [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo Test examples
  for f in doc/examples/*/*.py; do
    python "${f}"
    if [ $? -ne 0 ]; then
      exit 1
    fi
  done
fi

set +ev
