#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -evx

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc


python -m pip list
tools/build_versions.py

TEST_ARGS="--doctest-modules --cov=skimage"

if [[ ${WITHOUT_POOCH} == "1" ]]; then
  # remove pooch (previously installed via requirements/test.txt)
  pip uninstall pooch -y
fi

# When installing from sdist
# We can't run it in the git directory since there is a folder called `skimage`
# in there. pytest will crawl that instead of the module we installed and want to test
if [[ ${INSTALL_FROM_SDIST} == "1" ]]; then
  (cd .. && python -c "import skimage; skimage.test()")
else
  (cd .. && pytest $TEST_ARGS --pyargs skimage)
fi

flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples

if [[ "${BUILD_DOCS}" == "1" ]] || [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo Build or run examples
  python -m pip install $PIP_FLAGS -r ./requirements/docs.txt
  python -m pip list
  tools/build_versions.py
  echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi
if [[ "${BUILD_DOCS}" == "1" ]]; then
  echo Build docs
  export SPHINXCACHE=${HOME}/.cache/sphinx; make html
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
