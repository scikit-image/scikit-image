#!/usr/bin/env bash
# Test pytest and build docs in CI workflows on GitHub
#
# This script is not really intended to run locally outside of GitHub's CI
# environment.

# Fail on non-zero exit and echo the commands
set -evx

# Ensure that pytest picks up its configuration in pyproject.toml even if
# we shift directory
if [ -z "${GITHUB_WORKSPACE}" ]; then
  PYTEST_CONFIG_FILE_OPTION="--config-file ${GITHUB_WORKSPACE}/pyproject.toml"
fi

TEST_ARGS="--doctest-plus --cov=skimage --showlocals ${PYTEST_CONFIG_FILE_OPTION}"


# Combine requirement files for a more robust pip solve
# installing successively may update previously constrained dependencies
REQUIREMENT_FILES="-r requirements/default.txt -r requirements/test.txt"
if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
    REQUIREMENT_FILES="${REQUIREMENT_FILES} -r requirements/optional.txt"
fi

python -m pip install $PIP_FLAGS $REQUIREMENT_FILES

if [[ ${WITHOUT_POOCH} == "1" ]]; then
  # remove pooch (previously installed via requirements/test.txt)
  python -m pip uninstall pooch -y
fi

python -m pip list


# Run the tests, step out of project dir to make sure that installed version is tested
(cd .. && pytest $TEST_ARGS --pyargs skimage)


# Optionally, prepare building the docs (and test examples)
if [[ "${BUILD_DOCS}" == "1" ]] || [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo "Build or run examples"
  # Use previous installed requirements as well, otherwise installing
  # successively may update previously constraint dependencies
  python -m pip install $PIP_FLAGS $REQUIREMENT_FILES -r ./requirements/docs.txt
  python -m pip list

  export MPL_DIR
  MPL_DIR=$(python -c 'import matplotlib; print(matplotlib.get_configdir())')
  if [[ -n "${MPL_DIR}" ]]; then
    mkdir -p "${MPL_DIR}"
    touch "${MPL_DIR}/matplotlibrc"
    echo 'backend : Template' > "${MPL_DIR}/matplotlibrc"
  fi
fi

# Optionally, build the docs or test examples
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
