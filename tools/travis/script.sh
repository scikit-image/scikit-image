#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ex
export PY=${TRAVIS_PYTHON_VERSION}

# Matplotlib directory location changes depending on OS and verison
export MPL_DIR=`python -c 'import matplotlib; print(matplotlib.get_configdir())'`

mkdir -p $MPL_DIR
# by default, do not show figures
# this may be overwritten by subsequent install script
echo 'backend : Template' > $MPL_DIR/matplotlibrc

if [[ "${RUN_TESTS}" != "0" ]]; then
  section "Flake8.test"
  flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
  section_end "Flake8.test"

  section "Test.with.min.requirements"
  # Show what's installed
  pip list
  tools/build_versions.py
  pytest skimage
  section_end "Test.with.min.requirements"
fi

# Install optional dependencies and pyqt
section "Install.optional.dependencies"
mkdir -p ${MPL_DIR}
touch ${MPL_DIR}/matplotlibrc
# matplotlib is one optional dependency that definitely needs to be installed
# from the wheelhouse if we plan on testing linux 32 bit
pip install --retries 3 -q -r ./requirements/optional.txt $WHEELHOUSE

tools/travis/install_qt.sh
section_end "Install.optional.dependencies"

if [[ "${RUN_TESTS}" != "0" ]]; then
  section "Test.with.optional.requirements"
  # Show what's installed
  pip list
  tools/build_versions.py
  pytest --doctest-modules --cov=skimage skimage
  section_end "Test.with.optional.requirements"
fi

# Run example applications
echo Build or run examples
if [[ "${BUILD_DOCS}" == "1" ]]; then
  section "build.docs"
  pip install --retries 3 -q -r ./requirements/docs.txt
  pip list
  tools/build_versions.py
  export SPHINXCACHE=${HOME}/.cache/sphinx
  make html
  section_end "build.docs"
elif [[ "${RUN_EXAMPLES}" != "0" ]]; then
  section "Tests.examples"
  pip install --retries 3 -q -r ./requirements/docs.txt
  pip list
  tools/build_versions.py
  # These test are run by sphinx automatically
  echo 'backend : Template' > $MPL_DIR/matplotlibrc
  for f in doc/examples/*/*.py; do
    python "${f}"
  done
  section_end "Tests.examples"
fi

set +ex
