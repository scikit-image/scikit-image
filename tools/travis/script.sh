#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev
export PY=${TRAVIS_PYTHON_VERSION}

# Matplotlib directory location changes depending on OS and verison
MPL_DIR=`python -c 'import matplotlib; print(matplotlib.get_configdir())'`

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc

# Matplotlib settings - do not show figures during doc examples
if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi

section "Flake8.test"
flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
section_end "Flake8.test"

section "Test.with.min.requirements"
# Show what's installed
pip list
tools/build_versions.py
pytest skimage
section_end "Test.with.min.requirements"

# Install optional dependencies and pyqt
section "Install.optional.dependencies"
mkdir -p ${MPL_DIR}
touch ${MPL_DIR}/matplotlibrc
# matplotlib is one optional dependency that definitely needs to be installed
# from the wheelhouse if we plan on testing linux 32 bit
pip install --retries 3 -q -r ./requirements/optional.txt $WHEELHOUSE

tools/travis/install_qt.sh
section_end "Install.optional.dependencies"

section "Test.with.optional.requirements"
# Show what's installed
pip list
tools/build_versions.py
pytest --doctest-modules --cov=skimage skimage
section_end "Test.with.optional.requirements"

section "Tests.examples"
# Run example applications
echo Build or run examples
pip install --retries 3 -q -r ./requirements/docs.txt
pip list
tools/build_versions.py

if [[ "${BUILD_DOCS}" == "1" ]]; then
  export SPHINXCACHE=${HOME}/.cache/sphinx
  make html
else
  # These test are run by sphinx automatically
  cp $MPL_DIR/matplotlibrc $MPL_DIR/matplotlibrc_backup
  echo 'backend : Template' > $MPL_DIR/matplotlibrc
  for f in doc/examples/*/*.py; do
    python "${f}"
  done
  mv $MPL_DIR/matplotlibrc_backup $MPL_DIR/matplotlibrc
fi
section_end "Tests.examples"

set +ev
