#!/usr/bin/env bash
set -ex

# Matplotlib settings - do not show figures during doc examples
mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc
echo 'backend : Template' > $MPL_DIR/matplotlibrc


tools/header.py "Run doc examples"
for f in doc/examples/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

for f in doc/examples/applications/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

# Now configure Matplotlib to use Qt4
echo 'backend: Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc


tools/header.py "Run tests with all dependencies"
# run tests again with optional dependencies to get more coverage
if [[ $TRAVIS_PYTHON_VERSION == 3.3 ]]; then
    export TEST_ARGS="--with-cov --cover-package skimage"
else
    export TEST_ARGS=""
fi
nosetests --exe -v --with-doctest $TEST_ARGS

