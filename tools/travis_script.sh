#!/usr/bin/env bash
set -ex

PY=$TRAVIS_PYTHON_VERSION

section "Test.with.min.requirements"
nosetests $TEST_ARGS skimage
section_end "Test.with.min.requirements"

section "Build.docs"
if [[ ($PY != 2.6) && ($PY != 3.2) ]]; then
    make html
fi
section_end "Build.docs"

section "Flake8.test"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples
section_end "Flake8.test"


section "Install.optional.dependencies"

# Install Qt and then update the Matplotlib settings
retry pip install -q PySide $WHEELHOUSE
python ~/venv/bin/pyside_postinstall.py -install

# Install imread from wheelhouse if available (not 3.2)
if [[ $PY != 3.2 ]]; then
    retry pip install -q $WHEELHOUSE
fi

# Install SimpleITK from wheelhouse if available (not 3.2 or 3.4)
if [[ $PY =~ 3\.[24] ]]; then
    echo "SimpleITK unavailable on $PY"
else
    retry pip  install -q SimpleITK $WHEELHOUSE
fi

retry pip install -q astropy $WHEELHOUSE

if [[ $PY == 2.* ]]; then
    retry pip install -q pyamg
fi

retry pip install -q tifffile

section_end "Install.optional.dependencies"


section "Run.doc.examples"

# Matplotlib settings - do not show figures during doc examples
if [[ $PY == 2.7* ]]; then
    MPL_DIR=$HOME/.matplotlib
else
    MPL_DIR=$HOME/.config/matplotlib
fi

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc
echo 'backend : Template' > $MPL_DIR/matplotlibrc


for f in doc/examples/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

section_end "Run.doc.examples"


section "Run.doc.applications"

for f in doc/examples/applications/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

# Now configure Matplotlib to use Qt4
if [[ $PY == 2.7* ]]; then
    MPL_QT_API=PyQt4
    export QT_API=pyqt
else
    MPL_QT_API=PySide
    export QT_API=pyside
fi
echo 'backend: Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc

section_end "Run.doc.applications"


section "Test.with.optional.dependencies"

# run tests again with optional dependencies to get more coverage
if [[ $PY == 3.3 ]]; then
    TEST_ARGS="$TEST_ARGS --with-cov --cover-package skimage"
fi
nosetests $TEST_ARGS

section_end "Test.with.optional.dependencies"
