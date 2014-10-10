travis_retry pip install wheel flake8 coveralls nose

# on Python 2.7, use the system versions of numpy, scipy, and matplotlib
# and the minimum version of cython and networkx
if [[  $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    travis_retry sudo apt-get install python-scipy python-matplotlib
    pip install https://github.com/cython/cython/archive/0.19.2.tar.gz
    pip install https://github.com/networkx/networkx/archive/networkx-1.8.tar.gz
fi

travis_retry pip install  -r requirements.txt $WHEELHOUSE
python check_bento_build.py
