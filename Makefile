.PHONY: all clean test
PYTHON=python
NOSETESTS=nosetests

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;

test:
	$(PYTHON) -c "import skimage, sys, io; sys.exit(skimage.test_verbose())"

doctest:
	$(PYTHON) -c "import skimage, sys, io; sys.exit(skimage.doctest_verbose())"

coverage:
	$(NOSETESTS) skimage --with-coverage --cover-package=skimage

html:
	pip install -q sphinx
	export SPHINXOPTS=-W; make -C doc html
