.PHONY: all clean test
PYTHON=python
PYTESTS=py.test

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_c_file.sh {} \;

test:
	$(PYTESTS) skimage --doctest-modules

doctest:
	$(PYTHON) -c "import skimage, sys, io; sys.exit(skimage.doctest_verbose())"

coverage:
	$(PYTESTS) skimage --cov=skimage

html:
	pip install -q sphinx
	export SPHINXOPTS=-W; make -C doc html
