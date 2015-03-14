.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	python -c "import skimage, sys, io; sys.exit(skimage.test_verbose())"

doctest:
	python -c "import skimage, sys, io; sys.exit(skimage.doctest_verbose())"

coverage:
	nosetests skimage --with-coverage --cover-package=skimage

html:
	pip install -q sphinx
	export SPHINXOPTS=-W; make -C doc html
