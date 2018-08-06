.PHONY: all clean test
PYTHON=python
PYTESTS=pytest
BENCHMARK=benchmark

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
ifeq ($(suite),$(BENCHMARK))
	$(PYTESTS) benchmarks --cov=skimage
else
	$(PYTESTS) -o python_functions=test_* skimage --cov=skimage
endif

html:
	pip install -q sphinx pytest-runner sphinx-gallery
	export SPHINXOPTS=-W; make -C doc html

