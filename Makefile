.PHONY: all clean test
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" | xargs rm -f
	find . -name "*.pyx" -exec ./tools/rm_pyx_assoc_c_cpp.sh {} \;
	rm -f MANIFEST

cleandoc:
	rm -rf doc/build

test:
	$(PYTEST) skimage --doctest-modules

doctest:
	$(PYTHON) -c "import skimage, sys, io; sys.exit(skimage.doctest_verbose())"

benchmark_coverage:
	$(PYTEST) benchmarks --cov=skimage --cov-config=setup.cfg

coverage: test_coverage

test_coverage:
	$(PYTEST) -o python_functions=test_* skimage --cov=skimage

html:
	pip install -q -r requirements/docs.txt
	export SPHINXOPTS=-W; make -C doc html
