.PHONY: all clean test

all:
	python setup.py build_ext --inplace
	git update-index --assume-unchanged skimage/version.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	nosetests skimage

coverage:
	nosetests skimage --with-coverage --cover-package=skimage
