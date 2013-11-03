.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

test:
	nosetests skimage

doctest:
	nosetests \
		--with-doctest \
		--ignore-files="^\." \
		--ignore-files="^setup\.py$$" \
		skimage

coverage:
	nosetests skimage --with-coverage --cover-package=skimage
