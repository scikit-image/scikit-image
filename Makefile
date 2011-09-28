.PHONY: all clean test

all:
	python setup.py build_ext --inplace
	git update-index --assume-unchanged scikits/image/version.py

clean:
	find . -name "*.so" -o -name "*.pyc" | xargs rm -f

test:
	nosetests scikits/image

coverage:
	nosetests scikits/image --with-coverage --cover-package=scikits.image
