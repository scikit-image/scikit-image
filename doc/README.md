# Building docs #
To build docs, run `make` in this directory. `make help` lists all targets.

## Requirements ##
Sphinx is needed to build doc. Install with `pip install sphinx`.

## Fixing Warnings ##

- "citation not found: R###"
  $ cd doc/build; grep -rin R### .
  There is probably an underscore after the reference (e.g. [1]_)

- "Duplicate citation R###, other instance in...""
  There is probably a [2] without a [1] in one of
  the docstrings

- Make sure to use pre-sphinxification paths to images
  (not the _images directory)
