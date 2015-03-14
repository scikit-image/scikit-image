# Building docs #
To build docs, run `make` in this directory. `make help` lists all targets.

## Requirements ##
Sphinx and Latex is needed to build doc.

**Spinx:**
```sh
pip install sphinx
```

**Latex Ubuntu:**
```sh
sudo apt-get install -qq texlive texlive-latex-extra dvipng
```

**Latex Mac:**

Install the full [MacTex](http://www.tug.org/mactex/) installation or install the smaller [BasicTex](http://www.tug.org/mactex/morepackages.html) and add *ucs* and *dvipng* packages:
```sh
sudo tlmgr install ucs dvipng
```


## Fixing Warnings ##

- "citation not found: R###"
  $ cd doc/build; grep -rin R### .
  There is probably an underscore after a reference
  in the first line of a docstring (e.g. [1]_)

- "Duplicate citation R###, other instance in...""
  There is probably a [2] without a [1] in one of
  the docstrings

- Make sure to use pre-sphinxification paths to images
  (not the _images directory)
