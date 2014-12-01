# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        ?= python
SPHINXOPTS    ?=
SPHINXBUILD   ?= python $(shell which sphinx-build)
PAPER         ?=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
DEST = build
.PHONY: all help clean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest gitwash gh-pages coveragetable random_gallery

all: html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  devhelp   to make HTML files and a Devhelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf  to make LaTeX files and run them through pdflatex"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"
	@echo "  gitwash   to update the gitwash documentation"
clean:
	-rm -rf $(DEST)/*
	-rm -rf source/api
	-find ./source/auto_examples/* -type f | grep -v blank | xargs rm -f
api:
	@mkdir -p source/api
	$(PYTHON) tools/build_modref_templates.py
	@echo "Build API docs...done."

random_gallery:
	@cd source && $(PYTHON) random_gallery.py

coveragetable:
	@cd source && $(PYTHON) coverage_generator.py

html: api coveragetable random_gallery
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(DEST)/html
	cp -r source/plots $(DEST)/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(DEST)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in build/dirhtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(DEST)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(DEST)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(DEST)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(DEST)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in build/qthelp, like this:"
	@echo "# qcollectiongenerator build/qthelp/scikitimage.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile build/qthelp/scikitimage.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(DEST)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/scikitimage"
	@echo "# ln -s build/devhelp $$HOME/.local/share/devhelp/scikitimage"
	@echo "# devhelp"

latex: api
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(DEST)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(DEST)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

latexpdf: latex
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(DEST)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C build/latex all-pdf
	@echo "pdflatex finished; the PDF files are in build/latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(DEST)/changes
	@echo
	@echo "The overview file is in build/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(DEST)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(DEST)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in build/doctest/output.txt."

gh-pages:
	$(PYTHON) gh-pages.py

gitwash:
	$(PYTHON) tools/gitwash/gitwash_dumper.py source scikit-image \
	--project-url=http://scikit-image.org \
	--project-ml-url=http://groups.google.com/group/scikit-image \
	--repo-name=scikit-image \
	--github-user=scikit-image \
	--source-suffix=.txt
