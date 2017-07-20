#!/usr/bin/env bash
set -x
set -e

# Decide what kind of documentation build to run, and run it.
#
# If the last commit message has a "[doc skip]" marker, do not build
# the doc. On the contrary if a "[doc build]" marker is found, build the doc
# instead of relying on the subsequent rules.
#
# We always build the documentation for jobs that are not related to a specific
# PR (e.g. a merge to master or a maintenance branch).
#
# If this is a PR, do a full build if there are some files in this PR that are
# under the "doc/" or "examples/" folders, otherwise perform a quick build.
#
# If the inspection of the current commit fails for any reason, the default
# behavior is to quick build the documentation.

get_build_type() {
	if [ -z "$CIRCLE_SHA1" ]
	then
		echo SKIP: undefined CIRCLE_SHA1
		return
	fi
	commit_msg=$(git log --format=%B -n 1 $CIRCLE_SHA1)
	if [ -z "$commit_msg" ]
	then
		echo QUICK BUILD: failed to inspect commit $CIRCLE_SHA1
		return
	fi
	if [[ "$commit_msg" =~ \[doc\ skip\] ]]
	then
		echo SKIP: [doc skip] marker found
		return
	fi
	if [[ "$commit_msg" =~ \[doc\ quick\] ]]
	then
		echo QUICK: [doc quick] marker found
		return
	fi
	if [[ "$commit_msg" =~ \[doc\ build\] ]]
	then
		echo BUILD: [doc build] marker found
		return
	fi
	if [ -z "$CI_PULL_REQUEST" ]
	then
		echo BUILD: not a pull request
		return
	fi
	git_range="origin/master...$CIRCLE_SHA1"
	git fetch origin master >&2 || (echo QUICK BUILD: failed to get changed filenames for $git_range; return)
	filenames=$(git diff --name-only $git_range)
	if [ -z "$filenames" ]
	then
		echo QUICK BUILD: no changed filenames for $git_range
		return
	fi
	if echo "$filenames" | grep -q -e ^examples/
	then
		echo BUILD: detected examples/ filename modified in $git_range: $(echo "$filenames" | grep -e ^examples/ | head -n1)
		return
	fi
	echo QUICK BUILD: no examples/ filename modified in $git_range:
	echo "$filenames"
}

build_type=$(get_build_type)
if [[ "$build_type" =~ ^SKIP ]]
then
    exit 0
fi

if [[ "$CIRCLE_BRANCH" =~ ^master$|^[0-9]+\.[0-9]+\.X$ && -z "$CI_PULL_REQUEST" ]]
then
    MAKE_TARGET=dist  # PDF linked into HTML
elif [[ "$build_type" =~ ^QUICK ]]
then
	MAKE_TARGET=html
else
    MAKE_TARGET=html
fi

# Installing required system packages to support the rendering of math
# notation in the HTML documentation
sudo -E apt-get -yq update
sudo -E apt-get -yq remove texlive-binaries --purge
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes \
    install dvipng texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
if [[ ! -f miniconda.sh ]]
then
   wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
   -O miniconda.sh
fi
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
cd ..
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n $CONDA_ENV_NAME --yes --quiet python numpy scipy \
  cython pytest coverage matplotlib sphinx pillow PyWavelets networkx
source activate testenv
pip install sphinx-gallery imread numpydoc tifffile

# Build and install scikit-image in dev mode
python setup.py develop

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $MAKE_TARGET 2>&1 | tee ~/log.txt

cd -
set +o pipefail

affected_doc_paths() {
	files=$(git diff --name-only origin/master...$CIRCLE_SHA1)
	echo "$files" | grep ^doc/.*\.rst | sed 's/^doc\/\(.*\)\.rst$/\1.html/'
	echo "$files" | grep ^examples/.*.py | sed 's/^\(.*\)\.py$/auto_\1.html/'
	skimage_files=$(echo "$files" | grep '^skimage/')
	if [ -n "$skimage_files" ]
	then
		grep -hlR -f<(echo "$skimage_files" | sed 's/^/scikit-image\/blob\/[a-z0-9]*\//') doc/_build/html/stable/modules/generated | cut -d/ -f5-
	fi
}

if [ -n "$CI_PULL_REQUEST" ]
then
	echo "The following documentation files may have been changed by PR #$CI_PULL_REQUEST:"
	affected=$(affected_doc_paths)
	echo "$affected" | sed 's|^|* http://scikit-image.org/circle?'$CIRCLE_BUILD_NUM'/|'
	(
	echo '<html><body><ul>'
	echo "$affected" | sed 's|.*|<li><a href="&">&</a></li>|'
	echo '</ul></body></html>'
	) > 'doc/_build/html/stable/_changed.html'
fi
