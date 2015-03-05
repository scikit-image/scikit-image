#!/usr/bin/env python
"""Script to commit the doc build outputs into the github-pages repo.

Use:

  gh-pages.py [tag]

If no tag is given, the current output of 'git describe' is used.  If given,
that is how the resulting directory will be named.

In practice, you should use either actual clean tags from a current build or
something like 'current' as a stable URL for the mest current version of the """

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import re
import shutil
import sys
from os import chdir as cd

from subprocess import Popen, PIPE, CalledProcessError, check_call

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

pages_dir = 'gh-pages'
html_dir = 'build/html'
pdf_dir = 'build/latex'
pages_repo = 'https://github.com/scikit-image/docs.git'

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def sh2(cmd):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x"""
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        print(out.rstrip())
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()


def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr

    If anything appears in stderr, print it out to sys.stderr"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


def init_repo(path):
    """clone the gh-pages repo if we haven't already."""
    sh("git clone %s %s"%(pages_repo, path))
    here = os.getcwd()
    cd(path)
    sh('git checkout gh-pages')
    cd(here)

#-----------------------------------------------------------------------------
# Script starts
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # find the version number from setup.py
    setup_lines = open('../setup.py').readlines()
    tag = 'vUndefined'
    for l in setup_lines:
        if l.startswith('VERSION'):
            tag = l.split("'")[1]

            if "dev" in tag:
                tag = "dev"
            else:
                # Rename e.g. 0.9.0 to 0.9.x
                tag = '.'.join(tag.split('.')[:-1] + ['x'])

            break


    startdir = os.getcwd()
    if not os.path.exists(pages_dir):
        # init the repo
        init_repo(pages_dir)
    else:
        # ensure up-to-date before operating
        cd(pages_dir)
        sh('git checkout gh-pages')
        sh('git pull')
        cd(startdir)

    dest = os.path.join(pages_dir, tag)
    # This is pretty unforgiving: we unconditionally nuke the destination
    # directory, and then copy the html tree in there
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(html_dir, dest)
    # copy pdf file into tree
    #shutil.copy(pjoin(pdf_dir, 'scikits.image.pdf'), pjoin(dest, 'scikits.image.pdf'))

    try:
        cd(pages_dir)
        status = sh2('git status | head -1')
        branch = re.match(b'On branch (.*)$', status).group(1)
        if branch != b'gh-pages':
            e = 'On %r, git branch is %r, MUST be "gh-pages"' % (pages_dir,
                                                                 branch)
            raise RuntimeError(e)
        sh("touch .nojekyll")
        sh('git add .nojekyll')
        sh('git add index.html')
        sh('git add --all %s' % tag)

        status = sh2('git status | tail -1')
        if not re.match(b'nothing to commit', status):
            sh2('git commit -m"Updated doc release: %s"' % tag)
        else:
            print('\n! Note: no changes to commit\n')

        print('Most recent commit:')
        sys.stdout.flush()
        sh('git --no-pager log --oneline HEAD~1..')
    finally:
        cd(startdir)

    print('')
    print('Now verify the build in: %r' % dest)
    print("If everything looks good, run 'git push' inside doc/gh-pages.")
