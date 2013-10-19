#!/usr/bin/env python
import subprocess
import sys
import string
import shlex

if len(sys.argv) != 2:
    print "Usage: ./contributors.py tag-of-previous-release"
    sys.exit(-1)

tag = sys.argv[1]

def call(cmd):
    return subprocess.check_output(shlex.split(cmd)).split('\n')


tag_date = call("git show --format='%%ci' %s" % tag)[0]
print "Release %s was on %s" % (tag, tag_date)
print "\nCommitters since, alphabetical by last name:\n"

authors = call("git log --since='%s' --format=%%aN" % tag_date)
authors = [a.strip() for a in authors if a.strip()]

def key(author):
    author = [v for v in author.split() if v[0] in string.letters]
    return author[-1]

authors = sorted(set(authors), key=key)

for a in authors:
    print '-', a
