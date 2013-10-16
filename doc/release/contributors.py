#!/usr/bin/env python
import subprocess
import sys
import string

if len(sys.argv) != 2:
    print "Usage: ./contributors.py tag-of-previous-release"
    sys.exit(-1)

tag = sys.argv[1]

cmd = "git log %s..HEAD --format=%%aN"
authors = subprocess.check_output((cmd % tag).split()).split('\n')
authors = [a.strip() for a in authors if a.strip()]

def key(author):
    author = [v for v in author.split() if v[0] in string.letters]
    return author[-1]

authors = sorted(set(authors), key=key)

for a in authors:
    print '-', a
