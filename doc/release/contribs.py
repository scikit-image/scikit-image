#!/usr/bin/env python
from __future__ import print_function
import subprocess
import sys
import string
import shlex

if sys.version_info[0] < 3:
    from urllib import urlopen, urlencode
else:
    from urllib.request import urlopen
    from urllib.parse import urlencode

USER = 'scikit-image'
REPO = 'scikit-image'

if len(sys.argv) != 2:
    print("Usage: ./contribs.py tag-of-previous-release")
    sys.exit(-1)

tag = sys.argv[1]


def call(cmd):
    return subprocess.check_output(shlex.split(cmd), universal_newlines=True).split('\n')


# See https://git-scm.com/docs/pretty-formats - '%cI' is strict ISO-8601 format
tag_date = call("git log -n1 --format='%%cI' %s" % tag)[0]
print("Release %s was on %s\n" % (tag, tag_date))

# merges = call("git log --since='%s' --merges --format='>>>%%B' --reverse" % tag_date)
# merges = [m for m in merges if m.strip()]
# merges = '\n'.join(merges).split('>>>')
# merges = [m.split('\n')[:2] for m in merges]
# merges = [m for m in merges if len(m) == 2 and m[1].strip()]

num_commits = call("git rev-list %s..HEAD --count" % tag)[0]
print("A total of %s changes have been committed.\n" % num_commits)

# See https://developer.github.com/v3/search/#search-issues
# See https://help.github.com/articles/understanding-the-search-syntax/#query-for-dates
query_string = ('user:' + USER + '+'
                + 'repo:' + REPO + '+'
                + 'merged:>=' + tag_date)
merges_url = ('https://api.github.com/search/issues?'
              + urlencode(dict(q=query_string)))
merges = urlopen(merges_url)
merges = merges['items']
PRs = []

print("It contained the following %d merged pull requests:\n" % len(merges))
for merge in merges:
    PR = merge['number']
    title = merge['title']
    PRs += PR

    print('- ' + title + PR)

print("\nMade by the following committers [alphabetical by last name]:\n")

authors = call("git log --since='%s' --format=%%aN" % tag_date)
authors = [a.strip() for a in authors if a.strip()]


def key(author):
    author = [v for v in author.split() if v[0] in string.ascii_letters]
    if len(author) > 0:
        return author[-1]


authors = sorted(set(authors), key=key)

for a in authors:
    print('- ' + a)
