#!/usr/bin/env python
from __future__ import print_function
import subprocess
import sys
import os
import string
import shlex
import json

if sys.version_info[0] < 3:
    from urllib import urlopen, urlencode
    from urllib2 import HTTPError
else:
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.error import HTTPError

USER = 'scikit-image'
REPO = 'scikit-image'
GH_TOKEN = os.environ.get('GH_TOKEN')
if GH_TOKEN is None:
    print('No GH_TOKEN found. API may be timed out.')

if len(sys.argv) != 2:
    print("Usage: ./contribs.py tag-of-previous-release")
    sys.exit(-1)

tag = sys.argv[1]


def call(cmd):
    return subprocess.check_output(shlex.split(cmd),
                                   universal_newlines=True).split('\n')


def req(url):
    try:
        response = urlopen(url).read()
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        return json.loads(response)
    except HTTPError:
        print('=== API limit rate exceeded ===')
    return None


def get_user_name(login):
    # See https://developer.github.com/v3/users/#get-a-single-user
    url = 'https://api.github.com/users/' + login
    if GH_TOKEN is not None:
        url += '?access_token=' + GH_TOKEN
    user = req(url)
    if user is None:
        return None
    return user.get('name')


# See https://git-scm.com/docs/pretty-formats - '%cI' is strict ISO-8601 format
tag_date = call("git log -n1 --format='%%cI' %s" % tag)[0]
print("Release %s was on %s\n" % (tag, tag_date))

num_commits = call("git rev-list %s..HEAD --count" % tag)[0]
print("A total of %s changes have been committed.\n" % num_commits)

authors = call("git log --since='%s' --format=%%aN" % tag_date)
authors = [a.strip() for a in authors if a.strip()]

# See https://developer.github.com/v3/search/#search-issues
# See https://help.github.com/articles/understanding-the-search-syntax/#query-for-dates
query_string = ('user:' + USER + ' '
                + 'repo:' + REPO + ' '
                + 'merged:>=' + tag_date)
merges_url = ('https://api.github.com/search/issues?'
              + urlencode(dict(q=query_string)))
if GH_TOKEN is not None:
    merges_url += '&access_token=' + GH_TOKEN
merges = req(merges_url)
PRs = []

if merges is not None:
    merges = merges.get('items')
    print("It contained the following %d merged pull requests:"
          % len(merges))
    for merge in merges:
        PR = str(merge.get('number'))
        title = merge.get('title')
        PRs += [PR]

        author = merge.get('user')
        if author is not None:
            author = author.get('login')
            if author is not None:
                name = get_user_name(author)
                if name is not None:
                    author = name
                if author not in authors:
                    authors += [author]

        print('- ' + PR + ' : ' + title)

print("\nMade by the following committers [alphabetical by last name]:")


def key(name):
    name = [v for v in name.split() if v[0] in string.ascii_letters]
    if len(name) > 0:
        return name[-1]


authors = sorted(set(authors), key=key)

for a in authors:
    print('- ' + a)


# See https://developer.github.com/v3/pulls/reviews/
pr_url = ('https://api.github.com/repos/'
          + USER + '/'
          + REPO
          + '/pulls/%s/reviews')
if GH_TOKEN is not None:
    pr_url += '?access_token=' + GH_TOKEN
reviewers = {}

for PR in PRs:
    req_url = pr_url % PR
    reviews = req(req_url)

    if reviews is None:
        continue
    for review in reviews:
        if not isinstance(review, dict):
            continue
        reviewer = review.get('user')
        if reviewer is not None:
            handle = reviewer.get('login')
        reviewer = get_user_name(handle)

        if reviewer is None:
            reviewer = handle

        if handle is not None:
            reviewers[handle] = reviewer

reviewers = reviewers.values()

if reviewers == []:
    quit()

print('\nReviewed by the following reviewers [alphabetical by last name]:')

reviewers = sorted(set(reviewers), key=key)

for r in reviewers:
    print('- ' + r)
