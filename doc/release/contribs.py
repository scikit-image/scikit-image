#!/usr/bin/env python
from __future__ import print_function
import subprocess
import os
import sys
import string
import shlex
import json
from datetime import datetime
import math
import time

if sys.version_info[0] < 3:
    from urllib import urlopen, urlencode
    from urllib2 import HTTPError
else:
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.error import HTTPError

GH_USER = 'scikit-image'
GH_REPO = 'scikit-image'

GH_TOKEN = os.environ.get('GH_TOKEN')

if len(sys.argv) != 2:
    print("Usage: ./contribs.py [--latest|tag-of-previous-release]")
    sys.exit(-1)

tag = sys.argv[1]


def call(cmd):
    return subprocess.check_output(shlex.split(cmd),
                                   universal_newlines=True).split('\n')


def request(url, query=None):
    query_string = ''
    if query:
        query_string += urlencode(query)
    if GH_TOKEN:
        if query_string:
            query_string += '&'
        query_string += 'access_token=' + GH_TOKEN
    if query_string:
        query_string = '?' + query_string

    try:
        response = urlopen(url + query_string).read()
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        return json.loads(response)
    except HTTPError as e:
        if GH_TOKEN and e.hdrs.get('X-RateLimit-Remaining') == 0:
            # wait until can try again
            reset = datetime.fromtimestamp(e.hdrs['X-RateLimit-Reset'])
            time_left = reset - datetime.today()
            time_left = math.ceil(time_left.total_seconds())
            time.sleep(time_left)
            request(url, query=query)
        else:
            raise Exception(e.info)


def get_user(login):
    # See https://developer.github.com/v3/users/#get-a-single-user
    url = 'https://api.github.com/users/' + login
    user = request(url)
    return user


def get_merged_pulls(user, repo, date, page=1, results=0):
    # See https://developer.github.com/v3/search/#search-issues
    # See https://help.github.com/articles/understanding-the-search-syntax/#query-for-dates
    query = 'user:%s repo:%s merged:>=%s' % (user, repo, date)
    url = 'https://api.github.com/search/issues'
    merges = request(url, query=dict(q=query, page=page, per_page=100))

    count = len(merges['items'])

    if results < merges['total_count']:
        merges['items'] += get_merged_pulls(user, repo, date,
                                            page=page + 1,
                                            results=results + count)['items']

    return merges


def get_reviews(user, repo, pull):
    # See https://developer.github.com/v3/pulls/reviews/
    url = 'https://api.github.com/repos/%s/%s/pulls/%s/reviews'
    url %= user, repo, pull
    reviews = request(url)
    return reviews


if tag == '--latest':
    tag = call('git tag -l v*.*.* --sort="-version:refname"')[0]
    if tag == '':
        tag = call('git rev-list HEAD')[-1]

# See https://git-scm.com/docs/pretty-formats - '%cI' is strict ISO-8601 format
tag_date = call("git log -n1 --format='%%cI' %s" % tag)[0]
num_commits = call("git rev-list %s..HEAD --count" % tag)[0]
authors = call("git log --since='%s' --format=%%aN" % tag_date)
authors = {a.strip() for a in authors if a.strip()}


merges = get_merged_pulls(GH_USER, GH_REPO, tag_date)
num_merges = merges['total_count']

pulls = merges['items']
reviewers = set()
users = dict()  # keep track of known usernames
for pull in pulls:
    id = pull['number']
    title = pull['title']

    try:
        author = pull['user']['login']
        if author not in users:
            name = get_user(author).get('name')
            if name is None:
                name = author
            elif author in authors:
                authors.discard(author)
                authors.add(name)
            users[author] = name
    except KeyError:
        pass

    reviews = get_reviews(GH_USER, GH_REPO, id)
    for review in reviews:
        try:
            reviewer = review['user']['login']
            if reviewer not in users:
                name = get_user(reviewer).get('name')
                if name is None:
                    name = handle
                elif reviewer in authors:
                    authors.discard(reviewer)
                    authors.add(name)
                users[reviewer] = name
            else:
                name = users[reviewer]
            reviewers.add(name)
        except KeyError:
            pass


def key(name):
    name = [v for v in name.split() if v[0] in string.ascii_letters]
    if len(name) > 0:
        return name[-1]


print("Release %s was on %s\n" % (tag, tag_date))
print("A total of %s changes have been committed.\n" % num_commits)

print("It contained the following %d merged pull requests:" % num_merges)
for pull in pulls:
    print('- %s : %s' % (pull['number'], pull['title']))
print()

print("Made by the following committers [alphabetical by last name]:")
for a in sorted(authors, key=key):
    print('- %s' % a)
print()

print('Reviewed by the following reviewers [alphabetical by last name]:')
for r in sorted(reviewers, key=key):
    print('- %s' % r)
