#!/usr/bin/env python
import subprocess
import os
import sys
import argparse
from collections import OrderedDict

import string
import shlex
import json

from datetime import datetime
import math
import time

from warnings import warn

from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError


GH_USER = 'scikit-image'
GH_REPO = 'scikit-image'

GH_TOKEN = os.environ.get('GH_TOKEN')


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
        if e.hdrs.get('X-RateLimit-Remaining') == 0:
            if GH_TOKEN:
                # wait until can try again
                reset = datetime.fromtimestamp(e.hdrs['X-RateLimit-Reset'])
                time_left = reset - datetime.today()
                time_left = math.ceil(time_left.total_seconds())
                time.sleep(time_left)
                request(url, query=query)
            else:
                warn("API rate limit exceeded while no 'GH_TOKEN' set.")
                sys.exit(0)
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
    query = 'repo:{user}/{repo} merged:>{date} sort:created-asc'
    query = query.format(user=user, repo=repo, date=date)
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
    url = ('https://api.github.com/repos/{}/{}/pulls/{}/reviews'
           .format(user, repo, pull))
    reviews = request(url)
    return reviews


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('tag', nargs='?', default='latest',
                   help='Tag of the previous release')
    p.add_argument('version', nargs='?', default='0.14.1',
                   help='Version name of the current release')
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tag = args.tag
    version = args.version

    if GH_TOKEN is None:
        print("It is recommended that the environment variable `GH_TOKEN` "
              "be set to avoid running into problems with rate limiting. "
              "One can be acquired at https://github.com/settings/tokens.\n\n")

    if tag == 'latest':
        tag = call('git tag -l v*.*.* --sort="-version:refname"')[0]
        if tag == '':
            tag = call('git rev-list HEAD')[-1]

    # See https://git-scm.com/docs/pretty-formats - '%cI' is strict ISO-8601 format
    tag_date = call("git log -n1 --format='%cI' {}".format(tag))[0]
    num_commits = call("git rev-list {}..HEAD --count".format(tag))[0]
    committers = call("git log --since='{}' --format=%aN".format(tag_date))
    committers = {c.strip() for c in committers if c.strip()}

    print('got here')
    merges = get_merged_pulls(GH_USER, GH_REPO, tag_date)
    print('got here')
    num_merges = merges['total_count']

    reviewers = set()
    authors = set()
    users = dict()  # keep track of known usernames

    pulls = merges['items']
    for pull in pulls:
        id = pull['number']
        title = pull['title']

        try:
            author = pull['user']['login']
            if author not in users:
                name = get_user(author).get('name')
                if name is None:
                    name = author
                elif author in committers:
                    committers.discard(author)
                    committers.add(name)
                users[author] = name
            else:
                name = users[author]
            authors.add(name)
        except KeyError:
            author = None

        reviews = get_reviews(GH_USER, GH_REPO, id)
        for review in reviews:
            try:
                reviewer = review['user']['login']
                if author == reviewer:  # author reviewing own PR
                    continue
                if reviewer not in users:
                    name = get_user(reviewer).get('name')
                    if name is None:
                        name = reviewer
                    elif reviewer in committers:
                        committers.discard(reviewer)
                        committers.add(name)
                    users[reviewer] = name
                else:
                    name = users[reviewer]
                reviewers.add(name)
            except KeyError:
                pass

    def name_sorting_key(name):
        name = [v for v in name.split() if v[0] in string.ascii_letters]
        if len(name) > 0:
            return name[-1]



    highlights = OrderedDict()

    highlights['New Feature'] = []
    highlights['Improvements'] = []
    highlights['Bugfixes'] = []
    highlights['API Change'] = []
    highlights['Deprecations'] = []
    highlights['Build Tool'] = []
    highlights['Other Pull Request'] = []
    for pull in pulls:
        for key, key_list in highlights.items():
            pr_title_prefix = key + ': '
            if key == 'Other Pull Request':
                key_list.append(pull)
                break
            elif pull['title'].lower().startswith(pr_title_prefix):
                pull['title'] = pull['title'][len(pr_title_prefix):]
                key_list.append(pull)
                break

    announcement_title = "Announcement: scikit-image {}".format(version)
    print(announcement_title)
    print("="*len(announcement_title))

    print("""
We're happy to announce the release of scikit-image v{version}!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.
{python27_warning}
For more information, examples, and documentation, please visit our website:

http://scikit-image.org

This release contains {num_changes} changes.
""".format(version=version, num_changes=num_commits,
           python27_warning="""
This is the last major release with official support for Python 2.7. Future
releases will be developed using Python 3-only syntax.

However, 0.14 is a long-term support (LTS) release and will receive bug fixes
and backported features deemed important (by community demand) until January
1st 2020 (end of maintenance for Python 2.7; see PEP 373 for details).
""" if version.startswith('0.14') else "")
    )

    for key, key_list in highlights.items():
        if not key_list:
            continue
        print("""
{key}
{key_underline}
""".format(key=key+"s", key_underline='-'*(len(key)+1))
    )
        for pull in key_list:
            print('- {} (#{})'.format(pull['title'], pull['number']))

    print()
    commiter_str = (
        '{} committers added to this release [alphabetical by last name]'
        .format(len(committers)))
    print(commiter_str)
    print('-'*len(commiter_str))
    for c in sorted(committers, key=name_sorting_key):
        print('- {}'.format(c))
    print()

    authored_str = (
        '{} authors conttributed to this release [alphabetical by last name]'
        .format(len(authors))
    )
    print(authored_str)
    print('-'*len(authored_str))
    for a in sorted(authors, key=name_sorting_key):
        print('- {}'.format(a))
    print()

    reviewer_str = (
        '{} reviewers conttributed to this release [alphabetical by last name]'
        .format(len(authors))
    )
    print(reviewer_str)
    print('-'*len(reviewer_str))
    for r in sorted(reviewers, key=name_sorting_key):
        print('- {}'.format(r))
    print()


    print("I don't think the following is correct.")
    print('- Release {} was on {}'.format(tag, tag_date))
