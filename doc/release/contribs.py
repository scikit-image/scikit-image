#!/usr/bin/env python
import subprocess
import os
import sys
import argparse

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


def get_merged_pulls(user, repo, date, page=1, results=0, branch='master'):
    # See https://developer.github.com/v3/search/#search-issues
    # See https://help.github.com/articles/understanding-the-search-syntax/#query-for-dates
    query = f'repo:{user}/{repo} merged:>{date} sort:created-asc base:{branch}'
    url = 'https://api.github.com/search/issues'
    merges = request(url, query=dict(q=query, page=page, per_page=100))

    count = len(merges['items'])

    if results < merges['total_count']:
        merges['items'] += get_merged_pulls(user, repo, date,
                                            page=page + 1,
                                            results=results + count,
                                            branch=branch)['items']

    return merges


def get_reviews(user, repo, pull):
    # See https://developer.github.com/v3/pulls/reviews/
    url = f'https://api.github.com/repos/{user}/{repo}/pulls/{pull}/reviews'
    reviews = request(url)
    return reviews


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('last_release_tag', help='Tag of the previous release; can be `last`')
    p.add_argument('--dev-branch', default='master',
                   help='Name of the branch for this release against which PRs '
                        'were made.')
    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    last_release_tag = args.last_release_tag
    dev_branch = args.dev_branch

    if GH_TOKEN is None:
        print("! It is recommended that the environment variable `GH_TOKEN` \n"
              "  be set to avoid running into problems with rate limiting.\n\n"
              "  One can be acquired at https://github.com/settings/tokens.\n\n"
              "  You do not need to select any permission boxes while generating"
              "  the token.\n\n")

    if last_release_tag == 'last':
        last_release_tag = call('git tag -l v*.*.* --sort="-version:refname"')[0]
        if last_release_tag == '':
            last_release_tag = call('git rev-list HEAD')[-1]

    # See https://git-scm.com/docs/pretty-formats - '%cI' is strict ISO-8601 format
    tag_date = call(f"git log -n1 --format='%cI' {last_release_tag}")[0]
    num_commits = call(f"git rev-list {last_release_tag}..HEAD --count")[0]
    committers = call(f"git log --since='{tag_date}' --format=%aN")
    committers = {c.strip() for c in committers if c.strip()}

    merges = get_merged_pulls(GH_USER, GH_REPO, tag_date,
                              branch=dev_branch)
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

    def key(name):
        name = [v for v in name.split() if v[0] in string.ascii_letters]
        if len(name) > 0:
            return name[-1]

    print(f'Release {last_release_tag} was on {tag_date}\n')
    print(f'A total of {num_commits} changes have been committed.\n')

    print(f'Made by the following {len(committers)} committers [alphabetical by last name]:')
    for committer in sorted(committers, key=key):
        print(f'- {committer}')
    print()

    print(f'It contained the following {num_merges} merged pull requests:')
    for pull in pulls:
        print(f'- {pull["title"]} (#{pull["number"]})')
    print()

    print(f'Created by the following {len(authors)} authors [alphabetical by last name]:')
    for author in sorted(authors, key=key):
        print(f'- {author}')
    print()

    print(f'Reviewed by the following {len(reviewers)} reviewers [alphabetical by last name]:')
    for reviewer in sorted(reviewers, key=key):
        print(f'- {reviewer}')
    print()
