#!/usr/bin/env python
# Requires package 'editdistance'

# A mailmap file is used (by GitHub and other tools) to associate multiple
# commit emails with one user.  This helps to count number of commits,
# contributors, etc.

import subprocess
import shlex
import numpy as np
from collections import defaultdict

from editdistance import eval as dist

threshold = 5

def call(cmd):
    return subprocess.check_output(shlex.split(cmd), universal_newlines=True).split('\n')


def _clean_email(email):
    if not '@' in email:
        return

    name, domain = email.split('@')
    name = name.split('+', 1)[0]

    return '{}@{}'.format(name, domain).lower()


call("rm -f .mailmap")
authors = call("git log --format='%aN::%aE'")

names, emails = [], []

for (name, email) in (author.split('::') for author in authors if author.strip()):
    if email not in emails:
        names.append(name)
        emails.append(email)

N = len(names)
D = np.zeros((N, N)) + np.infty

for i in range(1, N):
    for j in range(i):
        D[i, j] = dist(names[i], names[j])

for i in range(N):
    dupes, = np.where(D[:, i] < threshold)
    for j in dupes:
        names[j] = names[i]

mailmap = defaultdict(set)
for (name, email) in zip(names, emails):
    email = _clean_email(email)
    if email:
        mailmap[name].add(email)

for key, value in list(mailmap.items()):
    if len(value) < 2 or (len(key.split()) < 2):
        mailmap.pop(key)

entries = []
for name, emails in mailmap.items():
    entries.append([name])
    entries[-1].extend(['<{}>'.format(email) for email in emails])

entries = sorted(entries, key=lambda x: x[0].split()[-1])
for entry in entries:
    print(' '.join(entry))
