"""Generate the release notes automatically from Github pull requests.

Start with:
```
export GH_TOKEN=<your-gh-api-token>
```

Then, for a major release:
```
python /path/to/generate_release_notes.py v0.14.0 master --version 0.15.0
```

For a minor release:
```
python /path/to/generate_release_notes.py v.14.2 v0.14.x --version 0.14.3
```

You should probably redirect the output with:
```
python /path/to/generate_release_notes.py [args] | tee release_notes.rst
```

You'll require PyGitHub and tqdm, which you can install with:
```
pip install -r requirements/_release_tools.txt
```

References
https://github.com/scikit-image/scikit-image/issues/3404
https://github.com/scikit-image/scikit-image/issues/3405
"""
import os
import argparse
from datetime import datetime
from collections import OrderedDict
import string
from warnings import warn

from github import Github
try:
    from tqdm import tqdm
except ImportError:
    from warnings import warn
    warn('tqdm not installed. This script takes approximately 5 minutes '
         'to run. To view live progressbars, please install tqdm. '
         'Otherwise, be patient.')
    def tqdm(i, **kwargs):
        return i


GH_USER = 'scikit-image'
GH_REPO = 'scikit-image'
GH_TOKEN = os.environ.get('GH_TOKEN')
if GH_TOKEN is None:
    raise RuntimeError(
        "It is necessary that the environment variable `GH_TOKEN` "
        "be set to avoid running into problems with rate limiting. "
        "One can be acquired at https://github.com/settings/tokens.\n\n"
        "You do not need to select any permission boxes while generating "
        "the token.")

g = Github(GH_TOKEN)
repository = g.get_repo(f'{GH_USER}/{GH_REPO}')


parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('from_commit', help='The starting tag.')
parser.add_argument('to_commit', help='The head branch.')
parser.add_argument('--version', help="Version you're about to release.",
                    nargs=1, default='0.15.0')

args = parser.parse_args()

for tag in repository.get_tags():
    if tag.name == args.from_commit:
        previous_tag = tag
        break
else:
    raise RuntimeError(f'Desired tag ({args.from_commit}) not found')

# For some reason, go get the github commit from the commit to get
# the correct date
github_commit =  previous_tag.commit.commit
previous_tag_date = datetime.strptime(github_commit.last_modified,
                                      '%a, %d %b %Y %H:%M:%S %Z')


all_commits = list(tqdm(repository.get_commits(sha=args.to_commit,
                                               since=previous_tag_date),
                        desc=f'Getting all commits between {args.from_commit} '
                             f'and {args.to_commit}'))
all_hashes = set(c.sha for c in all_commits)

authors = set()
reviewers = set()
committers = set()
users = dict()  # keep track of known usernames

def add_to_users(users, new_user):
    if new_user.login not in users:
        if new_user.name is None:
            users[new_user.login] = new_user.login
        else:
            users[new_user.login] = new_user.name

for commit in tqdm(all_commits, desc='Getting commiters and authors'):
    # committer can be None?
    if commit.committer:
        add_to_users(users, commit.committer)
        committers.add(users[commit.committer.login])

    # Users that deleted their accounts will appear as None
    # So annoying
    if commit.author is None:
        warn('Could not find author of commit :' + commit.sha)
        continue

    add_to_users(users, commit.author)
    authors.add(users[commit.author.login])
    if None in authors:
        import pdb; pdb.set_trace()
        print('Why do I get here')
# this gets found as a commiter
committers.discard('GitHub Web Flow')
authors.discard('Azure Pipelines Bot')
assert None not in authors
assert None not in committers

highlights = OrderedDict()

highlights['New Feature'] = {}
highlights['Improvement'] = {}
highlights['Bugfix'] = {}
highlights['API Change'] = {}
highlights['Deprecations'] = {}
highlights['Build Tool'] = {}
other_pull_requests = {}

for pull in tqdm(g.search_issues(f'repo:{GH_USER}/{GH_REPO} '
                                 f'merged:>{previous_tag_date.isoformat()} '
                                 'sort:created-asc'),
                 desc='Iterating through Pull Requests'):
    pr = repository.get_pull(pull.number)
    if pr.merge_commit_sha in all_hashes:
        summary = pull.title
        for r in pr.get_reviews():
            if r.user.login not in users:
                users[r.user.login] = r.user.name
            reviewers.add(users[r.user.login])
        for key, key_dict in highlights.items():
            pr_title_prefix = (key + ': ').lower()
            if summary.lower().startswith(pr_title_prefix):
                key_dict[pull.number] = {
                    'summary': summary[len(pr_title_prefix):]}
                break
        else:
            other_pull_requests[pull.number] = {
                'summary': summary
            }


# add Other PRs to the ordered dict to make doc generation easier.
highlights['Other Pull Request'] = other_pull_requests



def name_sorting_key(name):
    name = [v for v in name.split() if v[0] in string.ascii_letters]
    if len(name) > 0:
        return name[-1]


# Now generate the release notes
announcement_title = "Announcement: scikit-image {}".format(args.version)
print(announcement_title)
print("="*len(announcement_title))

print(f"""
We're happy to announce the release of scikit-image v{args.version}!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and moreself.
""")

if args.version.startswith('0.14'):
    print("""
This is the last major release with official support for Python 2.7. Future
releases will be developed using Python 3-only syntax.

However, 0.14 is a long-term support (LTS) release and will receive bug fixes
and backported features deemed important (by community demand) until January
1st 2020 (end of maintenance for Python 2.7; see PEP 373 for details).
""")

print("""
For more information, examples, and documentation, please visit our website:

http://scikit-image.org

"""
)

for section, pull_request_dicts in highlights.items():
    if not pull_request_dicts:
        continue
    print("""
{section}
{key_underline}
""".format(section=section+"s", key_underline='-'*(len(section)+1)))
    for number, pull_request_info in pull_request_dicts.items():
        print('- {} (#{})'.format(pull_request_info['summary'], number))


contributors = OrderedDict()

contributors['authors'] = authors
contributors['committers'] = committers
contributors['reviewers'] = reviewers

for section_name, contributor_set in contributors.items():
    print()
    committer_str = (f'{len(committers)} {section_name} added to this release '
                     ' [alphabetical by last name]')
    print(committer_str)
    print('-'*len(committer_str))
    for c in sorted(contributor_set, key=name_sorting_key):
        print(f'- {c}')
    print()
