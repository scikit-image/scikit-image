"""Generate release notes automatically from GitHub pull requests.

Start with:
```
pip install pygithub gitpython tqdm
```
and export an GitHub token (classic) with public repo access
```
export GH_TOKEN=<your-gh-api-token>
```
Then, for to include everything from a certain release to main:
```
python /path/to/generate_release_notes.py v0.14.0 main --version 0.15.0
```
Or two include only things between two releases:
```
python /path/to/generate_release_notes.py v.14.2 v0.14.3 --version 0.14.3
```

References
https://github.com/scikit-image/scikit-image/blob/main/tools/generate_release_notes.py
https://github.com/scikit-image/scikit-image/issues/3404
https://github.com/scikit-image/scikit-image/issues/3405
"""


import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import requests_cache
from tqdm import tqdm
from github import Github, Repository, PullRequest, NamedUser


logger = logging.getLogger(__name__)

here = Path(__file__).parent

GH_URL = "https://github.com"
GH_ORG = 'scikit-image'
GH_REPO = 'scikit-image'


def lazy_tqdm(*args, **kwargs):
    kwargs["file"] = kwargs.get("file", sys.stderr)
    yield from tqdm(*args, **kwargs)


def commits_between(repo: Repository, start_rev, stop_rev):
    # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#compare-two-commits
    comparison = repo.compare(base=start_rev, head=stop_rev)
    return comparison.commits


def pull_requests_from_commits(commits):
    all_pull_requests = set()
    for commit in commits:
        commit_pull_requests = list(commit.get_pulls())
        if len(commit_pull_requests) != 1:
            logger.info(
                "commit %s with no or multiple PR(s): %r",
                commit.html_url,
                [p.html_url for p in commit_pull_requests]
            )
        if any(not p.merged for p in commit_pull_requests):
            logger.error(
                "commit %s with unmerged PRs: %r",

            )
        for pull in commit_pull_requests:
            if pull in all_pull_requests:
                # May happen if
                logger.info(
                    "pull request associated with multiple commits: %r",
                    pull.html_url,
                )
        all_pull_requests.update(commit_pull_requests)
    return all_pull_requests


def contributors(commits, pull_requests):
    authors = set()
    reviewers = set()

    for commit in commits:
        # TODO include co-authors?
        if commit.author:
            authors.add(commit.author)
        if commit.committer:
            reviewers.add(commit.committer)

    for pull in pull_requests:
        for review in pull.get_reviews():
            if review.user:
                reviewers.add(review.user)

    return authors, reviewers


@dataclass
class MdFormatter:

    pull_requests: set[PullRequest]
    authors: set[NamedUser]
    reviewers: set[NamedUser]

    version: str = "x.y.z"
    title_template: str = "scikit-image {version} release notes"
    intro_template: str = """
We're happy to announce the release of scikit-image {version}!
scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:
https://scikit-image.org
"""
    label_section_map: tuple[str, str] = (
        (":trophy: type: Highlight" , "Highlights"),
        (":baby: type: New feature" , "New Features"),
        (":fast_forward: type: Enhancement" , "Enhancements"),
        (":chart_with_upwards_trend: type: Performance" , "Performance"),
        (":adhesive_bandage: type: Bug fix" , "Bug Fixes"),
        (":scroll: type: API" , "API Changes"),
        (":wrench: type: Maintenance" , "Maintenance"),
        (":page_facing_up: type: Documentation" , "Documentation"),
        (":robot: type: Infrastructure" , "Infrastructure"),
    )
    ignored_user_logins: tuple[str] = ("web-flow",)

    def __str__(self):
        return "\n".join(self.iter_lines())

    @property
    def intro(self):
        return self.intro_template.format(version=self.version)

    def _prs_by_section(self) -> dict[str, set[PullRequest]]:
        label_section_map = {k: v for k, v in self.label_section_map}
        prs_by_section = {
            section_name: set() for section_name in label_section_map.values()
        }
        prs_by_section["Other"] = set()
        for pr in self.pull_requests:
            pr_labels = {label.name for label in pr.labels}
            pr_labels = pr_labels & label_section_map.keys()
            if not pr_labels:
                logger.warning(
                    "pull request %s without known section label, sorting into 'Other'",
                    pr.html_url
                )
                prs_by_section["Other"].add(pr)
            for name in pr_labels:
                prs_by_section[label_section_map[name]].add(pr)

        return prs_by_section

    def _format_pr_title(self, title):
        title = title.strip()
        return title

    def _format_section_title(self, title, level):
        return f"{'#' * level} {title}"

    def _format_link(self, name, target):
        return f"[{name}]({target})"

    def _format_pr_section(self, title, pull_requests):
        pull_url = f"{GH_URL}/{GH_ORG}/{GH_REPO}/pull"
        yield self._format_section_title(title, 2)
        for pr in sorted(pull_requests, key=lambda pr: pr.merged_at):
            yield f"- {self._format_pr_title(pr.title)}"
            link = self._format_link(f"#{pr.number}", f"{pull_url}/{pr.number}")
            yield f"  ({link})."
        yield ""

    def _format_contributor_section(self, authors, reviewers):
        authors = {u for u in authors if u.login not in self.ignored_user_logins}
        reviewers = {u for u in reviewers if u.login not in self.ignored_user_logins}

        yield self._format_section_title("Contributors", 2)

        def format_user(user):
            line = self._format_link(f"@{user.login}", user.html_url)
            if user.name:
                line = f"{user.name} ({line})"
            return line

        authors = sorted(authors, key=lambda user: user.login)
        paragraph = f"{len(authors)} authors added to this release (sorted by login):\n"
        paragraph += ",\n".join(format_user(user) for user in authors)
        yield paragraph
        yield ""

        reviewers = sorted(reviewers, key=lambda user: user.login)
        paragraph = (
            f"{len(reviewers)} reviewers added to this release (sorted by login):\n"
        )
        paragraph += ",\n".join(format_user(user) for user in reviewers)
        yield paragraph
        yield ""

    def iter_lines(self):
        yield self._format_section_title(
            self.title_template.format(version=self.version), 1
        )
        yield self.intro_template.format(version=self.version)
        for title, pull_requests in self._prs_by_section().items():
            yield from self._format_pr_section(title, pull_requests)
        yield from self._format_contributor_section(self.authors, self.reviewers)


class RstFormatter(MdFormatter):

    def _format_pr_title(self, title):
        title = super()._format_pr_title(title)
        title = title.replace("`", "``")
        return title

    def _format_section_title(self, title, level):
        underline = {
            1: "=", 2: "-", 3: "~"
        }
        return f"{title}\n{underline[level] * len(title)}"

    def _format_link(self, name, target):
        return f"`{name} <{target}>`_"


def cli(func):
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('start_rev', help='The starting revision (excluded)')
    parser.add_argument('stop_rev', help='The stop revision (included)')
    parser.add_argument(
        '--version', help="Version you're about to release", default='0.2.0'
    )
    parser.add_argument(
        "--out", help="Write to file, prints to STDOUT otherwise"
    )
    parser.add_argument(
        "--format", choices=["rst", "md"],
        default="md",
        help="Choose format, defaults to Markdown"
    )

    def wrapped(**kwargs):
        if not kwargs:
            kwargs = vars(parser.parse_args())
        return func(**kwargs)

    return wrapped


@cli
def main(*, start_rev, stop_rev, version, out, format):
    # TODO option to delete cache
    requests_cache.install_cache(
        'github_cache', backend='sqlite', expire_after=3600
    )

    gh_token = os.environ.get('GH_TOKEN')
    if gh_token is None:
        raise RuntimeError(
            "It is necessary that the environment variable `GH_TOKEN` "
            "be set to avoid running into problems with rate limiting. "
            "One can be acquired at https://github.com/settings/tokens.\n\n"
            "You do not need to select any permission boxes while generating "
            "the token."
        )
    gh = Github(gh_token)
    repo = gh.get_repo(f"{GH_ORG}/{GH_REPO}")

    print("Getting commits...", file=sys.stderr)
    commits = commits_between(repo, start_rev, stop_rev)
    pull_requests = pull_requests_from_commits(
        lazy_tqdm(commits, desc="Getting pull requests")
    )
    authors, reviewers = contributors(
        commits=lazy_tqdm(commits, desc="Getting authors", ),
        pull_requests=lazy_tqdm(pull_requests, desc="Getting reviewers"),
    )

    Formatter = {"md": MdFormatter, "rst": RstFormatter}[format]
    formatter = Formatter(pull_requests, authors, reviewers, version)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as io:
            io.writelines(formatter.iter_lines())
    else:
        print()
        print(str(formatter), file=sys.stdout)


if __name__ == "__main__":
    logging.basicConfig()
    main()
