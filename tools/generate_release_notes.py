"""Generate release notes automatically from GitHub pull requests."""


import os
import re
import sys
import argparse
import logging
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Union
from collections.abc import Iterable
from collections import OrderedDict

try:
    import requests
    import requests_cache
    from tqdm import tqdm
    from github import Github
    from github.PullRequest import PullRequest
    from github.NamedUser import NamedUser
    from github.Commit import Commit
except ImportError as e:
    raise ImportError(
        "This script depends on the third party libraries PyGithub, requests, "
        "requests-cache, and tqdm"
    ) from e


logger = logging.getLogger(__name__)

here = Path(__file__).parent

REQUESTS_CACHE_PATH = Path(tempfile.gettempdir()) / "github_cache.sqlite"

GH_URL = "https://github.com"


def lazy_tqdm(*args, **kwargs):
    """Defer initialization of progress bar until first item is requested.

    Calling `tqdm(...)` prints the progress bar right there and then. This can scramble
    output, if more than one progress bar are initialized at the same time but their
    iteration is meant to be done later in successive order.
    """
    kwargs["file"] = kwargs.get("file", sys.stderr)
    yield from tqdm(*args, **kwargs)


def commits_between(
    gh: Github, org_name: str, start_rev: str, stop_rev: str
) -> set[Commit]:
    """Fetch commits between two revisions excluding the commit of `start_rev`."""
    repo = gh.get_repo(org_name)
    comparison = repo.compare(base=start_rev, head=stop_rev)
    commits = set(comparison.commits)
    assert repo.get_commit(start_rev) not in commits
    assert repo.get_commit(stop_rev) in commits
    return commits


def pull_requests_from_commits(commits: Iterable[Commit]) -> set[PullRequest]:
    """Fetch pull requests that are associated with the given `commits`."""
    all_pull_requests = set()
    for commit in commits:
        commit_pull_requests = list(commit.get_pulls())
        if len(commit_pull_requests) != 1:
            logger.info(
                "%s with no or multiple PR(s): %r",
                commit.html_url,
                [p.html_url for p in commit_pull_requests],
            )
        if any(not p.merged for p in commit_pull_requests):
            logger.error(
                "%s with unmerged PRs: %r",
            )
        for pull in commit_pull_requests:
            if pull in all_pull_requests:
                # Expected if pull request is merged without squashing
                logger.debug(
                    "%r associated with multiple commits",
                    pull.html_url,
                )
        all_pull_requests.update(commit_pull_requests)
    return all_pull_requests


@dataclass(frozen=True, kw_only=True)
class GitHubGraphQl:
    """Interface to query GitHub's GraphQL API for a particular repository."""

    org_name: str
    repo_name: str

    URL: str = "https://api.github.com/graphql"
    GRAPHQL_COAUTHOR: str = """
    query {{
      repository (owner: "{org_name}" name: "{repo_name}") {{
        object(expression: "{commit_sha}" ) {{
          ... on Commit {{
            commitUrl
            authors(first:{page_limit}) {{
              edges {{
                cursor
                node {{
                  name
                  email
                  user {{
                    login
                    databaseId
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    PAGE_LIMIT: int = 100

    def find_authors(self, commit_sha: str) -> dict[int, str]:
        """Find ID and login of (co-)author(s) for a commit.

        Other than GitHub's REST API, the GraphQL API supports returning all authors,
        including co-authors, of a commit.
        """
        headers = {"Authorization": f"Bearer {os.environ.get('GH_TOKEN')}"}
        query = self.GRAPHQL_COAUTHOR.format(
            org_name=self.org_name,
            repo_name=self.repo_name,
            commit_sha=commit_sha,
            page_limit=self.PAGE_LIMIT,
        )
        sanitized_query = json.dumps({"query": query.replace("\n", "")})
        response = requests.post(self.URL, data=sanitized_query, headers=headers)
        data = response.json()
        commit = data["data"]["repository"]["object"]
        edges = commit["authors"]["edges"]
        if len(edges) == self.PAGE_LIMIT:
            # TODO implement pagination if this becomes an issue, e.g. see
            # https://github.com/scientific-python/devstats-data/blob/e3cd826518bf590083409318b0a7518f7781084f/query.py#L92-L107
            logger.warning(
                "reached page limit while querying authors in %r, "
                "only the first %i authors will be included",
                commit["commitUrl"],
                self.PAGE_LIMIT,
            )

        coauthors = dict()
        for i, edge in enumerate(edges):
            node = edge["node"]
            user = node["user"]
            if user is None:
                logger.warning(
                    "could not determine GitHub user for %r in %r",
                    node,
                    commit["commitUrl"],
                )
                continue
            coauthors[user["databaseId"]] = user["login"]

        assert coauthors
        return coauthors


def contributors(
    gh: Github,
    org_repo: str,
    commits: Iterable[Commit],
    pull_requests: Iterable[PullRequest],
) -> tuple[set[NamedUser], set[NamedUser]]:
    """Fetch commit authors, co-authors and reviewers.

    `authors` are users which created or co-authored a commit.
    `reviewers` are users, who added reviews to a merged pull request or merged a
    pull request (committer of the merge commit).
    """
    authors = set()
    reviewers = set()

    org_name, repo_name = org_repo.split("/")
    ql = GitHubGraphQl(org_name=org_name, repo_name=repo_name)

    for commit in commits:
        if commit.author:
            authors.add(commit.author)
        if commit.committer:
            reviewers.add(commit.committer)
        if "Co-authored-by:" in commit.commit.message:
            # Fallback on GraphQL API to find co-authors as well
            user_ids = ql.find_authors(commit.sha)
            for user_id, user_login in user_ids.items():
                named_user = gh.get_user_by_id(user_id)
                assert named_user.login == user_login
                authors.add(named_user)
        else:
            logger.debug("no co-authors in %r", commit.html_url)

    for pull in pull_requests:
        for review in pull.get_reviews():
            if review.user:
                reviewers.add(review.user)

    return authors, reviewers


@dataclass(frozen=True, kw_only=True)
class MdFormatter:
    """Format release notes in Markdown from PRs, authors and reviewers."""

    pull_requests: set[PullRequest]
    authors: set[Union[NamedUser]]
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
    outro_template: str = (
        "*These lists are automatically generated, and may not be complete or may "
        "contain duplicates.*\n"
    )
    # Associate regexes matching PR labels to a section titles in the release notes
    regex_section_map: tuple[tuple[str, str], ...] = (
        (".*Highlight.*", "Highlights"),
        (".*New feature.*", "New Features"),
        (".*Enhancement.*", "Enhancements"),
        (".*Performance.*", "Performance"),
        (".*Bug fix.*", "Bug Fixes"),
        (".*API.*", "API Changes"),
        (".*Maintenance.*", "Maintenance"),
        (".*Documentation.*", "Documentation"),
        (".*Infrastructure.*", "Infrastructure"),
    )
    ignored_user_logins: tuple[str] = ("web-flow",)
    pr_summary_regex = re.compile(
        r"^```release-note\s*(?P<summary>[\s\S]*?\w[\s\S]*?)\s*^```", flags=re.MULTILINE
    )

    def __str__(self) -> str:
        """Return complete release notes document as a string."""
        return self.document

    def __iter__(self) -> Iterable[str]:
        """Iterate the release notes document line-wise."""
        return self.iter_lines()

    @property
    def document(self) -> str:
        """Return complete release notes document as a string."""
        return "".join(self.iter_lines())

    def iter_lines(self) -> Iterable[str]:
        """Iterate the release notes document line-wise."""
        yield from self._format_section_title(
            self.title_template.format(version=self.version), level=1
        )
        yield from self._format_intro(version=self.version)
        for title, pull_requests in self._prs_by_section.items():
            yield from self._format_pr_section(title, pull_requests)
        yield from self._format_contributor_section(self.authors, self.reviewers)
        yield from self._format_outro()

    @property
    def _prs_by_section(self) -> OrderedDict[str, set[PullRequest]]:
        """Map pull requests to section titles.

        Pull requests whose labels do not match one of the sections given in
        `regex_section_map`, are sorted into a section named "Other".
        """
        label_section_map = {
            re.compile(pattern): section_name
            for pattern, section_name in self.regex_section_map
        }
        prs_by_section = OrderedDict()
        for _, section_name in self.regex_section_map:
            prs_by_section[section_name] = set()
        prs_by_section["Other"] = set()

        for pr in self.pull_requests:
            matching_sections = [
                section_name
                for regex, section_name in label_section_map.items()
                if any(regex.match(label.name) for label in pr.labels)
            ]
            for section_name in matching_sections:
                prs_by_section[section_name].add(pr)
            if not matching_sections:
                logger.warning(
                    "%s without matching label, sorting into section 'Other'",
                    pr.html_url,
                )
                prs_by_section["Other"].add(pr)

        return prs_by_section

    def _sanitize_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace("\r\n", " ")
        text = text.replace("\n", " ")
        return text

    def _format_link(self, name: str, target: str) -> str:
        return f"[{name}]({target})"

    def _format_section_title(self, title: str, *, level: int) -> Iterable[str]:
        yield f"{'#' * level} {title}\n"

    def _parse_pull_request_summary(self, pr: PullRequest) -> str:
        if pr.body and (match := self.pr_summary_regex.search(pr.body)):
            summary = match["summary"]
        else:
            logger.debug("falling back to title for %s", pr.html_url)
            summary = pr.title
        summary = self._sanitize_text(summary)
        return summary

    def _format_pull_request(self, pr: PullRequest) -> Iterable[str]:
        summary = self._parse_pull_request_summary(pr).rstrip(".")
        yield f"- {summary}\n"
        link = self._format_link(f"#{pr.number}", f"{pr.html_url}")
        yield f"  ({link}).\n"

    def _format_pr_section(
        self, title: str, pull_requests: set[PullRequest]
    ) -> Iterable[str]:
        """Format a section title and list its pull requests sorted by merge date."""
        yield from self._format_section_title(title, level=2)
        for pr in sorted(pull_requests, key=lambda pr: pr.merged_at):
            yield from self._format_pull_request(pr)
        yield "\n"

    def _format_user_line(self, user: Union[NamedUser]) -> str:
        line = f"@{user.login}"
        line = self._format_link(line, user.html_url)
        if user.name:
            line = f"{user.name} ({line})"
        return line + ",\n"

    def _format_contributor_section(
        self,
        authors: set[Union[NamedUser]],
        reviewers: set[NamedUser],
    ) -> Iterable[str]:
        """Format contributor section and list users sorted by login handle."""
        authors = {u for u in authors if u.login not in self.ignored_user_logins}
        reviewers = {u for u in reviewers if u.login not in self.ignored_user_logins}

        yield from self._format_section_title("Contributors", level=2)
        yield "\n"

        yield f"{len(authors)} authors added to this release (alphabetically):\n"
        author_lines = map(self._format_user_line, authors)
        yield from sorted(author_lines, key=lambda s: s.lower())
        yield "\n"

        yield f"{len(reviewers)} reviewers added to this release (alphabetically):\n"
        reviewers_lines = map(self._format_user_line, reviewers)
        yield from sorted(reviewers_lines, key=lambda s: s.lower())
        yield "\n"

    def _format_intro(self, version):
        intro = self.intro_template.format(version=version)
        # Make sure to return exactly one line at a time
        yield from (f"{line}\n" for line in intro.split("\n"))

    def _format_outro(self) -> Iterable[str]:
        outro = self.outro_template
        # Make sure to return exactly one line at a time
        yield from (f"{line}\n" for line in outro.split("\n"))


class RstFormatter(MdFormatter):
    """Format release notes in reStructuredText from PRs, authors and reviewers."""

    def _sanitize_text(self, text) -> str:
        text = super()._sanitize_text(text)
        text = text.replace("`", "``")
        return text

    def _format_link(self, name: str, target: str) -> str:
        return f"`{name} <{target}>`_"

    def _format_section_title(self, title: str, *, level: int) -> Iterable[str]:
        yield title + "\n"
        underline = {1: "=", 2: "-", 3: "~"}
        yield underline[level] * len(title) + "\n"


def parse_command_line(func: Callable) -> Callable:
    """Define and parse command line options.

    Has no effect if any keyword argument is passed to the underlying function.
    """
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(
        "org_repo",
        help="Org and repo name of a repository on GitHub (delimited by a slash), "
        "e.g. 'numpy/numpy'",
    )
    parser.add_argument(
        "start_rev",
        help="The starting revision (excluded), e.g. the tag of the previous release",
    )
    parser.add_argument(
        "stop_rev",
        help="The stop revision (included), e.g. the 'main' branch or the current "
        "release",
    )
    parser.add_argument(
        "--version",
        default="0.0.0",
        help="Version you're about to release, used title and description of the notes",
    )
    parser.add_argument("--out", help="Write to file, prints to STDOUT otherwise")
    parser.add_argument(
        "--format",
        choices=["rst", "md"],
        default="md",
        help="Choose format, defaults to Markdown",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached requests to GitHub's API before running",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging level",
    )

    def wrapped(**kwargs):
        if not kwargs:
            kwargs = vars(parser.parse_args())
        return func(**kwargs)

    return wrapped


@parse_command_line
def main(
    *,
    org_repo: str,
    start_rev: str,
    stop_rev: str,
    version: str,
    out: str,
    format: str,
    clear_cache: bool,
    verbose: int,
):
    """Main function of the script.

    See :func:`parse_command_line` for a description of the accepted input.
    """
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logger.setLevel(level)

    requests_cache.install_cache(
        REQUESTS_CACHE_PATH, backend="sqlite", expire_after=3600
    )
    print(f"Using requests cache at {REQUESTS_CACHE_PATH}")
    if clear_cache:
        requests_cache.clear()
        logger.info("cleared requests cache at %s", REQUESTS_CACHE_PATH)

    gh_token = os.environ.get("GH_TOKEN")
    if gh_token is None:
        raise RuntimeError(
            "You need to set the environment variable `GH_TOKEN`. "
            "The token is used to avoid rate limiting, "
            "and can be created at https://github.com/settings/tokens.\n\n"
            "The token does not require any permissions (we only use the public API)."
        )
    gh = Github(gh_token)

    print("Fetching commits...", file=sys.stderr)
    commits = commits_between(gh, org_repo, start_rev, stop_rev)
    pull_requests = pull_requests_from_commits(
        lazy_tqdm(commits, desc="Fetching pull requests")
    )
    authors, reviewers = contributors(
        gh=gh,
        org_repo=org_repo,
        commits=lazy_tqdm(commits, desc="Fetching authors"),
        pull_requests=lazy_tqdm(pull_requests, desc="Fetching reviewers"),
    )

    Formatter = {"md": MdFormatter, "rst": RstFormatter}[format]
    formatter = Formatter(
        pull_requests=pull_requests,
        authors=authors,
        reviewers=reviewers,
        version=version,
    )

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as io:
            io.writelines(formatter.iter_lines())
    else:
        print()
        for line in formatter.iter_lines():
            assert line.endswith("\n")
            assert line.count("\n") == 1
            print(line, end="", file=sys.stdout)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(filename)s::%(funcName)s: %(message)s",
        stream=sys.stderr,
    )
    main()
