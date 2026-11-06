#!/usr/bin/env python3
"""Acme contribution preflight (scope + focused tests).

Trust: execute the checker blob and read the allowlist blob from --base.
A working-tree copy is only a bootstrap; it re-execs the baseline blob.
This is not a tamperproof CI control. A workflow that skips these steps
can still be changed on a pull_request event.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

CHECKER_PATH = "tools/acme_preflight.py"
ALLOWLIST_PATH = "acme/task-allowlist.txt"
TRUST_ENV = "ACME_PREFLIGHT_TRUSTED_BASE"

DEFAULT_TEST_COMMAND = (
    "spin test -- tests/skimage/util/test_slice_along_axes.py "
    "tests/skimage2/util/test_slice_along_axes.py"
)

PROHIBITED_EXACT = {
    "pyproject.toml",
    "meson.build",
    "meson.options",
    CHECKER_PATH,
}
PROHIBITED_NAMES = {"pyproject.toml", "meson.build", "meson.options"}
PROHIBITED_PREFIXES = (
    "requirements/",
    ".github/",
    ".cursor/",
    "acme/",
)

NOT_CHECKED = (
    "whether added tests are strong or complete",
    "full test suite",
    "public signatures inside allowed files",
    "Grok Bot QA, maintainer review, merge, or deployment",
    "workflow integrity (pull_request can change CI YAML)",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = str(Path(args.repo).resolve())
    try:
        base_sha = _resolve_base(repo, args.base)
    except PreflightError as exc:
        _print_header(base="unresolved", revision="unresolved")
        print(f"baseline: failed\n  {exc}")
        print("allowlist: not checked")
        print("prohibited paths: not checked")
        print("focused tests: skipped (baseline unavailable)")
        _print_not_checked()
        return 1

    try:
        checker_blob = _git_cat_file(repo, base_sha, CHECKER_PATH)
        allowlist_blob = _git_cat_file(repo, base_sha, ALLOWLIST_PATH)
    except PreflightError as exc:
        _print_header(base=base_sha, revision=_revision_label(repo))
        print(f"baseline policy files: failed\n  {exc}")
        print("allowlist: not checked")
        print("prohibited paths: not checked")
        print("focused tests: skipped (policy files unavailable)")
        _print_not_checked()
        return 1

    if os.environ.get(TRUST_ENV) != base_sha:
        return _reexec_baseline(checker_blob, base_sha, repo, sys.argv[1:])

    running = Path(__file__).read_bytes()
    if running != checker_blob:
        print(
            f"executed checker does not match {base_sha}:{CHECKER_PATH}",
            file=sys.stderr,
        )
        return 1

    revision = _revision_label(repo)
    _print_header(base=base_sha, revision=revision)

    allowlist = _parse_allowlist(allowlist_blob)
    changed = _changed_paths(repo, base_sha)
    print(f"changed paths ({len(changed)}):")
    if changed:
        for path in changed:
            print(f"  {path}")
    else:
        print("  (none)")

    prohibited = [path for path in changed if _is_prohibited(path)]
    extra = [path for path in changed if path not in allowlist]

    allowlist_ok = not extra
    prohibited_ok = not prohibited
    print(f"allowlist: {'passed' if allowlist_ok else 'failed'}")
    if extra:
        print("  not on allowlist:")
        for path in extra:
            print(f"    {path}")
    print(f"prohibited paths: {'passed' if prohibited_ok else 'failed'}")
    if prohibited:
        print("  prohibited:")
        for path in prohibited:
            print(f"    {path}")

    if not allowlist_ok or not prohibited_ok:
        print("focused tests: skipped (scope check failed)")
        _print_not_checked()
        return 1

    if args.skip_tests:
        print("focused tests: skipped (--skip-tests)")
        _print_not_checked()
        return 0

    test_cmd = shlex.split(args.test_command)
    print("focused tests: running")
    print(f"  command: {shlex.join(test_cmd)}")
    result = subprocess.run(test_cmd, cwd=repo)
    if result.returncode != 0:
        print(f"focused tests: failed (exit {result.returncode})")
        _print_not_checked()
        return 1

    print("focused tests: passed")
    _print_not_checked()
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        help="Trusted baseline ref or SHA. Policy is read only from this commit.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Git repository to inspect (default: current directory).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Run scope checks only; skip the focused contribution tests.",
    )
    parser.add_argument(
        "--test-command",
        default=DEFAULT_TEST_COMMAND,
        help="Focused test command run only after scope checks pass.",
    )
    return parser.parse_args(argv)


class PreflightError(RuntimeError):
    pass


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=check,
        capture_output=True,
    )


def _resolve_base(repo: str, base: str) -> str:
    result = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}", check=False)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").strip()
        raise PreflightError(f"cannot resolve baseline {base!r} to a commit SHA: {err}")
    return result.stdout.decode("ascii").strip()


def _git_cat_file(repo: str, sha: str, path: str) -> bytes:
    probe = _git(repo, "cat-file", "-e", f"{sha}:{path}", check=False)
    if probe.returncode != 0:
        raise PreflightError(f"cannot read {path} from baseline {sha}")
    result = _git(repo, "cat-file", "-p", f"{sha}:{path}")
    return result.stdout


def _reexec_baseline(
    checker_blob: bytes, base_sha: str, repo: str, argv_rest: list[str]
) -> int:
    handle, tmp_name = tempfile.mkstemp(prefix="acme_preflight_", suffix=".py")
    os.write(handle, checker_blob)
    os.close(handle)
    env = os.environ.copy()
    env[TRUST_ENV] = base_sha
    rewritten: list[str] = []
    i = 0
    while i < len(argv_rest):
        item = argv_rest[i]
        if item == "--base" and i + 1 < len(argv_rest):
            rewritten.extend(["--base", base_sha])
            i += 2
            continue
        if item == "--repo" and i + 1 < len(argv_rest):
            rewritten.extend(["--repo", repo])
            i += 2
            continue
        rewritten.append(item)
        i += 1
    if "--base" not in rewritten:
        rewritten.extend(["--base", base_sha])
    if "--repo" not in rewritten:
        rewritten.extend(["--repo", repo])
    os.execve(sys.executable, [sys.executable, tmp_name, *rewritten], env)
    return 1


def _parse_allowlist(blob: bytes) -> set[str]:
    allowed: set[str] = set()
    for raw in blob.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(line)
    return allowed


def _changed_paths(repo: str, base_sha: str) -> list[str]:
    paths: set[str] = set()
    # Union of disjoint Git layers so a later staged deletion cannot hide
    # a path that still exists on HEAD relative to the trusted baseline.
    paths.update(_name_status_paths(repo, "-M", base_sha, "HEAD"))
    paths.update(_name_status_paths(repo, "-M", "--cached"))
    paths.update(_name_status_paths(repo, "-M"))
    untracked = _git(repo, "ls-files", "-z", "--others", "--exclude-standard")
    for item in untracked.stdout.split(b"\0"):
        if item:
            paths.add(item.decode("utf-8"))
    return sorted(paths)


def _name_status_paths(repo: str, *diff_args: str) -> set[str]:
    result = _git(repo, "diff", "-z", "--name-status", *diff_args)
    return _parse_name_status(result.stdout)


def _parse_name_status(data: bytes) -> set[str]:
    parts = data.split(b"\0")
    paths: set[str] = set()
    i = 0
    while i < len(parts) and parts[i]:
        status = parts[i].decode("ascii")
        i += 1
        if i >= len(parts):
            break
        if status.startswith("R") or status.startswith("C"):
            old = parts[i].decode("utf-8")
            i += 1
            new = parts[i].decode("utf-8") if i < len(parts) else ""
            i += 1
            if old:
                paths.add(old)
            if new:
                paths.add(new)
        else:
            path = parts[i].decode("utf-8")
            i += 1
            if path:
                paths.add(path)
    return paths


def _is_prohibited(path: str) -> bool:
    if path in PROHIBITED_EXACT:
        return True
    if Path(path).name in PROHIBITED_NAMES:
        return True
    return any(path.startswith(prefix) for prefix in PROHIBITED_PREFIXES)


def _revision_label(repo: str) -> str:
    head = _git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    dirty = _git(repo, "status", "--porcelain", "-uall", check=False).stdout.strip()
    return f"{head} (dirty)" if dirty else head


def _print_header(base: str, revision: str) -> None:
    print("=== Acme preflight ===")
    print(f"policy baseline: {base}")
    print(f"revision under test: {revision}")


def _print_not_checked() -> None:
    print("not checked:")
    for item in NOT_CHECKED:
        print(f"  - {item}")


if __name__ == "__main__":
    sys.exit(main())
