"""Guardrail tests for the Acme preflight checker.

These use temporary Git repositories. They do not modify this clone's
dependency or CI files, and they do not prove contribution-test quality.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_SRC = REPO_ROOT / "tools" / "acme_preflight.py"
ALLOWLIST_SRC = REPO_ROOT / "acme" / "task-allowlist.txt"
WORKFLOW_SRC = REPO_ROOT / ".github" / "workflows" / "acme-preflight.yaml"

CONTRIB_PATHS = [
    "src/_skimage2/util/_slice_along_axes.py",
    "tests/skimage/util/test_slice_along_axes.py",
    "tests/skimage2/util/test_slice_along_axes.py",
]

PASSING_TEST = f"{sys.executable} -c \"raise SystemExit(0)\""
FAILING_TEST = f"{sys.executable} -c \"raise SystemExit(1)\""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _init_baseline(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "-c", "init.templateDir=", "init", "-b", "main")
    _git(repo, "config", "user.name", "acme-preflight")
    _git(repo, "config", "user.email", "acme-preflight@example.test")
    _git(repo, "config", "commit.gpgsign", "false")
    checker = repo / CHECKER_SRC.relative_to(REPO_ROOT)
    allowlist = repo / ALLOWLIST_SRC.relative_to(REPO_ROOT)
    checker.parent.mkdir(parents=True)
    allowlist.parent.mkdir(parents=True)
    checker.write_bytes(CHECKER_SRC.read_bytes())
    allowlist.write_bytes(ALLOWLIST_SRC.read_bytes())
    _git(repo, "add", str(CHECKER_SRC.relative_to(REPO_ROOT)))
    _git(repo, "add", str(ALLOWLIST_SRC.relative_to(REPO_ROOT)))
    _git(repo, "commit", "-m", "trusted baseline")
    baseline = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, baseline


def _write_contrib(repo: Path) -> None:
    for rel in CONTRIB_PATHS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("contrib\n", encoding="utf-8")


def _documented_run(
    repo: Path, base: str, extra_args: list[str]
) -> subprocess.CompletedProcess:
    """Run the checker blob from the resolved baseline, not the working tree."""
    sha = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").stdout.strip()
    _git(repo, "cat-file", "-e", f"{sha}:tools/acme_preflight.py")
    _git(repo, "cat-file", "-e", f"{sha}:acme/task-allowlist.txt")
    blob = _git(
        repo, "cat-file", "-p", f"{sha}:tools/acme_preflight.py"
    ).stdout.encode()
    handle, tmp_name = tempfile.mkstemp(prefix="acme_preflight_test_", suffix=".py")
    Path(tmp_name).write_bytes(blob)
    import os

    os.close(handle)
    return subprocess.run(
        [
            sys.executable,
            tmp_name,
            "--base",
            sha,
            "--repo",
            str(repo),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_only_approved_paths_are_accepted(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    result = _documented_run(repo, base, ["--test-command", PASSING_TEST])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "allowlist: passed" in result.stdout
    assert "prohibited paths: passed" in result.stdout
    assert "focused tests: passed" in result.stdout


def test_extra_untracked_file_is_rejected(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    (repo / "oops.txt").write_text("extra\n", encoding="utf-8")
    result = _documented_run(repo, base, ["--test-command", PASSING_TEST])
    assert result.returncode != 0
    assert "allowlist: failed" in result.stdout
    assert "oops.txt" in result.stdout
    assert "focused tests: skipped (scope check failed)" in result.stdout


def test_staged_prohibited_change_is_rejected(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    prohibited = repo / "requirements" / "out-of-scope.txt"
    prohibited.parent.mkdir()
    prohibited.write_text("not a real dependency edit\n", encoding="utf-8")
    _git(repo, "add", "requirements/out-of-scope.txt")
    result = _documented_run(repo, base, ["--test-command", PASSING_TEST])
    assert result.returncode != 0
    assert "prohibited paths: failed" in result.stdout
    assert "requirements/out-of-scope.txt" in result.stdout
    assert "focused tests: skipped (scope check failed)" in result.stdout


def test_working_tree_allowlist_cannot_authorize_extra_file(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    (repo / "sneaky.txt").write_text("extra\n", encoding="utf-8")
    allowlist = repo / "acme" / "task-allowlist.txt"
    allowlist.write_text(
        allowlist.read_text(encoding="utf-8") + "sneaky.txt\n",
        encoding="utf-8",
    )
    result = _documented_run(repo, base, ["--test-command", PASSING_TEST])
    assert result.returncode != 0
    assert "sneaky.txt" in result.stdout
    assert "allowlist: failed" in result.stdout
    assert "focused tests: skipped (scope check failed)" in result.stdout


def test_working_tree_checker_cannot_bypass_documented_path(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    (repo / "bypass.txt").write_text("extra\n", encoding="utf-8")
    evil = repo / "tools" / "acme_preflight.py"
    evil.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    documented = _documented_run(repo, base, ["--skip-tests"])
    assert documented.returncode != 0
    assert "bypass.txt" in documented.stdout

    direct = subprocess.run(
        [
            sys.executable,
            str(evil),
            "--base",
            base,
            "--repo",
            str(repo),
            "--skip-tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0


def test_failing_focused_tests_fail_preflight(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    result = _documented_run(repo, base, ["--test-command", FAILING_TEST])
    assert result.returncode != 0
    assert "allowlist: passed" in result.stdout
    assert "focused tests: failed" in result.stdout
    assert "skipped" not in result.stdout.split("focused tests:", 1)[1].splitlines()[0]


def test_rename_reports_old_and_new_paths(tmp_path: Path) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    for rel in CONTRIB_PATHS:
        _git(repo, "add", rel)
    _git(repo, "commit", "-m", "contrib")
    _git(repo, "mv", CONTRIB_PATHS[0], "renamed.py")
    result = _documented_run(repo, base, ["--skip-tests"])
    assert result.returncode != 0
    assert CONTRIB_PATHS[0] in result.stdout
    assert "renamed.py" in result.stdout


def test_committed_prohibited_file_not_hidden_by_staged_removal(
    tmp_path: Path,
) -> None:
    repo, base = _init_baseline(tmp_path)
    _write_contrib(repo)
    prohibited = repo / "requirements" / "out-of-scope.txt"
    prohibited.parent.mkdir()
    prohibited.write_text("committed then staged for deletion\n", encoding="utf-8")
    _git(repo, "add", "requirements/out-of-scope.txt")
    _git(repo, "commit", "-m", "prohibited after baseline")
    _git(repo, "rm", "requirements/out-of-scope.txt")
    result = _documented_run(repo, base, ["--skip-tests"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert "prohibited paths: failed" in result.stdout
    assert "requirements/out-of-scope.txt" in result.stdout
    assert "focused tests: skipped (scope check failed)" in result.stdout


def test_parse_name_status_includes_both_rename_paths() -> None:
    spec = importlib.util.spec_from_file_location("acme_preflight", CHECKER_SRC)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = module._parse_name_status(b"R100\0old.py\0new.py\0M\0kept.py\0")
    assert paths == {"old.py", "new.py", "kept.py"}


def test_workflow_extracts_checker_from_pr_base() -> None:
    text = WORKFLOW_SRC.read_text(encoding="utf-8")
    assert "pull_request_target" not in text
    assert "persist-credentials: false" in text
    assert "contents: read" in text
    assert "github.event.pull_request.base.sha" in text
    assert 'git cat-file -p "${BASE_SHA}:tools/acme_preflight.py"' in text
    assert "--skip-tests" in text
    assert "Scope validation" in text
    assert "Install contribution test environment" in text
    assert (
        "python -m pip install -r requirements/build.txt "
        "-r requirements/default.txt -r requirements/test.txt"
    ) in text
    assert "python -m pip install --no-build-isolation -e ." in text
    skip_at = text.index("Scope validation")
    install_at = text.index("Install contribution test environment")
    guardrail_at = text.index("Guardrail tests")
    focused_at = text.index("Focused contribution tests")
    assert skip_at < install_at < guardrail_at < focused_at
