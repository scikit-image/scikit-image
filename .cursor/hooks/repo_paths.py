"""Shared repo-relative path checks for Cursor hooks."""

from __future__ import annotations

import fnmatch
from pathlib import Path

SKILL_PATH_PREFIX = ".cursor/skills/"


def repo_relative(path: str, cwd: str) -> str:
    root = Path(cwd).resolve()
    p = Path(path)
    try:
        if p.is_absolute():
            return p.resolve().relative_to(root).as_posix()
        return (root / p).resolve().relative_to(root).as_posix()
    except ValueError:
        return p.resolve().as_posix() if p.is_absolute() else p.as_posix()


def is_audit_path(path: str, *, cwd: str) -> bool:
    norm = repo_relative(path, cwd)
    return norm == ".cursor/audit" or norm.startswith(".cursor/audit/")


def is_skill_path(path: str, *, cwd: str) -> bool:
    return repo_relative(path, cwd).startswith(SKILL_PATH_PREFIX)


def matches(path: str, patterns: tuple[str, ...], *, cwd: str) -> bool:
    norm = repo_relative(path, cwd)
    for pat in patterns:
        if fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(norm, pat.rstrip("/")):
            return True
    return False
