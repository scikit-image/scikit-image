#!/usr/bin/env python3
"""Block agent edits to high-risk paths unless user confirms (Cursor preToolUse)."""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_log import emit

# Paths/globs requiring approval to create, modify, or delete.
PROTECTED = (
    ".cursor/hooks.json",
    ".cursor/hooks/*",
    ".cursor/rules/security.mdc",
    "AGENTS.md",
    ".github/workflows/*",
    ".github/workflows/**",
    "CODEOWNERS",
    ".github/CODEOWNERS",
    ".pre-commit-config.yaml",
    ".gitignore",
    "pyproject.toml",
    "requirements/*",
)

ALWAYS_DENY = (
    ".git/*",
    ".git/**",
)

_OUTCOME = {"deny": "denied", "ask": "approval_requested"}


def _repo_relative(path: str, cwd: str) -> str:
    """Normalize to a repo-relative POSIX path for portable pattern matching.

    Absolute paths from different machines (e.g. /Users/.../AGENTS.md vs
    /home/ci/.../AGENTS.md) both collapse to AGENTS.md when under cwd.
    """
    root = Path(cwd).resolve()
    p = Path(path)
    try:
        if p.is_absolute():
            return p.resolve().relative_to(root).as_posix()
        return (root / p).resolve().relative_to(root).as_posix()
    except ValueError:
        # Outside the repo — keep POSIX form; will not match relative patterns.
        return p.resolve().as_posix() if p.is_absolute() else p.as_posix()


def _matches(path: str, patterns: tuple[str, ...], *, cwd: str) -> bool:
    norm = _repo_relative(path, cwd)
    for pat in patterns:
        if fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(norm, pat.rstrip("/")):
            return True
    return False


def _extract_paths(payload: dict) -> list[str]:
    tool = payload.get("tool_name") or payload.get("tool") or ""
    inp = payload.get("tool_input") or payload.get("input") or {}
    paths: list[str] = []

    if tool in {"Write", "StrReplace", "Delete"}:
        for key in ("path", "file_path", "target_file"):
            if key in inp and inp[key]:
                paths.append(str(inp[key]))
    return paths


def _log_and_respond(
    permission: str,
    user_message: str,
    agent_message: str,
    *,
    payload: dict,
    event_type: str,
    path: str,
    tool: str,
    rule: dict,
) -> None:
    emit(
        event_type=event_type,
        hook="preToolUse",
        outcome=_OUTCOME[permission],
        target={"path": path, "tool": tool},
        rule=rule,
        messages={"user": user_message, "agent": agent_message},
        payload=payload,
    )
    print(
        json.dumps(
            {
                "permission": permission,
                "user_message": user_message,
                "agent_message": agent_message,
            }
        )
    )


def main() -> int:
    payload = json.load(sys.stdin)
    tool = payload.get("tool_name") or payload.get("tool") or ""
    paths = _extract_paths(payload)
    cwd = payload.get("cwd") or os.getcwd()

    for path in paths:
        if _matches(path, ALWAYS_DENY, cwd=cwd):
            _log_and_respond(
                "deny",
                f"Blocked edit to git internals: {path}",
                "Never modify .git/ contents.",
                payload=payload,
                event_type="file.edit.git_internals",
                path=path,
                tool=tool,
                rule={"source": "protect-paths.py", "id": "git_internals"},
            )
            return 0
        if _matches(path, PROTECTED, cwd=cwd):
            _log_and_respond(
                "ask",
                f"Protected path edit requires approval: {path}",
                "This path affects repo security/CI guardrails. "
                "Only edit when the user explicitly requested it.",
                payload=payload,
                event_type="file.edit.protected_path",
                path=path,
                tool=tool,
                rule={"source": "protect-paths.py", "id": "protected"},
            )
            return 0

    print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
