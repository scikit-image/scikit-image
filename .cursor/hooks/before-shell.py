#!/usr/bin/env python3
"""Gate destructive and git-mutating shell commands (Cursor beforeShellExecution)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_log import emit

# Match git even when prefixed (env vars, cd &&, /usr/bin/git, git -C).
_GIT = r"(?:^|[;&|]\s*|\b)(?:/\S+/)?git(?:\s+-[CKc]\s+\S+|\s+--[\w-]+)*\s+"

DENY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # --force / --force-with-lease anywhere after push
    ("force push", re.compile(_GIT + r"push\b[^;&|\n]*--force(?:-with-lease)?\b")),
    # -f inside a short-flag cluster (-f, -fu, -uf, …)
    ("force push", re.compile(_GIT + r"push\b[^;&|\n]*\s-[a-zA-Z]*f[a-zA-Z]*\b")),
    ("hard reset", re.compile(r"git reset --hard")),
    ("git clean", re.compile(r"git clean -[a-z]*x")),
    ("checkout discard", re.compile(r"git checkout -- \.")),
    ("delete main", re.compile(r"git branch -[dD] main")),
    ("delete master", re.compile(r"git branch -[dD] master")),
    ("git config", re.compile(r"git config ")),
    ("rm -rf /", re.compile(r"rm -rf /")),
    ("rm -rf .", re.compile(r"rm -rf \.")),
    ("rm -rf ..", re.compile(r"rm -rf \.\.")),
    ("chmod 777", re.compile(r"chmod 777")),
    (
        "curl pipe sh",
        re.compile(r"curl\b.*\|\s*(?:sudo\s+)?(?:(?:/usr)?/bin/)?(?:ba|z|da)?sh\b"),
    ),
    (
        "wget pipe sh",
        re.compile(r"wget\b.*\|\s*(?:sudo\s+)?(?:(?:/usr)?/bin/)?(?:ba|z|da)?sh\b"),
    ),
]

BYPASS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("hook bypass", re.compile(r"--no-verify\b")),
    ("hook bypass", re.compile(r"--no-gpg-sign\b")),
]

ASK_GIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(_GIT + r"commit\b"),
    re.compile(_GIT + r"push\b"),
    re.compile(_GIT + r"tag\b"),
    re.compile(_GIT + r"revert\b"),
    re.compile(_GIT + r"rebase\b"),
    re.compile(_GIT + r"merge\b"),
    re.compile(_GIT + r"cherry-pick\b"),
    re.compile(_GIT + r"stash\s+(drop|clear|pop)\b"),
]

ASK_CLEAN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"spin build.*--clean"),
    re.compile(r"git clean"),
]

DIRECT_NETWORK_ASK: list[tuple[str, re.Pattern[str]]] = [
    ("http client", re.compile(r"\b(curl|wget|httpie|httpx)\b")),
]

_OUTCOME = {"allow": "allowed", "deny": "denied", "ask": "approval_requested"}


def _respond(
    permission: str,
    user_message: str,
    agent_message: str,
    *,
    payload: dict,
    event_type: str,
    target: dict,
    rule: dict | None = None,
) -> None:
    emit(
        event_type=event_type,
        hook="beforeShellExecution",
        outcome=_OUTCOME[permission],
        target=target,
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
    command = payload.get("command") or payload.get("cmd") or ""
    target = {"command": command}

    for label, pattern in BYPASS_PATTERNS:
        if pattern.search(command):
            _respond(
                "deny",
                f"Blocked hook bypass ({label}): {command}",
                "Never skip git hooks (--no-verify) or GPG signing "
                "(--no-gpg-sign). Fix the underlying issue instead.",
                payload=payload,
                event_type="guardrail_bypass.attempted",
                target=target,
                rule={
                    "source": "before-shell.py",
                    "id": "hook_bypass",
                    "label": label,
                },
            )
            return 0

    for label, pattern in DENY_PATTERNS:
        if pattern.search(command):
            _respond(
                "deny",
                f"Blocked destructive command ({label}): {command}",
                "This command matches a repo deny-list. Do not retry without explicit "
                "user approval and a safer alternative.",
                payload=payload,
                event_type="shell.command.denied",
                target=target,
                rule={
                    "source": "before-shell.py",
                    "id": label.replace(" ", "_"),
                    "label": label,
                },
            )
            return 0

    for pattern in ASK_GIT_PATTERNS:
        if pattern.search(command):
            _respond(
                "ask",
                f"Git mutation requires approval: {command}",
                "Only run git write commands when the user explicitly asked. "
                "Never use --no-verify.",
                payload=payload,
                event_type="shell.git_mutation.approval_requested",
                target=target,
            )
            return 0

    for label, pattern in DIRECT_NETWORK_ASK:
        if pattern.search(command):
            _respond(
                "ask",
                f"Direct network call requires approval ({label}): {command}",
                "Only run direct network commands when the user explicitly asked.",
                payload=payload,
                event_type="shell.network.approval_requested",
                target=target,
                rule={
                    "source": "before-shell.py",
                    "id": label.replace(" ", "_"),
                    "label": label,
                },
            )
            return 0

    for pattern in ASK_CLEAN_PATTERNS:
        if pattern.search(command):
            _respond(
                "ask",
                "Clean/build command may delete generated artifacts. Approve?",
                "Confirm this clean is intentional before proceeding.",
                payload=payload,
                event_type="shell.clean.approval_requested",
                target=target,
            )
            return 0

    print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
