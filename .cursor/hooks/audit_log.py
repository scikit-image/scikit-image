#!/usr/bin/env python3
"""Append-only audit log for Cursor hook decisions."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
LOG_DIR = Path(".cursor/audit")
LOG_FILE = LOG_DIR / "audit.jsonl"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git_branch(cwd: Path) -> str | None:
    head = cwd / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if ref.startswith("ref: "):
        return ref.removeprefix("ref: refs/heads/")
    return ref[:12]


def _actor(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": payload.get("actor_kind", "agent"),
        "session_id": payload.get("session_id") or payload.get("sessionId"),
        "conversation_id": payload.get("conversation_id")
        or payload.get("conversationId"),
    }


def _context(payload: dict[str, Any]) -> dict[str, Any]:
    cwd = Path(payload.get("cwd") or os.getcwd())
    ctx: dict[str, Any] = {"cwd": str(cwd.resolve())}
    branch = _git_branch(cwd)
    if branch:
        ctx["branch"] = branch
    return ctx


def emit(
    *,
    event_type: str,
    hook: str,
    outcome: str,
    target: dict[str, Any],
    rule: dict[str, Any] | None = None,
    messages: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Best-effort append; never raise into the hook decision path."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "timestamp": _utc_now(),
            "event_type": event_type,
            "hook": hook,
            "actor": _actor(payload or {}),
            "outcome": outcome,
            "target": target,
        }
        if rule:
            record["rule"] = rule
        if messages:
            record["messages"] = messages
        if payload:
            record["context"] = _context(payload)

        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
