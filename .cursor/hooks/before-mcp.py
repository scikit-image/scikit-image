#!/usr/bin/env python3
"""Prompt before MCP tools that make direct network calls (Cursor beforeMCPExecution)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_log import emit

# Tool name substrings for fetch/browser/network MCP tools.
DIRECT_NETWORK_MCP = (
    "browser_",
    "webfetch",
    "fetch",
    "navigate",
)


def main() -> int:
    payload = json.load(sys.stdin)
    tool = payload.get("tool_name") or payload.get("tool") or ""
    tool_lower = tool.lower()
    target = {"tool": tool, "server": payload.get("server")}

    if any(marker in tool_lower for marker in DIRECT_NETWORK_MCP):
        user_msg = f"Direct network MCP call requires approval: {tool}"
        agent_msg = (
            "Only use browser or fetch MCP tools when the user explicitly asked."
        )
        emit(
            event_type="mcp.network.approval_requested",
            hook="beforeMCPExecution",
            outcome="approval_requested",
            target=target,
            messages={"user": user_msg, "agent": agent_msg},
            payload=payload,
        )
        print(
            json.dumps(
                {
                    "permission": "ask",
                    "user_message": user_msg,
                    "agent_message": agent_msg,
                }
            )
        )
        return 0

    print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
