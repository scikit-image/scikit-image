#!/usr/bin/env python3
"""Pre-commit hook: reject obvious secrets in staged text files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# High-confidence patterns only (avoid false positives in docs/tests).
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "generic API key assignment",
        re.compile(
            r"(?i)(api[_-]?key|secret[_-]?key|password|token)\s*[:=]\s*"
            r"['\"][^'\"]{8,}['\"]"
        ),
    ),
    (
        "private key block",
        re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
]

ALLOWLIST_SUBSTRINGS = (
    "example",
    "placeholder",
    "your-token-here",
    "xxx",
    "REDACTED",
)


def main(argv: list[str]) -> int:
    failed = False
    for raw in argv[1:]:
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, pat in PATTERNS:
            m = pat.search(text)
            if m and not any(s in m.group(0) for s in ALLOWLIST_SUBSTRINGS):
                snippet = m.group(0)
                if len(snippet) > 40:
                    snippet = snippet[:40] + "…"
                print(f"{path}: possible {label}: {snippet}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
