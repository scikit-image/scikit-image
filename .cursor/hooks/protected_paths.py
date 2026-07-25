"""Single source of truth for hook-enforced protected path globs.

Update this file when changing which paths require user approval to edit.
Keep `.cursor/rules/security.mdc` in sync (summary only). See `.cursor/README.md`.
"""

from __future__ import annotations

# Repo-relative globs: edits require explicit user approval (preToolUse hook).
PROTECTED: tuple[str, ...] = (
    ".cursor/hooks.json",
    ".cursor/hooks/*",
    ".cursor/rules/security.mdc",
    ".cursor/rules/persona.mdc",
    ".cursor/rules/first-contribution.mdc",
    ".cursor/rules/pre-pr-gate.mdc",
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

# Always denied (no approval path).
ALWAYS_DENY: tuple[str, ...] = (
    ".git/*",
    ".git/**",
)

# Entire `.cursor/skills/**` tree is protected via _is_skill_path() in protect-paths.py.
