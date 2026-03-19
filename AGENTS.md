# Instructions for AI Agents

This file contains instructions for AI coding agents (e.g., Claude, Gemini,
GitHub Copilot, Cursor) acting on this repository.

## Tool Disclosure (required)

scikit-image requires that every pull request discloses all automated tools used.
Include a `## Tool Disclosure` section in the PR body listing every non-editor
tool used. For example:

> Gemini CLI (Google), Claude Code (Anthropic), GitHub Copilot, or custom scripts (e.g., sed).

**Note:** If custom scripts or `sed` commands were used, you must include the script
content or the exact command applied.

If no automated tools were used beyond a standard editor, write:
> No automated tools used.

This section is required regardless of how large or small the change is.

## Co-author Attribution (required)

You MUST include a `Co-authored-by` trailer in your commit messages or the PR
description for yourself. This is a project policy regardless of your local
configuration. For example:

> Co-authored-by: Claude <claude@anthropic.com>
> Co-authored-by: Gemini <gemini@google.com>

## General contribution guidelines

- Follow the [contribution guide](https://scikit-image.org/docs/dev/development/contribute.html).
- Run `pre-commit` before committing to check formatting and style.
- Add or update unit tests for every code change.
- Add or update docstrings (NumPy style) for every public API change.
- Reference related issues or PRs in the PR description.
