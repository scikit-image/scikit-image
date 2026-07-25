# Cursor onboarding framework (maintainer guide)

This directory configures AI-assisted contribution workflows for scikit-image.
Human entry point: [AGENTS.md](../AGENTS.md) (always applied in Cursor). Authoritative
project policy: [CONTRIBUTING.md](../CONTRIBUTING.md).

## How pieces fit together

```text
AGENTS.md (always) → persona.mdc → persona skill (Developer / PM / QA)
                   → routing rules (first-contribution, pre-pr-gate) → workflow skills
                   → skimage-source / skimage-tests.mdc (when editing src/ or tests/)
                   → security.mdc (always) + hooks (enforce on shell / file writes)
```

**Contributor scripts** (not under `.cursor/`, editable without hook approval on each file):
[tools/cursor/validate-contribution.sh](../tools/cursor/validate-contribution.sh) — see
[tools/cursor/README.md](../tools/cursor/README.md).

## File map

### Entry and policy

| Path                                  | Applies when            | Purpose                                                 |
| ------------------------------------- | ----------------------- | ------------------------------------------------------- |
| [AGENTS.md](../AGENTS.md)             | Always (workspace)      | Routing, sources of truth, layout, `spin`, PR/AI policy |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Human / agent reference | Authoritative contributor policy                        |

### Always-on rules

| Path                                     | Purpose                                              |
| ---------------------------------------- | ---------------------------------------------------- |
| [rules/persona.mdc](rules/persona.mdc)   | Developer / PM / QA selection (one persona per chat) |
| [rules/security.mdc](rules/security.mdc) | Git, shell, secrets, protected paths summary         |

### Routing rules (intent → skill)

| Path                                                         | Triggers (rule `description`)                | Skill                                                    |
| ------------------------------------------------------------ | -------------------------------------------- | -------------------------------------------------------- |
| [rules/first-contribution.mdc](rules/first-contribution.mdc) | First PR, starter issue, onboarding          | [first-contribution](skills/first-contribution/SKILL.md) |
| [rules/pre-pr-gate.mdc](rules/pre-pr-gate.mdc)               | Ready for PR, validate branch, pre-PR checks | [pre-pr-gate](skills/pre-pr-gate/SKILL.md)               |

Requires **Developer** persona for first contribution (see [persona.mdc](rules/persona.mdc) § Conflicts).

### File-scoped rules (edit-time conventions)

| Path                                                 | Globs      | Purpose                          |
| ---------------------------------------------------- | ---------- | -------------------------------- |
| [rules/skimage-source.mdc](rules/skimage-source.mdc) | `src/**`   | Source, API, deprecation, Cython |
| [rules/skimage-tests.mdc](rules/skimage-tests.mdc)   | `tests/**` | Assertions, RNG, threading       |

These rules are **not** in [protected_paths.py](hooks/protected_paths.py); agents may edit them with normal approval flow, but [CODEOWNERS](../CODEOWNERS) still applies on PRs.

### Skills

| Path                                                                     | Purpose                                                                     |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| [skills/first-contribution/SKILL.md](skills/first-contribution/SKILL.md) | Issue pick, branch, implement, verify via pre-pr-gate                       |
| [skills/pre-pr-gate/SKILL.md](skills/pre-pr-gate/SKILL.md)               | Run validate script + PR metadata checklist                                 |
| [skills/scaffold-test/SKILL.md](skills/scaffold-test/SKILL.md)           | Scaffold tests + weak-test checklist (links skimage-tests.mdc)              |
| [skills/persona-developer/SKILL.md](skills/persona-developer/SKILL.md)   | Engineering scope; points to first-contribution, pre-pr-gate, scaffold-test |
| [skills/persona-pm/SKILL.md](skills/persona-pm/SKILL.md)                 | Issues, acceptance criteria, no code                                        |
| [skills/persona-qa/SKILL.md](skills/persona-qa/SKILL.md)                 | Test plans, verification checklists                                         |

Entire `.cursor/skills/**` tree is hook-protected (approval to edit).

### Tools and hooks

| Path                                                                              | Purpose                                                         |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [tools/cursor/validate-contribution.sh](../tools/cursor/validate-contribution.sh) | Heuristics + pre-commit + `spin test --test-modified`           |
| [tools/cursor/README.md](../tools/cursor/README.md)                               | Script options and examples                                     |
| [hooks.json](hooks.json)                                                          | Registers shell, preToolUse, MCP hooks                          |
| [hooks/protected_paths.py](hooks/protected_paths.py)                              | **Canonical** `PROTECTED` / `ALWAYS_DENY` globs                 |
| [hooks/protect-paths.py](hooks/protect-paths.py)                                  | preToolUse: Write / StrReplace / Delete; path normalize + globs |
| [hooks/before-shell.py](hooks/before-shell.py)                                    | beforeShellExecution                                            |
| [hooks/before-mcp.py](hooks/before-mcp.py)                                        | beforeMCPExecution                                              |
| [hooks/audit_log.py](hooks/audit_log.py)                                          | Append-only [audit/](audit/audit.jsonl)                         |
| [audit/](audit/)                                                                  | Do not modify (hook deny)                                       |

## Layer contract (where to put changes)

| Kind of change                                    | Put it in                                      | Notes                                        |
| ------------------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| Project policy (style, testing, deprecations, AI) | **CONTRIBUTING.md** first                      | Cursor layers link here                      |
| Agent routing, layout, `spin`, verification       | **AGENTS.md**                                  | Always applied; link skills/scripts          |
| Pre-PR commands and heuristics                    | **tools/cursor/validate-contribution.sh**      | Document flags in **tools/cursor/README.md** |
| Conventions while editing `src/` or `tests/`      | **skimage-source.mdc** / **skimage-tests.mdc** | Actionable bullets; link CONTRIBUTING        |
| Workflow procedures (first PR, pre-PR, tests)     | **skills/**                                    | Link AGENTS; use scaffold-test + pre-pr-gate |
| Auto-route to a skill                             | **routing `.mdc`** + skill `description`       | e.g. first-contribution, pre-pr-gate         |
| Persona behavior                                  | **persona.mdc** + **persona-\*/SKILL.md**      | Update routing table in persona.mdc          |
| Protected path globs                              | **hooks/protected_paths.py**                   | Sync summary in **security.mdc**             |
| Shell / MCP enforcement                           | **hooks/\*.py** + **security.mdc**             | Hooks enforce; rule summarizes               |

**Link, do not paste:** skills and rules should not duplicate AGENTS.md or CONTRIBUTING.md
paragraphs. Keep procedural checklists in skills; keep edit-time rules in `.mdc` files.

## Enforcement vs guidance (single source per concern)

| Concern                                            | Canonical (behavior)                                                                        | Agent summary                                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Edits to high-risk paths                           | [protected_paths.py](hooks/protected_paths.py) + [protect-paths.py](hooks/protect-paths.py) | [security.mdc](rules/security.mdc) § Protected paths               |
| Skills tree                                        | `_is_skill_path()` in [protect-paths.py](hooks/protect-paths.py)                            | Same §                                                             |
| Destructive / git shell                            | [before-shell.py](hooks/before-shell.py)                                                    | [security.mdc](rules/security.mdc) § Git / Shell                   |
| MCP execution                                      | [before-mcp.py](hooks/before-mcp.py)                                                        | [security.mdc](rules/security.mdc) § Network                       |
| Pre-PR test/src pairing, TODO.txt for deprecations | [validate-contribution.sh](../tools/cursor/validate-contribution.sh)                        | [pre-pr-gate](skills/pre-pr-gate/SKILL.md) + AGENTS § Verification |

When changing policy: **edit the hook or script first**, then update the matching skill or
`security.mdc` section. Do not add a third copy in AGENTS.md unless it is a one-line pointer.

## Hook pipeline

Wiring lives in [hooks.json](hooks.json) (`failClosed: true` — hook errors block the action).
All hooks may append to [audit/audit.jsonl](audit/audit.jsonl) via [audit_log.emit](hooks/audit_log.py).

| Cursor event           | Script                                     | Tools / scope                   | Outcomes                                                                  |
| ---------------------- | ------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------- |
| `preToolUse`           | [protect-paths.py](hooks/protect-paths.py) | `Write`, `StrReplace`, `Delete` | **deny** audit + `.git/`; **ask** on `PROTECTED` + skills; else **allow** |
| `beforeShellExecution` | [before-shell.py](hooks/before-shell.py)   | Shell command                   | **deny** destructive / bypass; **ask** git write, network, risky clean    |
| `beforeMCPExecution`   | [before-mcp.py](hooks/before-mcp.py)       | MCP tool name                   | **ask** browser/fetch-like tools; else **allow**                          |

Flow for a file edit:

```text
Agent Write/StrReplace/Delete
  → protect-paths.py (paths from tool_input)
  → repo-relative path (_repo_relative in protect-paths.py)
  → audit path? → deny
  → .git/* ? → deny
  → skill path or PROTECTED glob? → ask (user approval card)
  → else allow
```

Shell hooks run independently in the same session. [security.mdc](rules/security.mdc) describes
intent; hook scripts enforce it.

## When the project changes

| Event                                | Update                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| Stylistic / API / deprecation policy | CONTRIBUTING.md → skim skimage-source.mdc                                                   |
| Package layout                       | AGENTS.md § Package layout → skim skimage-tests.mdc                                         |
| CI / local test commands             | AGENTS.md § Verification → validate-contribution.sh → pre-pr-gate skill                     |
| validate-contribution heuristics     | validate-contribution.sh → pre-pr-gate skill § 3 (document flags in tools/cursor/README.md) |
| Test writing workflow                | scaffold-test skill + skimage-tests.mdc                                                     |
| Beginner issue policy                | first-contribution skill + first-contribution.mdc                                           |
| Pre-PR routing wording               | pre-pr-gate.mdc `description` + pre-pr-gate skill                                           |
| New persona                          | persona.mdc table + new persona skill + this README                                         |
| New protected path                   | protected_paths.py → security.mdc → note here                                               |

## Changing guardrails safely

1. Edit [protected_paths.py](hooks/protected_paths.py) or shell/MCP hooks as appropriate.
2. Update the matching section in [security.mdc](rules/security.mdc). See **Enforcement vs guidance**.
3. `.cursor/skills/**` is protected via `is_skill_path()` — no per-file globs needed.
4. Smoke-test: edit AGENTS.md without user ask → approval prompt; run validate-contribution.sh after script changes.
5. Framework PRs that touch protected paths need **explicit user approval** in Cursor for agent edits.

## Adding a skill

1. Create `.cursor/skills/<name>/SKILL.md` with `name` and `description` in frontmatter.
2. Write a **procedure** (checklists, gates). Link **AGENTS.md** and file-scoped rules as needed.
3. Optional: add a **routing rule** `.cursor/rules/<name>.mdc` if intent-based routing helps (mirror first-contribution / pre-pr-gate).
4. If the skill is governance-critical, consider adding its routing rule to **protected_paths.py** (routing rules: first-contribution.mdc, pre-pr-gate.mdc).
5. Update this README file map and **When the project changes**.

## Smoke checks (manual)

After framework changes, in Cursor:

1. New chat → agent asks **Developer · PM · QA** once.
2. “Help with my first PR” → Developer + first-contribution skill; prefers `:beginner: Good first issue`.
3. “Ready to open a PR” / “validate my branch” → pre-pr-gate skill; runs `./tools/cursor/validate-contribution.sh`.
4. “Add tests for …” → scaffold-test skill; editing `tests/**` applies skimage-tests.mdc.
5. Edit under `src/**` → skimage-source conventions (relative imports, etc.).
6. Agent edits `AGENTS.md` without user ask → hook **ask** approval.
7. From repo root, `./tools/cursor/validate-contribution.sh` exits 0 on a clean branch (or fails with clear heuristic messages).

## Further reading

- [CONTRIBUTING.md](../CONTRIBUTING.md) — full contributor guide
- [AGENTS.md](../AGENTS.md) — agent and contributor routing
- [tools/cursor/README.md](../tools/cursor/README.md) — validate-contribution.sh options
