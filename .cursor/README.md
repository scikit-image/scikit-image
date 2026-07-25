# Cursor onboarding framework (maintainer guide)

This directory configures AI-assisted contribution workflows for scikit-image.
Human entry point: [AGENTS.md](../AGENTS.md) (always applied in Cursor). Authoritative
project policy: [CONTRIBUTING.md](../CONTRIBUTING.md).

## File map

| Path                                                                     | Applies when                         | Purpose                                                 |
| ------------------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------- |
| [AGENTS.md](../AGENTS.md)                                                | Always (workspace)                   | Routing, sources of truth, layout, `spin`, PR/AI policy |
| [rules/persona.mdc](rules/persona.mdc)                                   | Every chat                           | Developer / PM / QA selection                           |
| [rules/security.mdc](rules/security.mdc)                                 | Every chat                           | Git, shell, secrets, protected paths summary            |
| [rules/first-contribution.mdc](rules/first-contribution.mdc)             | First PR / onboarding intents        | Route to `first-contribution` skill                     |
| [rules/skimage-source.mdc](rules/skimage-source.mdc)                     | Editing `src/**`                     | Source, API, deprecation, Cython conventions            |
| [rules/skimage-tests.mdc](rules/skimage-tests.mdc)                       | Editing `tests/**`                   | Test layout, assertions, RNG, threading                 |
| [skills/first-contribution/SKILL.md](skills/first-contribution/SKILL.md) | Invoked for first PR workflow        | Issue pick, branch, verify, handoff                     |
| [skills/persona-developer/SKILL.md](skills/persona-developer/SKILL.md)   | Developer persona                    | Engineering scope and working style                     |
| [skills/persona-pm/SKILL.md](skills/persona-pm/SKILL.md)                 | PM persona                           | Issues, acceptance criteria, no code                    |
| [skills/persona-qa/SKILL.md](skills/persona-qa/SKILL.md)                 | QA persona                           | Test plans, verification checklists                     |
| [hooks.json](hooks.json)                                                 | Agent tool use                       | Shell guardrails, protected-path approval, audit        |
| [hooks/protected_paths.py](hooks/protected_paths.py)                     | Hook enforcement                     | **Canonical** list of protected path globs              |
| [hooks/protect-paths.py](hooks/protect-paths.py)                         | preToolUse (Write/StrReplace/Delete) | Enforces protected paths + skills tree                  |
| [hooks/before-shell.py](hooks/before-shell.py)                           | beforeShellExecution                 | Destructive shell deny-list                             |
| [hooks/before-mcp.py](hooks/before-mcp.py)                               | beforeMCPExecution                   | MCP guardrails                                          |
| [audit/](audit/)                                                         | —                                    | Append-only audit trail (do not edit)                   |

## Layer contract (where to put changes)

| Kind of change                                       | Put it in                                      | Notes                                           |
| ---------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| Project policy (style, testing, deprecations, AI)    | **CONTRIBUTING.md** first                      | Cursor layers link here                         |
| Agent routing, layout table, `spin` commands         | **AGENTS.md**                                  | Keep concise; always applied                    |
| Conventions needed **while editing** `src/` or tests | **skimage-source.mdc** / **skimage-tests.mdc** | Actionable bullets only; link for prose         |
| Step-by-step workflow (first PR, persona behavior)   | **skills/**                                    | Procedures and gates; link AGENTS for reference |
| Persona selection                                    | **persona.mdc** + one skill per role           | Update the routing table in persona.mdc         |
| Paths that require approval to edit                  | **hooks/protected_paths.py**                   | Then sync summary in **security.mdc**           |
| Shell/MCP destructive actions                        | **hooks/\*.py** + **security.mdc**             | Keep behavior and docs aligned                  |

**Link, do not paste:** skills and rules should not duplicate AGENTS.md or CONTRIBUTING.md
paragraphs. Keep procedural checklists in skills; keep edit-time rules in `.mdc` files.

## Enforcement vs guidance (single source per concern)

| Concern                  | Canonical (behavior)                                                                                  | Agent summary                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Edits to high-risk paths | [`hooks/protected_paths.py`](hooks/protected_paths.py) + [`protect-paths.py`](hooks/protect-paths.py) | [`rules/security.mdc`](rules/security.mdc) § Protected paths   |
| Skills tree              | [`repo_paths.is_skill_path()`](hooks/repo_paths.py) in `protect-paths.py`                             | Same § (skills bullet)                                         |
| Destructive / git shell  | [`hooks/before-shell.py`](hooks/before-shell.py) (`DENY_PATTERNS`, `ASK_*`)                           | [`rules/security.mdc`](rules/security.mdc) § Git / Shell       |
| MCP execution            | [`hooks/before-mcp.py`](hooks/before-mcp.py)                                                          | [`rules/security.mdc`](rules/security.mdc) § Network (partial) |

When changing policy: **edit the hook or `protected_paths.py` first**, then update the matching
`security.mdc` section so agents still see intent. Do not add a third copy in skills or AGENTS.md.

## Hook pipeline

Wiring lives in [hooks.json](hooks.json) (`failClosed: true` — hook errors block the action).
All hooks may append to [.cursor/audit/audit.jsonl](audit/audit.jsonl) via [audit_log.emit](hooks/audit_log.py).

| Cursor event           | Script                                     | Tools / scope                   | Outcomes                                                                                               |
| ---------------------- | ------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `preToolUse`           | [protect-paths.py](hooks/protect-paths.py) | `Write`, `StrReplace`, `Delete` | **deny** audit + `.git/`; **ask** on `PROTECTED` globs + `.cursor/skills/**`; else **allow**           |
| `beforeShellExecution` | [before-shell.py](hooks/before-shell.py)   | Shell command string            | **deny** destructive / hook bypass; **ask** on git write, network clients, risky clean; else **allow** |
| `beforeMCPExecution`   | [before-mcp.py](hooks/before-mcp.py)       | MCP tool name                   | **ask** on browser/fetch-like tools; else **allow**                                                    |

Flow for a file edit:

```text
Agent Write/StrReplace/Delete
  → protect-paths.py (paths from tool_input)
  → repo-relative path (repo_paths)
  → audit path? → deny
  → .git/* ? → deny
  → skill path or PROTECTED glob? → ask (user approval card)
  → else allow
```

Flow for shell is independent (same session, separate hook): agent can be **allowed** to edit
normal source files while **ask**ed before `git commit`. Rules in [security.mdc](rules/security.mdc)
describe intent; these scripts enforce it.

Shared helpers:

- [protected_paths.py](hooks/protected_paths.py) — `PROTECTED` / `ALWAYS_DENY` globs
- [repo_paths.py](hooks/repo_paths.py) — normalize paths, `matches()`, `is_skill_path()`

## When the project changes

| Event                                    | Update                                                                            |
| ---------------------------------------- | --------------------------------------------------------------------------------- |
| Stylistic / API / deprecation policy     | CONTRIBUTING.md → then `skimage-source.mdc` if agents need it at edit time        |
| Package layout (`_skimage2`, test trees) | AGENTS.md § Package layout → skim `skimage-source.mdc` / `skimage-tests.mdc`      |
| CI / local test commands                 | AGENTS.md § Build and test → `first-contribution` verify step if workflow changes |
| Beginner issue policy or labels          | `first-contribution/SKILL.md` + `first-contribution.mdc`                          |
| New persona                              | `persona.mdc` table + new `skills/persona-*/SKILL.md` + this README map           |
| New protected path                       | `hooks/protected_paths.py` → summary in `security.mdc` → note here                |

## Changing guardrails safely

1. Edit [hooks/protected_paths.py](hooks/protected_paths.py) for new globs (or `PROTECTED` entries), or edit shell/MCP logic in [before-shell.py](hooks/before-shell.py) / [before-mcp.py](hooks/before-mcp.py) as appropriate.
2. Update the matching summary in [rules/security.mdc](rules/security.mdc) (Protected paths, Git, Shell, or Network) — not a third copy elsewhere. See **Enforcement vs guidance** above.
3. `.cursor/skills/**` is protected via `repo_paths.is_skill_path()` — no need to list each skill file.
4. Test locally: attempt an agent edit to a protected path; expect an approval prompt. Run hook scripts with a sample JSON payload if you change logic.
5. Protected-path edits require **explicit user approval** in chat — maintainers should request that before landing framework PRs.

## Adding a skill

1. Create `.cursor/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`).
2. Write a **procedure** (checklists, gates, handoffs). Link **AGENTS.md** for layout, `spin`, and PR policy.
3. If the skill should auto-route, add or extend a rule (e.g. `first-contribution.mdc`) with a clear `description`.
4. Add a row to the file map table in this README.

## Smoke checks (manual)

After framework changes, in Cursor:

1. New chat without persona → agent asks Developer / PM / QA once.
2. “Help with my first PR” → routes to `first-contribution` skill; prefers `:beginner: Good first issue`.
3. Edit a file under `src/` → `skimage-source` conventions apply (relative imports, etc.).
4. Edit a file under `tests/` → `skimage-tests.mdc` applies (RandomState seed, `testing.assert_*`).
5. Agent tries to edit `AGENTS.md` without user ask → hook requests approval.

## Further reading

- [CONTRIBUTING.md](../CONTRIBUTING.md) — full contributor guide
- [AGENTS.md](../AGENTS.md) — agent and contributor routing
