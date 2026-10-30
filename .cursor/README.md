# Platform kit

Cursor kit on this fork: the layout rule, three typed skills, Bugbot review rules, a push guard, and a layout check on every pull request.

## Audiences

| Audience | Uses |
| --- | --- |
| New engineer | `/small-change` |
| PM | `/release-check` and the Notion board |
| QA | `/qa-plan` |
| DevOps | `/release-check` and the pull-request checks |
| Platform team | This guide. `.github/CODEOWNERS` assigns `/.cursor/`, `/.github/scripts/`, and `/.github/workflows/layout-check.yaml` to `@vikasramnani`. |

`/small-change` uses the task typed after the command. That task may be a Notion row the human names, in the workspace the human names. The skill uses that row's Acceptance text as the definition of done and, after the human approves the push, updates only that row (Status "In review", the PR link, a short summary of the test cases). A fork has no maintained task list. `/qa-plan` asks before reading a Notion task linked from the pull request, and uses its Acceptance text only if the human says so.

## Files and who reads them

| What | Who reads it |
| --- | --- |
| `.cursor/rules/repo-layout.mdc` | The agent when `src/**/*.py` or `tests/**/*.py` matches (`alwaysApply: false`). `/small-change` follows it. `.cursor/BUGBOT.md` cites it. |
| `.cursor/BUGBOT.md` | Bugbot, when it reviews a pull request. |
| `.cursor/skills/small-change/SKILL.md` | The agent, only when someone types `/small-change`. |
| `.cursor/skills/qa-plan/SKILL.md` | The agent, only when someone types `/qa-plan`. |
| `.cursor/skills/release-check/SKILL.md` | The agent, only when someone types `/release-check`. |
| `.cursor/hooks/guard-push.sh` (wired from `.cursor/hooks.json`) | Cursor, before a shell command that matches `push`. `failClosed` is true. |
| `.github/scripts/check_layout.py` and `.github/workflows/layout-check.yaml` | GitHub Actions on `pull_request`. The workflow runs the script. |
| Ruleset on `main` | `/release-check` reads it with `gh api repos/<fork>/rules/branches/main`. |

Each skill sets `disable-model-invocation: true`, so it runs only when typed.

## Change a rule or skill

1. Branch from `origin/main` with `--no-track`: `git fetch origin`, then `git switch -c <branch> --no-track origin/main`.
2. Open the pull request on the fork. The fork is `git remote get-url origin`. Never use `upstream`. Never open it against `scikit-image/scikit-image`.
3. Wait for the layout check and for Bugbot.
4. Never open a draft. Drafts get no Bugbot review.

## Test locally

The guard reads a JSON `command` from stdin. Running these does not push:

```shell
printf '%s\n' '{"command":"git push origin my-branch"}' | .cursor/hooks/guard-push.sh
printf '%s\n' '{"command":"git push upstream my-branch"}' | .cursor/hooks/guard-push.sh
```

The `origin` command is allowed. The `upstream` command is denied.

```shell
python3 .github/scripts/check_layout.py
```

The script prints how many test files are under `src/` and exits 1 when that count is not zero.

## Known limits

- The guard over-blocks and assumes `origin`. It allows only `git push` with at most one of `-u`, `--set-upstream`, or `--dry-run`, remote `origin`, and one branch that is not `main`, `master`, or `HEAD`.
- Skills are instructions, not locks. Typing one runs it. The hook denies a matching shell command.
- The improver agent is designed, not built.
