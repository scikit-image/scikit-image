# scikit-image — agent guide

Concise routing for AI agents. Full details: [CONTRIBUTING.md](CONTRIBUTING.md).

## Sources of truth

Derive project context **only** from:

1. This repository (source, tests, docs, config, CI)
2. [scikit-image stable documentation](https://scikit-image.org/docs/stable/)

Do **not** guess API behavior, conventions, or implementation details. Do **not** rely on external sources (Stack Overflow, blog posts, other doc versions, training-data assumptions).

If the repo and stable docs do not answer a question, state what you checked (files, symbols, doc sections), say what is unknown, and ask — do not infer or invent an answer.

## Read before edit

Before changing a module, read:

- The file(s) being edited
- A similar existing implementation in the same subpackage
- Existing tests for that module

## Scope

- Change only what the task requires
- Do not refactor, reformat, or rename unrelated code
- Do not add dependencies without explicit approval
- Match surrounding naming, imports, and docstring style

## Security

Follow `.cursor/rules/security.mdc` for git, shell, secrets, and protected-path rules.
Project hooks in `.cursor/hooks.json` enforce the destructive-action deny-list.

## Package layout

| Path              | Role                                                                           |
| ----------------- | ------------------------------------------------------------------------------ |
| `src/_skimage2/`  | Implementations (Python + Cython). Put new algorithm code here.                |
| `src/skimage/`    | Public v1 API: thin wrappers, migration warnings, re-exports from `_skimage2`. |
| `tests/skimage/`  | Tests for the public `skimage` API.                                            |
| `tests/skimage2/` | Tests for `_skimage2` implementations.                                         |
| `doc/examples/`   | Sphinx-Gallery examples (required for new user-facing features).               |

**Rule of thumb:** implement in `_skimage2`, expose via `skimage` when maintaining v1 compatibility. See existing wrappers (e.g. `src/skimage/morphology/gray.py`) and `src/skimage/_migration.py`.

## Build and test

Use `spin`, not raw `pytest` or `meson` directly:

```bash
spin install -v              # editable dev install (recommended)
spin build --clean           # after adding/removing source or Cython files
spin test                    # full suite
spin test --test-modified    # matches PR CI (changed subpackages only)
spin test -- tests/skimage/morphology -k threshold
spin docs                    # build documentation
```

- After adding/removing files under `src/`, update the relevant `meson.build` and rebuild.
- Do **not** pass `src/` as a test path; use package names (`skimage.io`) or `tests/skimage/...`.
- Install pre-commit hooks: `pre-commit install` (runs ruff, cython-lint, formatters).

## Verification

Before finishing a task:

- Run the narrowest relevant check (`spin test -- …`, `spin build` after Cython/meson changes)
- Before opening a PR, run `./tools/cursor/validate-contribution.sh` (see `.cursor/skills/pre-pr-gate/SKILL.md`)
- When adding or strengthening tests, use `.cursor/skills/scaffold-test/SKILL.md` (conventions while editing: `.cursor/rules/skimage-tests.mdc`)
- Report what you ran and the outcome
- Do not claim tests pass unless you ran them

## Pull requests

- Disclose all generative tools used; authors must understand every line changed. See [AI policy](CONTRIBUTING.md#ai-policy).
- CI needs a **category label** on new PRs or checks fail until a maintainer adds one.
- Optional `release-note` block in the PR description for non-trivial changes (see [PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)).
- API deprecations: follow [deprecation cycle](CONTRIBUTING.md#deprecation-cycle) and add an entry to `TODO.txt`.
- Do not merge PRs; two core approvals required for code changes.

## Further reading

- [Stylistic guidelines](CONTRIBUTING.md#stylistic-guidelines) — imports, dtypes, coordinates, Cython
- [Testing](CONTRIBUTING.md#testing) — coverage, RNG, multithreading
- [Gallery](CONTRIBUTING.md#gallery) — new feature examples
- [Benchmarks](CONTRIBUTING.md#benchmarks) — performance PRs
- [Cursor framework maintainer guide](.cursor/README.md) — rules, skills, hooks, and how to extend `.cursor/`
- [Pre-PR validate script](tools/cursor/README.md) — `validate-contribution.sh` options and heuristics
