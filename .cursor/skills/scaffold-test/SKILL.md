---
name: scaffold-test
description: >-
  Scaffold and strengthen scikit-image tests: pick test tree, read peer tests,
  weak-test checklist. Use when adding or updating tests, improving test coverage,
  or writing tests for a function or bugfix.
---

# Scaffold tests

Procedural guide for **writing** tests. Edit-time conventions (assertions, RNG, threading) live in
[`.cursor/rules/skimage-tests.mdc`](../../rules/skimage-tests.mdc) — follow that rule while editing
`tests/**`; do not duplicate it here.

## When this applies

- Any change that affects behavior under `src/`
- First contributions, bugfixes, and feature work
- Doc-only or comment-only changes: skip unless the task asks for tests

## Scaffold procedure

1. **Pick the test tree** — match [AGENTS.md](../../../AGENTS.md) § Package layout:

   - Changed `_skimage2` implementation → prefer `tests/skimage2/<subpackage>/`
   - Changed public `skimage` wrapper only → `tests/skimage/<subpackage>/`
   - Mirror the module path (`filters/thresholding.py` → `test_thresholding.py` or existing file in that folder)

2. **Read peers before writing** — open 1–2 tests in the same directory for the same kind of change (new function vs bugfix). Copy structure: imports, fixtures, class vs function tests, parametrization.

3. **Place the test** — extend an existing `test_*.py` when the change is a small addition; new file only when the module has no tests yet or the task asks for it.

4. **Import the symbol under test** — use the same import style as neighboring tests in that file (often `from _skimage2...` in `tests/skimage2/`).

5. **Run narrow tests immediately** — see **AGENTS.md** § Build and test:

   ```bash
   spin test -- tests/skimage2/<subpackage>/test_<module>.py::test_<name>
   spin test -- tests/skimage2/<subpackage>/
   ```

## Weak-test checklist (before pre-PR verify)

Self-check every new or changed test:

- [ ] Asserts **outputs** (values, shape, dtype), not only that the call completed
- [ ] Covers the **bug or feature** from the issue or task (regression case if fixing a bug)
- [ ] Uses **fixed-seed** `np.random.RandomState(...)` when random data is needed (see `skimage-tests.mdc`)
- [ ] Uses project helpers where peers do (`testing.assert_equal`, `assert_allclose`, `expected_warnings`)
- [ ] No unnecessary global state; use `tmp_path` for files; `@pytest.mark.thread_unsafe` if mutating plugins/globals
- [ ] If the change affects warnings or deprecations, tests use `expected_warnings` / stacklevel checks as peers do

If behavior changed under `src/` but no test file was added or updated, stop and add tests before handoff.

## Handoff note

In the PR summary, list test file(s) and the `spin test` command(s) you ran.
