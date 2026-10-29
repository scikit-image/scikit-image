# Bugbot PR review rules

Flag these when reviewing pull requests. Reasons cite this repo’s layout and docs.

1. **New logic in a pure re-export/shim under `src/skimage/` or `src/skimage2/`** — Flag only when the file solely imports from `_skimage2` and defines nothing of its own; put new logic in `src/_skimage2/` instead. Do not flag files that already contain real logic (e.g. `future/trainable_segmentation.py`, `util/_backends.py`) (`.cursor/rules/repo-layout.mdc`).

2. **Tests under `src/`** — Tests live only in top-level `tests/` (`tests/skimage/` for legacy API, `tests/skimage2/` for `skimage2` / `_skimage2` internals) (`.cursor/rules/repo-layout.mdc`).

3. **Public signature/default change without deprecation helpers** — Use `deprecate_parameter` / `deprecate_func` from `src/_skimage2/_shared/utils.py` when a deprecation cycle is required (`doc/source/development/deprecations.md`).

4. **Behaviour change with no test** — The suite must cover modifications in behavior (`doc/source/development/testing.md`; `doc/source/development/guidelines.md`).
