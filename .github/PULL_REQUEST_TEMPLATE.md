## Description

<!-- If this is a bug-fix or enhancement, state the issue # it closes -->
<!-- If this is a new feature, reference what paper it implements. -->

## Checklist

<!-- It's fine to submit PRs which are a work in progress! -->
<!-- But before they are merged, all PRs should provide: -->

- [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- Gallery example in `./doc/examples` (new features only)
- Benchmark in `./benchmarks`, if your changes aren't covered by an
  existing benchmark
- Unit tests
- Clean style in [the spirit of PEP8](https://www.python.org/dev/peps/pep-0008/)
- Descriptive commit messages (see below)

<!-- For detailed information on these and other aspects see -->
<!-- the scikit-image contribution guidelines. -->
<!-- https://scikit-image.org/docs/dev/contribute.html -->

## For reviewers

<!-- Don't remove the checklist below. -->

- Check that the PR title is short, concise, and will make sense 1 year
  later.
- Check that new functions are imported in corresponding `__init__.py`.
- Check that new features, API changes, and deprecations are mentioned in
  `doc/release/release_dev.rst`.
- There is a bot to help automate backporting a PR to an older branch. For
  example, to backport to v0.19.x after merging, add the following in a PR
  comment: `@meeseeksdev backport to v0.19.x`
- To run benchmarks on a PR, add the `run-benchmark` label. To rerun, the label
  can be removed and then added again. The benchmark output can be checked in
  the "Actions" tab.
