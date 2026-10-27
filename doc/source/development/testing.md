# Testing

The test suite must pass before a pull request can be merged, and
tests should be added to cover all modifications in behavior.

Tests are located in the `tests/` directory.
We also test examples in docstrings of our package (located in `src/`).

(rng-state)=

## Dealing with RNG state

Prefer creating local `np.random.RandomState` instances rather than using the
global NumPy RNG or `np.random.default_rng`. The `RandomState` class is
specifically intended for use in test code and the random number streams it
generates are guaranteed not to change.

Prefer using randomly generated but fixed RNG seeds. The
`np.random.RandomState` generator requires seeds between 0 and 2\*\*32-1, so
you can use the following command-line helper to generate seeds:

```bash
python -c "import random; print(random.randint(0, 2**32-1))"
```

And in your test, create a random number generator like so:

```python
def test_something():
    # hard-code a seed randomly generated while writing the test
    rng = np.random.RandomState(2376609660)
```

If you are comparing with known answers or if your test result otherwise
depends on a specific RNG seed, add a comment to that effect.

For local development we use `spin test` which wraps the
[pytest testing framework](https://docs.pytest.org/en/latest/).
Examples of running `spin test`:

```shell
# All tests
spin test

# Doctests only (tests in docstrings)
spin test -- skimage _skimage2

# Tests inside directory(s)
spin test -- tests/skimage/morphology
spin test -- src/skimage/morphology tests/skimage/morphology

# Tests matching an expression
spin test -- -k threshold

# Combine above with test path to reduce test collection time
spin test -- tests/skimage/filters -k threshold

# Specific test
spin test -- tests/skimage/morphology/test_gray.py::test_3d_fallback_black_tophat
```

:::{tip}
Arguments specified after the `--` are forwarded as
[options to pytest](https://docs.pytest.org/en/stable/reference/reference.html#command-line-flags).
:::

Testing requirements are listed in `requirements/test.txt`.

:::{note}
CI runs only modified tests on pull requests

To keep feedback fast, CI workflows run `spin test --test-modified` on
pull requests, which limits the test run to subpackages that were changed
relative to the base branch. The full test suite still runs on pushes to
`main` and on merge-queue entries.

To force the full suite on a pull request — for example when changes
affect test infrastructure rather than a specific subpackage — add the
**run-all-tests** label to the PR.
:::

You can also use `--test-modified` locally to replicate this CI behaviour:

```shell
# Run tests only for subpackages you have changed relative to your
# upstream tracking branch (e.g. origin/main)
spin test --test-modified

# Specify the base branch explicitly
spin test --test-modified --base-ref main

# Include doctests for modified subpackages
spin test --test-modified --doctest
```

`spin test` automatically detects whether scikit-image is installed as a
wheel (e.g. via `spin install`) or being tested from a meson build
directory. To test a pip-installed wheel, install it first and then run
`spin test` as usual:

```shell
spin install
spin test -- tests/skimage/morphology

# Combine with --test-modified to run only changed subpackages
spin test --test-modified
```

## Warnings during testing phase

By default, warnings raised by the test suite result in errors.
You can switch that behavior off by setting the environment variable
`SKIMAGE_TEST_STRICT_WARNINGS` to `0`.

## Test coverage

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure test coverage run:

```
$ spin test --coverage
```

This will run tests and print a report with one line for each file in
{py:obj}`skimage`, detailing the test coverage:

```
Name                                             Stmts   Exec  Cover   Missing
------------------------------------------------------------------------------
skimage/color/colorconv                             77     77   100%
skimage/filter/__init__                              1      1   100%
...
```

## Multithreaded Testing

scikit-image supports the free-threaded build and we endeavor to ensure the
implementation is thread-safe.

### `pytest-run-parallel`

All tests are automatically run under [pytest-run-parallel](https://github.com/quansight-labs/pytest-run-parallel) in the GitHub actions
CI using a free-threaded interpreter. The `pytest-run-parallel` plugin runs
each test in the entire test suite in a thread pool with many other instances of
the same test, simultaneously. This detects issues caused by use of global state
in the implementation of tested functionality. Since the thread pool runs the
same test several times across threads, the global state is shared, leading to
possible test failures.

Generally, the solution is to avoid using global state. For example, it is best
to avoid using the global NumPy RNGs exposed in the `np.random`
namespace. Instead, you should create and explicitly seed an RNG local to the
test. See {ref}`rng-state` for more detail on dealing with RNGs.

Another example is a test that writes to a file. You should use the pytest
tmp_path fixture rather than manually setting up temporary paths. This will
automatically handle creating a thread-local temporary directory for each worker
thread.

Sometimes using global state in a test is unavoidable. For example, Python
module, function, and type objects are global state. That means tests that
monkeypatch functionality are not thread-safe, however, sometimes monkeypatching
is the most straightforward way to test something. In cases like this, you can
mark a test as thread-unsafe using a pytest mark:

```python
@pytest.mark.thread_unsafe(reason="Test mutates global plugin state")
def test_plugins():
    ...
```

This test will still run under a free-threaded interpreter, but it will execute
on only one thread.

Another reason to mark tests as thread-unsafe is because a test spawns a thread
or process pool. Usually, tests should only use only one level of parallelism
to avoid CPU oversubscription.

Note that `pytest-run-parallel` marks many standard
library functions and built-in pytest fixtures as thread-unsafe
automatically. See the `pytest-run-parallel` [README](https://github.com/Quansight-Labs/pytest-run-parallel#caveats)
for more information.

### Explicitly multithreaded tests

The `threading` module is part of the Python standard library. This means that
all Python classes can be used and mutated freely by a thread pool. If you are
working on the implementation of a mutable object, you should consider whether
it makes sense to allow shared mutation of the object under multiple threads and
whether it is safe to do so. Clearly document the thread safety guarantees of
the object. Also consider adding explciitly multithreaded tests to exercise code
paths that only fire under shared multithreaded use of an object.

See the [Python free-threading guide](https://py-free-threading.github.io) for
more information on [test](https://py-free-threading.github.io/testing/) and
[document](https://py-free-threading.github.io/documentation-principles/)
Python projects for multithreaded use.
