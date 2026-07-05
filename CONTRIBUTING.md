(howto-contribute)=

# How to contribute to scikit-image

Developing open source software as part of a community is fun, and
often quite educational!

We coordinate our work using GitHub, where you can find lists of [open
issues](https://github.com/scikit-image/scikit-image/issues?q=is%3Aopen)
and [new feature requests](https://github.com/scikit-image/scikit-image/labels/%3Apray%3A%20Feature%20request).

To follow along with discussions, or to get in touch with the
developer team, please join us on the [scikit-image developer forum](https://discuss.scientific-python.org/c/contributor/skimage) and
the [Zulip chat](https://skimage.zulipchat.com/).

Please post questions to these public forums (rather than contacting
developers directly); that way, everyone can benefit from the answers,
and developers can answer according to their availability. Don't feel
shy, the team is very friendly!

```{contents}
:local: true
```

## Development process

The following is a brief overview about how changes to source code and documentation
can be contributed to scikit-image.

1. If you are a first-time contributor:

   - Go to [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) and click the
     "fork" button to create your own copy of the project.

   - [Set up GitHub SSH authentication](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).

   - Clone (download) the repository with the project source on your local computer:

     ```
     git clone --origin upstream git@github.com:scikit-image/scikit-image
     ```

   - Change into the root directory of the cloned repository:

     ```
     cd scikit-image
     ```

   - Add your fork as a
     [remote repository](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)
     that you will interact with.

     Assuming a GitHub username of `codemonkey`:

     ```
     git remote add codemonkey git@github.com:codemonkey/scikit-image
     git fetch codemonkey
     ```

   - You now have two remote repositories:

     - `upstream`, which refers to the `scikit-image` project repository, and
     - `codemonkey`, which refers to your personal fork.

   - Next, {ref}`set up your build environment <build-env-setup>`.

   - Finally, we recommend that you use our pre-commit hook, which runs code
     checkers and formatters each time you do a `git commit`:

     ```
     pip install pre-commit
     pre-commit install
     ```

2. Develop your contribution:

   - Pull the latest changes from the project:

     ```
     git switch main
     git fetch upstream main
     git merge upstream/main
     ```

   - Create a branch for the feature you want to work on. Use a sensible name,
     such as 'transform-speedups':

     ```
     git switch -c transform-speedups
     ```

   - Commit locally as you progress (with `git add` and `git commit`).
     Please write [good commit messages](https://vxlabs.com/software-development-handbook/#good-commit-messages).

   - It is a good idea to read our {ref}`guidelines` at this point.
     While we don't require a contribution to meet every guideline from the
     start, they will come up during review.

3. To submit your contribution:

   - Push your changes back to your fork on GitHub:

     ```
     git push codemonkey transform-speedups
     ```

     A message will be displayed with a URL to open in your browser to create a
     pull request (PR).

   - Before submitting the pull request:

     - Use a concise, descriptive title
     - Describe and link relevant context in the description
     - Disclose all _generative_ tools (AI, LLMs, agents) that you used, see our
       {ref}`ai-policy` for details.

   :::{tip}
   If you get stuck, reach out to us on
   [our Zulip chat](https://skimage.zulipchat.com/).
   :::

4. Review process:

   - Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your pull request (PR) to help
     you improve its implementation, documentation, and style. Every single
     developer working on the project has their code reviewed, and we've come
     to see it as a friendly conversation from which we all learn and the
     overall code quality benefits. Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of the project, not to criticize (we are, after all, very grateful for the
     time you're putting in!).

   - To update your pull request, make your changes on your local repository
     and commit. As soon as those changes are pushed up (to the same branch as
     before) the pull request will update automatically.

   - Continuous integration (CI) services are triggered after each
     pull request submission to build the package, run unit tests, and
     check the coding style and formatting of your branch. The tests
     must pass before your PR can be merged. If CI fails, you can find
     out why by clicking on the "failed" icon (red cross) and
     inspecting the build and test logs.

     :::{note}
     PR labeling

     CI will always fail on new PRs, until a maintainer adds a
     suitable category label.
     :::

   - A pull request must be approved by two core team members before merging.

(documenting-changes)=

5. Document changes

   If your change introduces a deprecation, add a reminder to `TODO.txt`
   for the team to remove the deprecated functionality in the future.

   scikit-image uses [changelist](https://github.com/scientific-python/changelist)
   to generate a list of release notes automatically from pull requests. By
   default, changelist will use the title of a pull request and its GitHub
   labels to sort it into the appropriate section. However, for more complex
   changes, we encourage you to describe them in more detail using the
   `release-note` code block within the pull request description; e.g.:

   ````
   ```release-note
   Remove the deprecated function `skimage.color.blue`. Blend
   `skimage.color.cyan` and `skimage.color.magenta` instead.
   ```
   ````

   You can refer to {doc}`/release_notes/index` for examples and to
   [changelist's documentation](https://github.com/scientific-python/changelist)
   for more details.

:::{note}
To reviewers: if it is not obvious from the PR description, make sure that
the reason and context for a change are described in the merge message.
:::

## Divergence between `upstream main` and your feature branch

If GitHub indicates that the branch of your PR can no longer
be merged automatically, merge the main branch into yours:

```
git fetch upstream main
git merge upstream/main
```

If any conflicts occur, they need to be [fixed before continuing](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

[We recommend](https://github.com/stefanv/git-tools?tab=readme-ov-file#conflict-diff-display) setting:

```
git config --global merge.conflictstyle zdiff3
```

to make conflict markers easier to read.

An alternative to merging is to rebase your branch—but we squash and merge all
PRs anyway, so we don't mind merge commits.

(ai-policy)=

## AI Policy

Regardless of how a PR was produced, scikit-image requires that
authors illustrate a thorough understanding of any proposed changes.
You **must review such code line-by-line**—it is **your
responsibility** to ensure that it is correct, and that it does not
breach copyright. You should expect the team to ask questions about
your work.

scikit-image is technically complex, key infrastructure; therefore, we
place a high premium on correctness, and on avoiding technical
complexity that may affect maintainability. If you want to make use of
LLMs in a significant way, it is a good idea to **check in with us
first**. Regardless, **always declare tool usage**.

AI agents that have followed all guidelines in this document (outside
of this section) may add 🤖 to their PR title. This signals to
maintainers that the agent has self-verified compliance, enabling
expedited review and acceptance.

(guidelines)=

## Guidelines

- All code should have tests (see [test coverage](#test-coverage) below for more details).
- All code should be documented, to the same
  [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) as NumPy and SciPy.
- For new functionality, always add an example to the gallery (see
  [Gallery](#gallery) below for more details).
- No changes are ever merged without review and approval by two core team members.
  There are two exceptions to this rule. First, pull requests which affect
  only the documentation require review and approval by only one core team
  member in most cases. If the maintainer feels the changes are large or
  likely to be controversial, two reviews should still be encouraged. The
  second case is that of minor fixes which restore CI to a working state,
  because these should be merged fairly quickly. Reach out on the
  [developer forum](https://discuss.scientific-python.org/c/contributor/skimage) if
  you get no response to your pull request.
  **Never merge your own pull request.**

## Stylistic Guidelines

- Use the following import conventions:

  ```
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy as sp
  import skimage as ski

  sp.ndimage.label(...)
  ski.measure.label(...)
  ```

- Use numpy data types instead of strings (`np.uint8` instead of
  `"uint8"`).

- When documenting array parameters, use `image : ndarray of shape (M, N)`
  and then refer to `M` and `N` in the docstring, if necessary.

- Refer to array dimensions as (plane), row, column, not as x, y, z. See
  {ref}`Coordinate conventions <numpy-images-coordinate-conventions>`
  in the user guide for more information.

- Functions should support all input image dtypes. Use utility functions such
  as `img_as_float` to help convert to an appropriate type. The output
  format can be whatever is most efficient. This allows us to string together
  several functions into a pipeline, e.g.:

  ```
  hough(canny(my_image))
  ```

- Use relative module imports, i.e. `from .._shared import xyz` rather than
  `from _skimage2._shared import xyz`.

- For Cython functions:

  - Release the GIL whenever possible, using `with nogil:`.
  - Wrap Cython code in a pure Python function, which defines the
    API. This improves compatibility with code introspection tools,
    which are often not aware of Cython code.

- Use `Py_ssize_t` as data type for all indexing, shape and size variables
  in C/C++ and Cython code.

## Installation

Your system needs a:

- C compiler,
- C++ compiler, and
- a version of Python supported by `scikit-image` (see
  [pyproject.toml](https://github.com/scikit-image/scikit-image/blob/main/pyproject.toml#L14)).

First, [fork the scikit-image repository on GitHub](https://github.com/scikit-image/scikit-image/fork).
Then clone your fork locally and set an `upstream` remote to point to the original scikit-image repository:

:::{note}
We use `git@github.com` below; if you don't have SSH keys setup, use
`https://github.com` instead.
:::

```sh
git clone git@github.com:YOURUSERNAME/scikit-image
cd scikit-image
git remote add upstream git@github.com:scikit-image/scikit-image
```

All commands below are run from within the cloned `scikit-image` directory.

(build-env-setup)=

### Build environment setup

Set up a Python development environment tailored for scikit-image.
Here we provide instructions for two popular environment managers:
`venv` (pip) and `conda` (miniforge).

#### venv

```sh
# Create a virtualenv named ``skimage-dev`` that lives outside of the repository.
# One common convention is to place it inside an ``envs`` directory under your home directory:
mkdir ~/envs
python -m venv ~/envs/skimage-dev

# Activate it
# (On Windows, use ``skimage-dev\Scripts\activate``)
source ~/envs/skimage-dev/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Install scikit-image in editable mode. In editable mode,
# scikit-image will be recompiled, as necessary, on import.
spin install -v
```

:::{tip}
The above installs scikit-image into your environment, which makes
it accessible to IDEs, IPython, etc.
This is not strictly necessary; you can also build with:

```sh
spin build
```

In that case, the library is not installed, but is accessible via
`spin` commands, such as `spin test`, `spin ipython`, `spin run`,
etc.
:::

#### conda

We recommend installing conda using
[miniforge](https://github.com/conda-forge/miniforge),
an alternative to Anaconda without licensing costs.

After installing miniforge:

```sh
# Create a conda environment with required dependencies
conda env create -f environment.yml

# Activate it
conda activate skimage-dev

# Install scikit-image in editable mode. In editable mode,
# scikit-image will be recompiled, as necessary, on import.
spin install -v
```

:::{tip}
The above installs scikit-image into your environment, which makes
it accessible to IDEs, IPython, etc.
This is not strictly necessary; you can also build with:

```sh
spin build
```

In that case, the library is not installed, but is accessible via
`spin` commands, such as `spin test`, `spin ipython`, `spin run`,
etc.
:::

### Adding a feature branch

When contributing a new feature, do so via a feature branch.

First, fetch the latest source:

```sh
git switch main
git pull upstream main
```

Create your feature branch:

```sh
git switch --create my-feature-name
```

Using an editable install, `scikit-image` will rebuild itself as
necessary.
If you are building manually, rebuild with:

```sh
spin build
```

Repeated, incremental builds usually work just fine, but if you notice build
problems, rebuild from scratch using:

```sh
spin build --clean
```

### Platform-specific notes

**Windows**

Building `scikit-image` on Windows is done as part of our continuous
integration testing; the steps are shown in this [Azure Pipeline].

**Debian and Ubuntu**

Install suitable compilers prior to library compilation:

```sh
sudo apt-get install build-essential
```

### Full requirements list

**Build Requirements**

```{eval-rst}
.. include:: ../../../requirements/build.txt
   :literal:
```

**Runtime Requirements**

```{eval-rst}
.. include:: ../../../requirements/default.txt
   :literal:
```

**Test Requirements**

```{eval-rst}
.. include:: ../../../requirements/test.txt
   :literal:
```

**Documentation Requirements**

```{eval-rst}
.. include:: ../../../requirements/docs.txt
   :literal:
```

**Developer Requirements**

```{eval-rst}
.. include:: ../../../requirements/developer.txt
   :literal:
```

**Data Requirements**

The full selection of demo datasets is only available with the
following installed:

```{eval-rst}
.. include:: ../../../requirements/data.txt
   :literal:
```

**Optional Requirements**

You can use `scikit-image` with the basic requirements listed above, but some
functionality is only available with the following installed:

- [Matplotlib](https://matplotlib.org)
  Used in various functions, e.g., for drawing, segmenting, reading images.
- [Dask](https://dask.org/)
  The `dask` module is used to parallelize certain functions.

More rarely, you may also need:

- [PyAMG](https://pyamg.org/)
  The `pyamg` module is used for the fast `cg_mg` mode of random
  walker segmentation.
- [Astropy](https://www.astropy.org)
  Provides FITS I/O capability.
- [SimpleITK](http://www.simpleitk.org/)
  Optional I/O plugin providing a wide variety of [formats](https://itk.org/Wiki/ITK_File_Formats).
  including specialized formats used in biomedical imaging.

```{eval-rst}
.. include:: ../../../requirements/optional.txt
  :literal:
```

### Help with contributor installation

See {ref}`additional-help`.

## Testing

The test suite must pass before a pull request can be merged, and
tests should be added to cover all modifications in behavior.

Tests are located in the `tests/` directory.
We also test examples in docstrings of our package (located in `src/`).

(rng-state)=

### Dealing with RNG state

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

### Warnings during testing phase

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

## Building docs

To build the HTML documentation, run:

```sh
spin docs
```

Output is in `scikit-image/doc/build/html/`. Add the `--clean`
flag to build from scratch, deleting any cached output.

### Gallery

The example gallery is built using
[Sphinx-Gallery](https://sphinx-gallery.github.io).
Refer to their documentation for complete usage instructions, and also
to existing examples in `doc/examples`.

Gallery examples should have a maximum figure width of 8 inches.
You can also [change a gallery entry's thumbnail](https://sphinx-gallery.github.io/stable/configuration.html#choosing-thumbnail).

### Fixing Warnings

- "citation not found: R###" There is probably an underscore after a
  reference in the first line of a docstring (e.g. [1]\_). Use this
  method to find the source file: \$ cd doc/build; grep -rin R####
- "Duplicate citation R###, other instance in..."" There is probably a
  [2] without a [1] in one of the docstrings
- Make sure to use pre-sphinxification paths to images (not the
  \_images directory)

(deprecation-cycle)=

## Deprecation cycle (advanced)

If the way a function is called has to be changed, a deprecation cycle
must be followed to warn users.

A deprecation cycle is _not_ necessary when:

- adding a new function, or
- adding a new keyword argument to the _end_ of a function signature, or
- fixing unexpected or incorrect behavior.

A deprecation cycle is necessary when:

- renaming keyword arguments, or
- changing the order of arguments or keywords, or
- adding arguments to a function, or
- changing a function's name or location, or
- changing the default value of function arguments or keywords.

Typically, deprecation warnings are in place for two releases, before
a change is made.

For example, consider the modification of a default value in
a function signature. In version N, we have:

```python
def some_function(image, rescale=True):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image unless ``False`` is given.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    out = do_something(image, rescale=rescale)
    return out
```

In version N+1, we will change this to:

```python
def some_function(image, rescale=None):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image unless ``False`` is given.

        .. warning:: The default value will change from ``True`` to
                     ``False`` in skimage N+3.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    if rescale is None:
        warn('The default value of rescale will change '
             'to `False` in version N+3.', stacklevel=2)
        rescale = True
    out = do_something(image, rescale=rescale)
    return out
```

And, in version N+3:

```python
def some_function(image, rescale=False):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image if ``True`` is given.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    out = do_something(image, rescale=rescale)
    return out
```

Here is the process for a 3-release deprecation cycle:

- Set the default to `None`, and modify the
  docstring to specify that the default is `True`.
- In the function, \_if\_ rescale is `None`, set it to `True` and warn that the
  default will change to `False` in version N+3.
- In `doc/release/release_dev.rst`, under deprecations, add "In
  `some_function`, the `rescale` argument will default to `False` in N+3."
- In `TODO.txt`, create an item in the section related to version
  N+3 and write "change rescale default to False in some_function".

Note that the 3-release deprecation cycle is not a strict rule and, in some
cases, developers can agree on a different procedure.

### Raising Warnings

`skimage` raises `FutureWarning`s to highlight changes in its
API, e.g.:

```python
from skimage._shared._warnings import warn_external
warn_external(
    "Automatic detection of the color channel was deprecated in "
    "v0.19, and `channel_axis=None` will be the new default in "
    "v0.22. Set `channel_axis=-1` explicitly to silence this "
    "warning.",
    category=FutureWarning,
)
```

### Deprecating Keywords and Functions

When removing keywords or entire functions, the
`_skimage2._shared.utils.deprecate_parameter` and
`_skimage2._shared.utils.deprecate_func` utility functions can be used
to perform the above procedure.

## Adding Data

While code is hosted on [GitHub](https://github.com/scikit-image/),
example datasets are on [GitLab](https://gitlab.com/scikit-image/data).
These are fetched with [pooch](https://github.com/fatiando/pooch)
when accessing `skimage.data.*`.

New datasets are submitted on GitLab and, once merged, the data
registry `skimage/data/_registry.py` in the main GitHub repository
can be updated.

## Benchmarks

While not mandatory for most pull requests, we ask that performance-related
PRs include a benchmark in order to clearly depict the use case that is being
optimized for.

In this section we will review how to setup the benchmarks,
and three commands `spin asv -- dev`, `spin asv -- run` and
`spin asv -- continuous`.

### Prerequisites

Begin by installing [airspeed velocity](https://asv.readthedocs.io/en/stable/)
in your development environment. Prior to installation, be sure to activate your
development environment, then if using `venv` you may install the requirement with:

```
source skimage-dev/bin/activate
pip install asv
```

If you are using conda, then the command:

```
conda activate skimage-dev
conda install asv
```

is more appropriate. Once installed, it is useful to run the command:

```
spin asv -- machine
```

To let airspeed velocity know more information about your machine.

### Writing a benchmark

To write benchmark, add a file in the `benchmarks` directory which contains a
a class with one `setup` method and at least one method prefixed with `time_`.

The `time_` method should only contain code you wish to benchmark.
Therefore it is useful to move everything that prepares the benchmark scenario
into the `setup` method. This function is called before calling a `time_`
method and its execution time is not factored into the benchmarks.

Take for example the `TransformSuite` benchmark:

```python
import numpy as np
from skimage import transform

class TransformSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000))
        idx = np.arange(500, 1500)
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

    def time_hough_line(self):
        result1, result2, result3 = transform.hough_line(self.image)
```

Here, the creation of the image is completed in the `setup` method, and not
included in the reported time of the benchmark.

It is also possible to benchmark features such as peak memory usage. To learn
more about the features, please refer to the official
[airspeed velocity documentation](https://asv.readthedocs.io/en/latest/writing_benchmarks.html).

Also, the benchmark files need to be importable when benchmarking old versions
of scikit-image. So if anything from scikit-image is imported at the top level,
it should be done as:

```python
try:
    from skimage import metrics
except ImportError:
    pass
```

The benchmarks themselves don't need any guarding against missing features,
only the top-level imports.

To allow tests of newer functions to be marked as "n/a" (not available)
rather than "failed" for older versions, the setup method itself can raise a
NotImplemented error. See the following example for the registration module:

```python
try:
    from skimage import registration
except ImportError:
    raise NotImplementedError("registration module not available")
```

### Testing the benchmarks locally

Prior to running the true benchmark, it is often worthwhile to test that the
code is free of typos. To do so, you may use the command:

```
spin asv -- dev -b TransformSuite
```

Where the `TransformSuite` above will be run once in your current environment
to test that everything is in order.

### Running your benchmark

The command above is fast, but doesn't test the performance of the code
adequately. To do that you may want to run the benchmark in your current
environment to see the performance of your change as you are developing new
features. The command `asv run -E existing` will specify that you wish to run
the benchmark in your existing environment. This will save a significant amount
of time since building scikit-image can be a time consuming task:

```
spin asv -- run -E existing -b TransformSuite
```

### Comparing results to main

Often, the goal of a PR is to compare the results of the modifications in terms
speed to a snapshot of the code that is in the main branch of the
`scikit-image` repository. The command `asv continuous` is of help here:

```
spin asv -- continuous main -b TransformSuite
```

This call will build out the environments specified in the `asv.conf.json`
file and compare the performance of the benchmark between your current commit
and the code in the main branch.

The output may look something like:

```
$ spin asv -- continuous main -b TransformSuite
· Creating environments
· Discovering benchmarks
·· Uninstalling from conda-py3.7-cython-numpy1.15-scipy
·· Installing 544c0fe3 <benchmark_docs> into conda-py3.7-cython-numpy1.15-scipy.
· Running 4 total benchmarks (2 commits * 2 environments * 1 benchmarks)
[  0.00%] · For scikit-image commit 37c764cb <benchmark_docs~1> (round 1/2):
[...]
[100.00%] ··· ...ansform.TransformSuite.time_hough_line           33.2±2ms

BENCHMARKS NOT SIGNIFICANTLY CHANGED.
```

In this case, the differences between HEAD and main are not significant
enough for airspeed velocity to report.

It is also possible to get a comparison of results for two specific revisions
for which benchmark results have previously been run via the `asv compare`
command:

```
spin asv -- compare v0.14.5 v0.17.2
```

Finally, one can also run ASV benchmarks only for a specific commit hash or
release tag by appending `^!` to the commit or tag name. For example to run
the skimage.filter module benchmarks on release v0.17.2:

```
spin asv -- run -b Filter v0.17.2^!
```

[azure pipeline]: https://github.com/scikit-image/scikit-image/blob/main/azure-pipelines.yml
