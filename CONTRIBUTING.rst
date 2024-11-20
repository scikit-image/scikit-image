.. _howto_contribute:

How to contribute to scikit-image
=================================

Developing open source software as part of a community is fun, and
often quite educational!

We coordinate our work using GitHub, where you can find lists of `open
issues
<https://github.com/scikit-image/scikit-image/issues?q=is%3Aopen>`__
and `new feature requests
<https://github.com/scikit-image/scikit-image/labels/%3Apray%3A%20Feature%20request>`__.

To follow along with discussions, or to get in touch with the
developer team, please join us on the `scikit-image developer forum
<https://discuss.scientific-python.org/c/contributor/skimage>`_ and
the `Zulip chat <https://skimage.zulipchat.com/>`_.

Please post questions to these public forums (rather than contacting
developers directly); that way, everyone can benefit from the answers,
and developers can answer according to their availability. Don't feel
shy, the team is very friendly!

.. contents::
   :local:

Development process
-------------------
The following is a brief overview about how changes to source code and documentation
can be contributed to scikit-image.

1. If you are a first-time contributor:

   * Go to `https://github.com/scikit-image/scikit-image
     <https://github.com/scikit-image/scikit-image>`_ and click the
     "fork" button to create your own copy of the project.

   * Clone (download) the repository with the project source on your local computer::

      git clone https://github.com/your-username/scikit-image.git

   * Change into the root directory of the cloned repository::

      cd scikit-image

   * Add the upstream repository::

      git remote add upstream https://github.com/scikit-image/scikit-image.git

   * Now, you have remote repositories named:

     - ``upstream``, which refers to the ``scikit-image`` repository, and
     - ``origin``, which refers to your personal fork.

   * Next, :ref:`set up your build environment <build-env-setup>`.

   * Finally, we recommend that you use a pre-commit hook, which runs code
     checkers and formatters each time you do a ``git commit``::

       pip install pre-commit
       pre-commit install

2. Develop your contribution:

   * Pull the latest changes from upstream::

      git checkout main
      git pull upstream main

   * Create a branch for the feature you want to work on. Use a sensible name,
     such as 'transform-speedups'::

      git checkout -b transform-speedups

   * Commit locally as you progress (with ``git add`` and ``git commit``).
     Please write `good commit messages
     <https://vxlabs.com/software-development-handbook/#good-commit-messages>`_.

3. To submit your contribution:

   * Push your changes back to your fork on GitHub::

      git push origin transform-speedups

   * Enter your GitHub username and password (repeat contributors or advanced
     users can remove this step by `connecting to GitHub with SSH
     <https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_).

   * Go to GitHub. The new branch will show up with a green "pull request"
     button -- click it.

   * If you want, post on the `developer forum
     <https://discuss.scientific-python.org/c/contributor/skimage>`_ to explain your changes or
     to ask for review.

For a more detailed discussion, read these :doc:`detailed documents
<../gitwash/index>` on how to use Git with ``scikit-image`` (:ref:`using-git`).

4. Review process:

   * Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your pull request (PR) to help
     you improve its implementation, documentation, and style.  Every single
     developer working on the project has their code reviewed, and we've come
     to see it as a friendly conversation from which we all learn and the
     overall code quality benefits.  Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of the project, not to criticize (we are, after all, very grateful for the
     time you're donating!).

   * To update your pull request, make your changes on your local repository
     and commit. As soon as those changes are pushed up (to the same branch as
     before) the pull request will update automatically.

   * Continuous integration (CI) services are triggered after each pull request
     submission to build the package, run unit tests, measure code coverage,
     and check the coding style (PEP8) of your branch. The tests must pass
     before your PR can be merged. If CI fails, you can find out why by
     clicking on the "failed" icon (red cross) and inspecting the build and
     test logs.

   * A pull request must be approved by two core team members before merging.

.. _documenting-changes:

5. Document changes

   If your change introduces a deprecation, add a reminder to ``TODO.txt``
   for the team to remove the deprecated functionality in the future.

   scikit-image uses `changelist <https://github.com/scientific-python/changelist>`_
   to generate a list of release notes automatically from pull requests. By
   default, changelist will use the title of a pull request and its GitHub
   labels to sort it into the appropriate section. However, for more complex
   changes, we encourage you to describe them in more detail using the
   `release-note` code block within the pull request description; e.g.::

       ```release-note
       Remove the deprecated function `skimage.color.blue`. Blend
       `skimage.color.cyan` and `skimage.color.magenta` instead.
       ```

   You can refer to :doc:`/release_notes/index` for examples and to
   `changelist's documentation <https://github.com/scientific-python/changelist>`_
   for more details.

.. note::

   To reviewers: if it is not obvious from the PR description, make sure that
   the reason and context for a change are described in the merge message.


Divergence between ``upstream main`` and your feature branch
------------------------------------------------------------

If GitHub indicates that the branch of your PR can no longer
be merged automatically, merge the main branch into yours::

   git fetch upstream main
   git merge upstream/main

If any conflicts occur, they need to be fixed before continuing.  See
which files are in conflict using::

   git status

Which displays a message like::

   Unmerged paths:
     (use "git add <file>..." to mark resolution)

     both modified:   file_with_conflict.txt

Inside the conflicted file, you'll find sections like these::

   The way the text looks in your branch

Choose one version of the text that should be kept, and delete the
rest::

   The way the text looks in your branch

Now, add the fixed file::

   git add file_with_conflict.txt

Once you've fixed all merge conflicts, do::

   git commit

.. note::

   Advanced Git users are encouraged to `rebase instead of merge
   <https://scikit-image.org/docs/dev/gitwash/development_workflow.html#rebasing-on-trunk>`__,
   but we squash and merge most PRs either way.

Guidelines
----------

* All code should have tests (see `test coverage`_ below for more details).
* All code should be documented, to the same
  `standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ as NumPy and SciPy.
* For new functionality, always add an example to the gallery (see
  `Gallery`_ below for more details).
* No changes are ever merged without review and approval by two core team members.
  There are two exceptions to this rule. First, pull requests which affect
  only the documentation require review and approval by only one core team
  member in most cases. If the maintainer feels the changes are large or
  likely to be controversial, two reviews should still be encouraged. The
  second case is that of minor fixes which restore CI to a working state,
  because these should be merged fairly quickly. Reach out on the
  `developer forum <https://discuss.scientific-python.org/c/contributor/skimage>`_ if
  you get no response to your pull request.
  **Never merge your own pull request.**

Stylistic Guidelines
--------------------

* Set up your editor to remove trailing whitespace.  Follow `PEP08
  <https://www.python.org/dev/peps/pep-0008/>`__.

* Use numpy data types instead of strings (``np.uint8`` instead of
  ``"uint8"``).

* Use the following import conventions::

   import numpy as np
   import matplotlib.pyplot as plt
   import scipy as sp
   import skimage as ski

   sp.ndimage.label(...)
   ski.measure.label(...)

   # only in Cython code
   cimport numpy as cnp
   cnp.import_array()

* When documenting array parameters, use ``image : (M, N) ndarray``
  and then refer to ``M`` and ``N`` in the docstring, if necessary.

* Refer to array dimensions as (plane), row, column, not as x, y, z. See
  :ref:`Coordinate conventions <numpy-images-coordinate-conventions>`
  in the user guide for more information.

* Functions should support all input image dtypes.  Use utility functions such
  as ``img_as_float`` to help convert to an appropriate type.  The output
  format can be whatever is most efficient.  This allows us to string together
  several functions into a pipeline, e.g.::

   hough(canny(my_image))

* Use ``Py_ssize_t`` as data type for all indexing, shape and size variables
  in C/C++ and Cython code.

* Use relative module imports, i.e. ``from .._shared import xyz`` rather than
  ``from skimage._shared import xyz``.

* Wrap Cython code in a pure Python function, which defines the API. This
  improves compatibility with code introspection tools, which are often not
  aware of Cython code.

* For Cython functions, release the GIL whenever possible, using
  ``with nogil:``.

Testing
-------

The test suite must pass before a pull request can be merged, and
tests should be added to cover all modifications in behavior.

We use the `pytest <https://docs.pytest.org/en/latest/>`__ testing
framework, with tests located in the various
``skimage/submodule/tests`` folders.

Testing requirements are listed in `requirements/test.txt`.
Run:

- **All tests**: ``spin test``
- Tests for a **submodule**: ``spin test skimage/morphology``
- Run tests from a **specific file**: ``spin test skimage/morphology/tests/test_gray.py``
- Run **a test inside a file**:
  ``spin test skimage/morphology/tests/test_gray.py::test_3d_fallback_black_tophat``
- Run tests with **arbitrary ``pytest`` options**:
  ``spin test -- any pytest args you want``.
- Run all tests and **doctests**:
  ``spin test -- --doctest-plus skimage``

Warnings during testing phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, warnings raised by the test suite result in errors.
You can switch that behavior off by setting the environment variable
``SKIMAGE_TEST_STRICT_WARNINGS`` to `0`.


Test coverage
-------------

Tests for a module should ideally cover all code in that module,
i.e., statement coverage should be at 100%.

To measure test coverage run::

  $ spin test --coverage

This will run tests and print a report with one line for each file in `skimage`,
detailing the test coverage::

  Name                                             Stmts   Exec  Cover   Missing
  ------------------------------------------------------------------------------
  skimage/color/colorconv                             77     77   100%
  skimage/filter/__init__                              1      1   100%
  ...


Building docs
-------------

To build the HTML documentation, run:

.. code:: sh

    spin docs

Output is in ``scikit-image/doc/build/html/``.  Add the ``--clean``
flag to build from scratch, deleting any cached output.

Gallery
^^^^^^^

The example gallery is built using
`Sphinx-Gallery <https://sphinx-gallery.github.io>`_.
Refer to their documentation for complete usage instructions, and also
to existing examples in ``doc/examples``.

Gallery examples should have a maximum figure width of 8 inches.
You can also `change a gallery entry's thumbnail
<https://sphinx-gallery.github.io/stable/configuration.html#choosing-thumbnail>`_.

Fixing Warnings
^^^^^^^^^^^^^^^

-  "citation not found: R###" There is probably an underscore after a
   reference in the first line of a docstring (e.g. [1]\_). Use this
   method to find the source file: $ cd doc/build; grep -rin R####

-  "Duplicate citation R###, other instance in..."" There is probably a
   [2] without a [1] in one of the docstrings

-  Make sure to use pre-sphinxification paths to images (not the
   \_images directory)

Deprecation cycle
-----------------

If the way a function is called has to be changed, a deprecation cycle
must be followed to warn users.

A deprecation cycle is *not* necessary when:

* adding a new function, or
* adding a new keyword argument to the *end* of a function signature, or
* fixing unexpected or incorrect behavior.

A deprecation cycle is necessary when:

* renaming keyword arguments, or
* changing the order of arguments or keywords, or
* adding arguments to a function, or
* changing a function's name or location, or
* changing the default value of function arguments or keywords.

Typically, deprecation warnings are in place for two releases, before
a change is made.

For example, consider the modification of a default value in
a function signature. In version N, we have:

.. code-block:: python

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

In version N+1, we will change this to:

.. code-block:: python

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

And, in version N+3:

.. code-block:: python

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

Here is the process for a 3-release deprecation cycle:

- Set the default to `None`, and modify the
  docstring to specify that the default is `True`.
- In the function, _if_ rescale is `None`, set it to `True` and warn that the
  default will change to `False` in version N+3.
- In ``doc/release/release_dev.rst``, under deprecations, add "In
  `some_function`, the `rescale` argument will default to `False` in N+3."
- In ``TODO.txt``, create an item in the section related to version
  N+3 and write "change rescale default to False in some_function".

Note that the 3-release deprecation cycle is not a strict rule and, in some
cases, developers can agree on a different procedure.

Raising Warnings
^^^^^^^^^^^^^^^^

``skimage`` raises ``FutureWarning``\ s to highlight changes in its
API, e.g.:

.. code-block:: python

   from warnings import warn
   warn(
       "Automatic detection of the color channel was deprecated in "
       "v0.19, and `channel_axis=None` will be the new default in "
       "v0.22. Set `channel_axis=-1` explicitly to silence this "
       "warning.",
       FutureWarning,
       stacklevel=2,
   )

The `stacklevel
<https://docs.python.org/3/library/warnings.html#warnings.warn>`_ is
a bit of a technicality, but ensures that the warning points to the
user-called function, and not to a utility function within.

In most cases, set the ``stacklevel`` to ``2``.
When warnings originate from helper routines internal to the
scikit-image library, set it to ``3``.

To test if your warning is being emitted correctly, try calling the function
from an IPython console. It should point you to the console input itself
instead of being emitted by files in the scikit-image library:

* **Good**: ``ipython:1: UserWarning: ...``
* **Bad**: ``scikit-image/skimage/measure/_structural_similarity.py:155: UserWarning:``

Deprecating Keywords and Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When removing keywords or entire functions, the
``skimage._shared.utils.deprecate_parameter`` and
``skimage._shared.utils.deprecate_func`` utility functions can be used
to perform the above procedure.

Adding Data
-----------
While code is hosted on `github <https://github.com/scikit-image/>`_,
example datasets are on `gitlab <https://gitlab.com/scikit-image/data>`_.
These are fetched with `pooch <https://github.com/fatiando/pooch>`_
when accessing `skimage.data.*`.

New datasets are submitted on gitlab and, once merged, the data
registry ``skimage/data/_registry.py`` in the main GitHub repository
can be updated.

Benchmarks
----------
While not mandatory for most pull requests, we ask that performance related
PRs include a benchmark in order to clearly depict the use-case that is being
optimized for. A historical view of our snapshots can be found on
at the following `website <https://pandas.pydata.org/speed/scikit-image/>`_.

In this section we will review how to setup the benchmarks,
and three commands ``spin asv -- dev``, ``spin asv -- run`` and
``spin asv -- continuous``.

Prerequisites
^^^^^^^^^^^^^
Begin by installing `airspeed velocity <https://asv.readthedocs.io/en/stable/>`_
in your development environment. Prior to installation, be sure to activate your
development environment, then if using ``venv`` you may install the requirement with::

  source skimage-dev/bin/activate
  pip install asv

If you are using conda, then the command::

  conda activate skimage-dev
  conda install asv

is more appropriate. Once installed, it is useful to run the command::

  spin asv -- machine

To let airspeed velocity know more information about your machine.

Writing a benchmark
^^^^^^^^^^^^^^^^^^^
To write  benchmark, add a file in the ``benchmarks`` directory which contains a
a class with one ``setup`` method and at least one method prefixed with ``time_``.

The ``time_`` method should only contain code you wish to benchmark.
Therefore it is useful to move everything that prepares the benchmark scenario
into the ``setup`` method. This function is called before calling a ``time_``
method and its execution time is not factored into the benchmarks.

Take for example the ``TransformSuite`` benchmark:

.. code-block:: python

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

Here, the creation of the image is completed in the ``setup`` method, and not
included in the reported time of the benchmark.

It is also possible to benchmark features such as peak memory usage. To learn
more about the features, please refer to the official
`airspeed velocity documentation <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_.

Also, the benchmark files need to be importable when benchmarking old versions
of scikit-image. So if anything from scikit-image is imported at the top level,
it should be done as:

.. code-block:: python

    try:
        from skimage import metrics
    except ImportError:
        pass

The benchmarks themselves don't need any guarding against missing features,
only the top-level imports.

To allow tests of newer functions to be marked as "n/a" (not available)
rather than "failed" for older versions, the setup method itself can raise a
NotImplemented error.  See the following example for the registration module:

.. code-block:: python

    try:
        from skimage import registration
    except ImportError:
        raise NotImplementedError("registration module not available")

Testing the benchmarks locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prior to running the true benchmark, it is often worthwhile to test that the
code is free of typos. To do so, you may use the command::

  spin asv -- dev -b TransformSuite

Where the ``TransformSuite`` above will be run once in your current environment
to test that everything is in order.

Running your benchmark
^^^^^^^^^^^^^^^^^^^^^^

The command above is fast, but doesn't test the performance of the code
adequately. To do that you may want to run the benchmark in your current
environment to see the performance of your change as you are developing new
features. The command ``asv run -E existing`` will specify that you wish to run
the benchmark in your existing environment. This will save a significant amount
of time since building scikit-image can be a time consuming task::

  spin asv -- run -E existing -b TransformSuite

Comparing results to main
^^^^^^^^^^^^^^^^^^^^^^^^^

Often, the goal of a PR is to compare the results of the modifications in terms
speed to a snapshot of the code that is in the main branch of the
``scikit-image`` repository. The command ``asv continuous`` is of help here::

  spin asv -- continuous main -b TransformSuite

This call will build out the environments specified in the ``asv.conf.json``
file and compare the performance of the benchmark between your current commit
and the code in the main branch.

The output may look something like::

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

In this case, the differences between HEAD and main are not significant
enough for airspeed velocity to report.

It is also possible to get a comparison of results for two specific revisions
for which benchmark results have previously been run via the `asv compare`
command::

    spin asv -- compare v0.14.5 v0.17.2

Finally, one can also run ASV benchmarks only for a specific commit hash or
release tag by appending ``^!`` to the commit or tag name. For example to run
the skimage.filter module benchmarks on release v0.17.2::

    spin asv -- run -b Filter v0.17.2^!
