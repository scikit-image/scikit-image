.. _skip_5_array_api:

===================================
SKIP 5 — Initial Array API adoption
===================================

:Author: Evgeni Burovski <evgeny.burovskiy@gmail.com>
:Status: Draft
:Type: Standards Track
:Created: 2026-05-18
:Resolved: <null>
:Resolution: <null>
:Version effective: 2.0

Abstract
--------

The Array API standard [3]_ aims to standardize a common subset of functionality
of the majority of array libraries, such as NumPy, PyTorch, CuPy and JAX, with
a view to remedy the fragmentation of the array computing ecosystem caused by the
accumulated divergences among these---almost, but not completely dissimilar---array/tensor libraries.

Adopting the Array API standard in ``scikit-image`` gives users flexibility in choosing
their software stack for array computing, and unlocks performance improvements from
hardware accelerated implementations [4]_. Specific ways to harness hardware acceleration
differ for different scikit-image functions. For functions implemented in Cython,
some form of dispatching is required (and scikit-image has experimented with already).

For functions which rely on SciPy however---``scipy.ndimage`` and ``scipy.fft`` are
heavily used in scikit-image, and delegate to CUDA kernels for CuPy array inputs internally---
adopting the Array API unlocks hardware acceleration at a marginal maintenance cost.
What fraction of the ``scikit-image`` API surface is amendable to this "immediate"
acceleration? While the answer varies by the submodule, from a partial sample of the
API surface, we estimate that 50-70% of it can use harware acceleration from porting
the Python code to be Array API compatible. See the Discussion section for details.

More generally, Array API compatibility gives a generic framework for, and implements
the foundational infrastructure of, dispatching to specialized accelerator-enabled
implementations (such as CUCIM and similar GPU libraries). Specific details
of the dispatching can take multiple forms, and are under discussions across the ecosystem.
What the Array API compatibility provides however, is a general and ecosystem-aligned
framework for working through these (both fascinating and difficult) details.


Motivation and Scope
--------------------

Historically, NumPy set the stage for array computing in Python and serves as the
base of the whole scientific computing ecosystem. With time, multiple alternative
array/tensor computing libraries appeared and became popular: CuPy for "NumPy on GPUs",
PyTorch and JAX, to name just a few. All these low-level libraries are similar but
have enough differences to make moving between libraries difficult. Each of these
libraries serves as a base of its respective collection of domain-specific packages
and software stacks. As a result, the ecosystem is fractured.

The Array API standard aims to unify the ecosystem by specifying the minimal common
subset that all array/tensor computing libraries implement. Then, when domain-specific
libraries support this minimal useful subset, end users automatically benefit from
hardware-specific implementations "hidden" in the array libraries.

Ongoing work on adopting the Array API standard in SciPy [5]_, [6]_ and
scikit-learn [7]_ has shown performance improvements of up to 50x or more from using
CuPy or PyTorch GPU for the array compute layer (see [4]_ for an partial overview).
Note that these performance gains for end users are essentially "free" (end users simply
feed the right array types to scipy/scikit-learn functions), and the work for enabling
them is borne by the library authors and maintainers.
The rest of this document details changes needed in scikit-image for an initial
Array API support.


Detailed description
--------------------

All of Array API processing is still *experimental*: users need to define the environment
variable ``SCIPY_ARRAY_API=1`` before importing SciPy and ``scikit-image``. This follows
SciPy and ``scikit-learn``, and is necessary as long as SciPy requires the environment
variable to enable the Array API dispatch (likely, until SciPy 2.0 is released).

If Array API dispatch is not activated via the environment variable, the behavior is
exactly backwards compatible. When the Array API dispatch is active:

- A high-level goal is that a scikit-image function is able to receive array arguments
  from a namespace X and return results with arrays of the same namespace X (with X
  being, for example, NumPy, CuPy or pytorch).

- Where non-CPU devices are involved, no silent device transfers are permitted by default
  (exceptions to this rule should be rare and need an explicit decision and documentation).

- Mixing arrays from different namespaces is not allowed: ``func(cupy_array, numpy_array)``
  raises an error instead of guessing the user intent or silently device transferring one
  of the arrays.

- For backwards compatibility, "array_like" arguments (e.g. lists) are treated as NumPy arrays.

- Two new runtime dependencies are added: ``array-api-compat`` and ``array-api-extra``.
  Both dependencies are pure python packages, available from PyPI and conda-forge.
  If adding runtime dependencies creates an undue burden for users, these two packages
  can be vendored instead. The need for these dependencies is as follows:

  - Not all array libraries implement the Array API spec completely. ``array-api-compat``
    is a lightweight compatibility layer which smoothens out the deviations from the spec
    for ``numpy``, ``cupy`` and ``pytorch``.
  - ``array-api-extra`` provides a collection of non-standard functions which were found
    to be broadly useful from previous adoption work in SciPy, scikit-learn and other
    packages (e.g. ``atleast_nd`` as a replacement for ``atleast_{1d,2d}``, testing
    helpers to replace those from ``np.testing``, and so on).

The initial Array API adoption effort targets the following array libraries: NumPy,
CuPy and Pytorch (in the eager mode). For testing, it is convenient to also
rely on ``array-api-strict``---a strict implementation of the Array API which is
specifically made to implement the specification to the letter, as an implement for
flagging deviations from the Array API standard. The package is available from
PyPI and conda-forge, and will be added as a new test-only dependency.


Potential follow-ups
~~~~~~~~~~~~~~~~~~~~

Potential targets for follow-up efforts may include JAX, ``marray`` and JIT modes of
JAX and pytorch. Why these are not included in the initial support:

- JAX arrays are immutable, and scikit-image extensively relies on in-place modifications
  of numpy arrays. While there is a wealth of experience of adding JAX support in SciPy,
  and array-api-extra contains useful primitives for supporting both mutable and immutable
  array libraries, we feel that implementing the support is best left for a follow-up.
- Supporting ``marray`` masked arrays would be a separate enhancement, if there's a sufficient
  interest.
- Supporting JIT modes of pytorch and jax. These can potentially provide non-trivial
  performance benefits; however adding the support increases the scope significantly,
  and is best left for a follow-up effort.


Related Work
------------

SciPy [6]_ and scikit-learn [7]_, are implementing a similar adoption process, and this
proposal heavily uses the experience of converting these two foundational libraries.

SciPy supports a wider range of array API providers out of the box (in particular, JAX;
there is also partial support for dask; scipy.stats is adding support for marray).
Particularly notable for scikit-image is SciPy's support of CuPy in the
``scipy.ndimage`` submodule: scikit-image will be able to rely on it (nearly) automatically.

scikit-learn supports a limited set of array libraries
(NumPy, CuPy and pytorch---essentially what we suggest here), and its estimators dispense
with the "no silent device transfer" design rule. This way, users can build "pipelines"
where internal steps do make device transfers. We believe this should be only enabled
only where there is a strong need, and we do not include it in the initial proposal here.


Implementation
--------------

**For pure python code**, majority of implementation changes amount to cleaning up
old-style numpy idioms. For instance, ``np.issubdtype(x.dtype.type, np.complexfloating)``
becomes ``xp.isdtype(x.dtype, "complex floating")``. Here ``xp`` is the array namespace,
which is typically computed in public functions by calling the ``array_namespace``
function from the ``array-api-compat`` package::

    def foo(x: array_like, y: array_like, mode: bool):
        xp = array_namespace(x, y)
        ...

See [8]_, [9]_ for a discussion and worked examples, and/or [14]_ for a SciPy-specific
discussion.

Using **functions from upstream libraries which have already adopted the Array API standard**
does not need any changes. For example, ``scipy.ndimage.sobel`` supports NumPy arrays,
CuPy arrays and pytorch CPU tensors, and adheres to the *namespace out = namespace in*
design rule. This is, in fact, the behavior of the the majority of the
``scipy.ndimage`` API. [10]_

**Compiled code** written in C or Cython becomes *CPU only* in the Array API jargon via
the basic sandwich pattern::

    def foo(x: array_like):
        xp = array_namespace(x)
        x_np = np.asarray(x)     # convert to numpy
        result_np = cython_implemented_function(x_np)   # compute
        return xp.asarray(result_np)   # convert back

Note that this pattern is zero-copy for arrays on CPU [11]_, and raises a clear error
if converting to NumPy requires a device transfer. See [14]_ for a detailed discussion.

**Backend-specific code paths**. In rare cases, there is a need to use backend-specific
code paths. A common occurrence is calling numpy-specific code as a performance
optimization. For example, a linear solve is ``xp.linalg.solve(a, b)``; for symmetric
matrices, SciPy allows an optimization, ``scipy.linalg.solve(a, b, assume_a="sym")``,
but 1) the additional ``assume_a`` argument is not available in Array API compatible
libraries, and 2) the SciPy version is numpy-specific and does not handle non-numpy
arrays. One can either use the "sandwich" and settle on CPU-only code, or branch
explicitly with the the ``is_numpy`` function from array-api-compat::

    xp = array_namespace(a, b)
    # ...
    if is_numpy(xp):
        # use the symmetric solve
        x = scipy.linalg.solve(a, b, assume_a="sym")
    else:
        # fall back to a general solve
        x = xp.linalg.solve(a, b)

The use of this pattern (and its analogs, ``is_torch``, ``is_cupy``) should be limited
to an absolute minimum.


**Testing practice**. Adapting tests to validate the behavior across backends is
relatively straightforward if laborious. Following the established SciPy practice [14]:

- Test functions acquire a new ``xp`` fixture; During the test run, it parameterizes
  the test over the installed backends::

    def test_foo(xp):
        x_np = np.random.uniform(size=42)
        x = xp.asarray(x_np)
        result = foo(x)
        # assert properties of `result`

- Use ``xp``-aware assertions from ``array-api-extra`` instead of those from ``np.testing``::

    from array_api_extra import xp_assert_close  # instead of np.testing.assert_allclose
    # ...
    def test_foo(xp):
        x = xp.arange(12)
        xp_assert_close(x[-2:], xp.asarray([10, 11], atol=1e-15)

  These assertions gracefully handle GPU arrays, and by default check for common errors
  (returning an array from a wrong namespace, wrong shapes/dtypes).


**CI practice**. The CPU CI can reuse the SciPy setup,
https://github.com/scipy/scipy/blob/main/.github/workflows/array_api.yml
The CUDA/GPU CI can also follow SciPy, once a solution to
https://github.com/scipy/scipy/issues/24990 emerges.


Compiled code: beyond the sandwich pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pattern of *convert to numpy, perform computations, convert back*, recommended
above for the compiled code, has an obvious drawback of not allowing non-CPU arrays.
There are several ways of removing this limitation, all of which come with
maintainability costs, and are thus excluded from the initial proposal of this SKIP.
Briefly:

- *delegate* to a compatible library with the relevant functionality, if it
  exists. Specific delegation / dispatch patterns of varying degrees of sophistication
  are under active discussion in the community;
- maintain parallel "backends": typically this means writing a "generic", pure python
  analog of the compiled function and call it for non-numpy array inputs. Examples
  of this approach include ``scipy.spatial.Rotation`` and
  ``scipy.interpolate.RBFInterpolator``. This has an obvious maintenance cost
  of having to maintain two parallel implementations, which has to be balanced with
  the performance benefits (if any). See [4]_ and [12]_ for demos of the latter.
- For functions implemented in ``pythran``, use the pythran's "dual mode" [13]_, where
  the same source code is used for both ahead-of-time compiling for numpy arrays and
  a "generic" Array API backend for other array types. This functionality is very
  new in Pythran and needs extensive investigation still.


Backward compatibility
----------------------

There is no pressing need to break backwards compatibility for the Array API support
itself. Alternative backends have some limitations relative to NumPy, thus
numpy-specific parts will keep being numpy-specific (for instance, longdouble dtypes
are non-existent in PyTorch).


Alternatives
------------

The proposed approach is heavily informed by the experience of adopting the Array API in
SciPy and scikit-learn.


Discussion
----------

Much of ``scikit-image`` relies on computational kernels written in Cython. A frequent
concern is whether adopting the Array API brings meaningful benefits---these handwritten
kernels are and will remain being NumPy-only, while the majority of gains reported
in [4]_ are from using hardware accelerators.

First of all, to an extent that ``scikit-image`` uses SciPy, the relevant SciPy submodules,
notably ``scipy.ndimage`` and ``scipy.fft``, do benefit from GPU execution for CuPy
arrays, today.
This way, scikit-image functions which wrap ``scipy.ndimage`` or
``scipy.fft`` functions *and* have their internals Array API compatible, use
CUDA automatically for CuPy array inputs. See [15]_ for a worked example.

It is instructive to see what fraction of the scikit-image functionality can use this
form of GPU acceleration. A table below gives a breakdown for several scikit-image
submodules [16]_. Each column of the table gives a percentage of the API surface of
a submodule, grouped according to the following criteria:

- a function is _"CuPy ready"_ if simply converting a function to the Array API makes
  it run natively on GPU. Typically, this means that the function currently only uses
  NumPy calls and its dependencies are GPU-enabled (typically, ``scipy.ndimage`` and
  ``scipy.fft``).

- _"minor dep"_ means that while not all dependencies are GPU-ready today, it can be made
  GPU ready with small amount of work, upstream or in scikit-image itself. Typical 
  reasons include ``scipy.spatial.KDTree`` (which has a delegation target in
  ``cupyx.scipy.spatial.KDTree``) or ``numpy.bincount`` (which is technically not
  in the Array API standard version 2025.12, but is considered for inclusion and
  has a delegation target in CuPy versions ``>= 13``).

- _"major dep"_  means that a function has dependencies which will either require a
  significant amount of work to make it GPU-compatible or a major new dependency.
  A typical example is ``scipy.spatial.ConvexHull``.

- _"Cython kernel"_ means that a function uses a dedicated Cython kernel, maintained in
  ``scikit-image`` itself. These Cython kernels are of course CPU only.


API dependency breakdown for several submodules

=========    ==========  =========  =========  =============  ==============
submodule    CuPy ready  minor dep  major dep  Cython kernel  number of APIs
=========    ==========  =========  =========  =============  ==============
morphology   46%         4%         9%         45%            46
exposure     40%         60%        0          0              10
filters      79%         17%        0          2%             47
filters.rank 0           0          0          100%           31
transform    68%         15%        3%         13%            38
metrics      40%         20%        30%        10%            10
=========    ==========  =========  =========  =============  ==============



References and Footnotes
------------------------
All SKIPs should be declared as dedicated to the public domain with the CC0
license [1]_, as in `Copyright`, below, with attribution encouraged with CC0+BY
[2]_.

.. [1] CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
   https://creativecommons.org/publicdomain/zero/1.0/
.. [2] https://dancohen.org/2013/11/26/cc0-by/
.. [3] https://data-apis.org/array-api/latest/
.. [4] https://labs.quansight.org/blog/array-api-meta-blogpost
.. [5] SciPy tracker issue for the Array API adoption,
   https://github.com/scipy/scipy/issues/18867
.. [6] SciPy Array API coverage,
   https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#api-coverage
.. [7] scikit-learn Array API coverage,
   https://scikit-learn.org/stable/modules/array_api.html
.. [8] Array API migration guide,
   https://data-apis.org/array-api/draft/migration_guide.html
.. [9] Array API migration tutorial,
   https://data-apis.org/array-api/draft/tutorial_basic.html
.. [10] Under the hood, `scipy.ndimage` functions *delegate* to eponymous
   ``cupyx.scipy.ndimage`` functions if their inputs are CuPy arrays. Since this
   delegation internal to ``scipy.ndimage``, ``scikit-learn`` developers automatically
   benefit.
.. [11] This is exactly the way ``scipy.ndimage`` support PyTorch CPU tensors.
.. [12] Benchmarks for RBFInterpolator "alternative backends",
   https://github.com/scipy/scipy/pull/23447#issuecomment-3224201868
.. [13] Pythran "dual mode" enhancement:
   https://github.com/serge-sans-paille/pythran/pull/2371 and a SciPy POC:
   https://github.com/scipy/scipy/pull/24306
.. [14] SciPy Array API implementation notes,
   https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#implementation-notes
.. [15] A proof of concept demonstration of the Array API compatibility in
   ``scikit-image``, https://github.com/scikit-image/scikit-image/pull/8182
.. [16] For the raw annotated breakdown see
   https://docs.google.com/spreadsheets/d/1Kz5b2G1FowAg0MjPxZxZ7yPs-IJrlaFGksE3AiScO98/edit?usp=sharing
   Data collected for `scikit-image` version 0.26.

Copyright
---------

This document is dedicated to the public domain with the Creative Commons CC0
license [1]_. Attribution to this source is encouraged where appropriate, as per
CC0+BY [2]_.
