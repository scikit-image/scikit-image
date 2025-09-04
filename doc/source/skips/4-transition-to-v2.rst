.. _skip_4_transition_v2:

==========================================
SKIP 4 — Transitioning to scikit-image 2.0
==========================================

:Author: Juan Nunez-Iglesias <juan.nunez-iglesias@monash.edu>
:Author: Lars Grüter
:Author: Stéfan van der Walt
:Author: Matthew Brett
:Author: Marianne Corvellec
:Status: Draft
:Type: Standards Track
:Created: 2025-08-16
:Resolved: <null>
:Resolution: <null>
:Version effective: None

Abstract
--------

scikit-image is preparing to release version 1.0. This :ref:`was seen
<skip_3_transition_v1>` as an opportunity to clean up the API, including
backwards incompatible changes. Some of these changes involve changing return
values without changing function signatures, which can ordinarily only be done
by adding an otherwise useless keyword argument (such as
``new_return_style=True``) whose default value changes over several releases.
The result is *still* a backwards incompatible change, but made over a longer
time period.

Despite being in beta and in a 0.x series of releases, scikit-image is used
extremely broadly, and any backwards incompatible changes are likely to be
disruptive. Given the rejection of :ref:`SKIP-3 <skip_3_transition_v1>`, this
document proposes an alternative pathway to create a new API. The new pathway
involves the following steps:

- Introduce a new namespace ``skimage2`` that will be included in the package
  alongside the existing ``skimage`` namespace during a transition period.
- The new API will be implemented in ``skimage2`` and will initially be marked as
  unstable and experimental. The old API in ``skimage`` will continue working.
- Eventually, when the new API in ``skimage2`` is complete, the old namespace
  ``skimage`` will be deprecated and eventually removed.

See the :ref:`skip4_implementation` section for a more detailed description of
the changes and steps involved.

Motivation and Scope
--------------------

.. note:: This is largely duplicated from :ref:`SKIP-3 <skip_3_transition_v1>`.

scikit-image has grown organically over the past 12+ years, with functionality
being added by a broad community of contributors from different backgrounds.
This has resulted in various parts of the API being inconsistent: for example,
``skimage.transform.warp`` inverts the order of coordinates, so that a
translation of (45, 32) actually moves the values in a NumPy array by 32 along
the 0th axis, and 45 along the 1st, *but only in 2D*. In 3D, a translation of
(45, 32, 77) moves the values in each axis by the number in the corresponding
position.

Additionally, as our user base has grown, it has become apparent that certain
early API choices turned out to be more confusing than helpful. For example,
scikit-image will automatically convert images to various data types,
*rescaling them in the process*. A uint8 image in the range [0, 255] will
automatically be converted to a float64 image in [0, 1]. This might initially
seem reasonable, but, for consistency, uint16 images in [0, 65535] are rescaled
to [0, 1] floats, and uint16 images with 12-bit range in [0, 4095], which are
common in microscopy, are rescaled to [0, 0.0625]. These silent conversions
have resulted in much user confusion.

Changing this convention would require adding a ``preserve_range=`` keyword
argument to almost *all* scikit-image functions, whose default value would
change from False to True over 4 versions. Eventually, the change would be
backwards-incompatible, no matter how gentle we made the deprecation curve.

Other major functions, such as ``skimage.measure.regionprops``, could use an
API tweak, for example by returning a dictionary mapping labels to properties,
rather than a list.

Given the accumulation of potential API changes that have turned out to be too
burdensome and noisy to fix with a standard deprecation cycle, principally
because they involve changing function outputs for the same inputs, it makes
sense to make all those changes in a transition to version 2.0.

Although semantic versioning [6]_ technically allows API changes with major
version bumps, we must acknowledge that (1) an enormous number of projects
depend on scikit-image and would thus be affected by backwards incompatible
changes, and (2) it is not yet common practice in the scientific Python
community to put upper version bounds on dependencies, so it is very unlikely
that anyone used ``scikit-image<1.*`` or ``scikit-image<2.*`` in their
dependency list. This implies that releasing a version 2.0 of scikit-image with
breaking API changes would disrupt a large number of users. Additionally, such
wide-sweeping changes would invalidate a large number of StackOverflow and
other user guides. Finally, releasing a new version with a large number of
changes prevents users from gradually migrating to the new API: an old code
base must be migrated wholesale because it is impossible to depend on both
versions of the API. This would represent an enormous barrier of entry for many
users.

Given the above, this SKIP proposes that we release a new package where we can
apply everything we have learned from over a decade of development, without
disrupting our existing user base.

Detailed description
--------------------

It is beyond the scope of this document to list all of the proposed API changes
for skimage2, many of which have yet to be decided upon. Indeed, the
scope and ambition of the 2.0 transition could grow if this SKIP is accepted.
This SKIP instead proposes a mechanism for managing the transition without
breaking users' code. A meta-issue tracking the proposed changes can be found
on GitHub, scikit-image/scikit-image#5439 [7]_. Some examples are briefly
included below for illustrative purposes:

- Stop rescaling input arrays when the dtype must be coerced to float.
- Stop swapping coordinate axis order in different contexts, such as drawing or
  warping.
- Allow automatic return of non-NumPy types, so long as they are coercible to
  NumPy with ``numpy.asarray``.
- Harmonizing similar parameters in different functions to have the same name;
  for example, we currently have ``random_seed``, ``random_state``, ``seed``,
  or ``sample_seed`` in different functions, all to mean the same thing.
- Changing ``measure.regionprops`` to return a dictionary instead of a list.
- Combine functions that have the same purpose, such as ``watershed``,
  ``slic``, or ``felzenschwalb``, into a common namespace. This would make it
  easier for new users to find out which functions they should try out for a
  specific task. It would also help the community grow around common APIs,
  where now scikit-image APIs are essentially unique for each function.

More examples can be found in `"API changes for skimage2" on our Wiki <https://github.com/scikit-image/scikit-image/wiki/API-changes-for-skimage2>`_.

To make this transition with a minimum amount of user disruption, this SKIP
proposes releasing a new namespace, ``skimage2``, that would provide the new
API, *but only if users explicitly opt in*. Additionally, by releasing a new
namespace, users could use *both* APIs at the same time, allowing users to
migrate their code progressively.

Related Work
------------

``pandas`` released 1.0.0 in January 2020, including many backwards-incompatible
API changes [3]_. SciPy released version 1.0 in 2017, but, given its stage of
maturity and position at the base of the scientific Python ecosystem, opted not
to make major breaking changes [4]_. However, SciPy has adopted a policy of
adding upper-bounds on dependencies [5]_, acknowledging that the ecosystem as a
whole makes backwards incompatible changes on a 2 version deprecation cycle.

Several libraries have successfully migrated their user community to a new
namespace with a version number on it, such as OpenCV (imported as ``cv2``) and
BeautifulSoup (imported as ``bs4``), Jinja (``jinja2``) and psycopg (currently
imported as ``psycopg2``). Further afield, R's ggplot is used as ``ggplot2``.

.. _skip4_implementation:

Implementation
--------------

As a first execution step of this SKIP, scikit-image 1.0 will be released, celebrating the maturity of the project.

First phase: Building `skimage2`
................................

Afterward, a new empty ``skimage2`` namespace will be created in our repository alongside the ``skimage`` namespace.
It will be marked as experimental – importing it will warn that content in ``skimage2`` is still unstable.
This namespace will not be included in `final releases <https://packaging.python.org/en/latest/specifications/version-specifiers/#final-releases>`_ (versioned 1.x) on PyPI and elsewhere but may already be included in our nightly releases.
Before the end of this phase, ``skimage2`` should be made available in `pre-releases <https://packaging.python.org/en/latest/specifications/version-specifiers/#pre-releases>`_ (2.0.0rcN or similar).

With the new namespace available, we will start building the new API inside it.
It will – where possible – wrap the implementation existing in the ``skimage`` namespace but will have its own independent test suite.

While ``skimage2`` will be a new API, we will try to keep the differences from the old API reasonably small to make the eventual transition easier for users.
As a general rule, it should always be possible to achieve the current behavior of the ``skimage`` API by some call or set of calls with the ``skimage2`` API.
There may be some situations where we have to break this general rule, but an argument should be made for the relevant change that breaks this rule.

We will record the pathway for migrating from the old to the new API in detail in a migration guide.
Additionally, deprecation warnings will be implemented for each API change.
These warnings should be ignored by default and should not be shown to users yet.
Users would have no means to act on these warnings because the ``skimage2`` namespace is not available yet.

During this phase, new (additional) features can still be introduced into the old ``skimage`` namespace, not only in the new one.


Second phase: Transitioning to `skimage2`
.........................................

Once we consider the API in ``skimage2`` complete and stable, it will be included in a "final release" versioned 2.0.0.
From now on importing ``skimage2`` is encouraged, and no warnings will be raised.
Instead, we will mark the API in ``skimage`` as deprecated by making deprecation messages from the first phase visible.

On completion of each of these deprecations, we will remove the internal implementation from the old ``skimage`` namespace and move them to the ``skimage2`` namespace.
This can happen over one or multiple releases.

Once the ``skimage`` namespace is empty, it will be removed.


Code translation helper
.......................

Before switching to the second phase, we will look into implementing a code translation tool to help users automate the transition to ``skimage2``.
This should alleviate the cost and work involved for switching – especially in cases that can be easily automated.
Still, this tool might not support more ambiguous or complex updates of our API, or all the complex ways in which users might use our library.
Supporting these cases might be impossible or might require prohibitive development effort.
Therefore, users and downstream libraries must always have other means of completing the transition manually, e.g., with the help of conventional deprecation warnings.

If this tool is successfully implemented, it will be included at the start of the second phase as an `entry point <https://packaging.python.org/en/latest/specifications/entry-points/>`_ alongside ``skimage2``.

Backward compatibility
----------------------

This proposal breaks backward compatibility in numerous places in the library.
However, it does so in a new namespace, so that this proposal does not raise
backward compatibility concerns for our users. That said, the authors will
attempt to limit the number of backward incompatible changes to those likely to
substantially improve the overall user experience. It is anticipated that
porting ``skimage`` code to ``skimage2`` will be a straightforward process
and we will publish a user guide for making the transition by the time of
the ``skimage2`` release. Users will be notified about these resources - among
other things - by a warning in scikit-image 1.1.

Alternatives
------------

Releasing the new API in the same package using semantic versioning
...................................................................

This is :ref:`SKIP-3 <skip_3_transition_v1>`, which was rejected after discussion
with the community.

Continuous deprecation over multiple versions
.............................................

This transition could occur gradually over many versions. For example, for
functions automatically converting and rescaling float inputs, we could add a
``preserve_range`` keyword argument that would initially default to False, but
the default value of False would be deprecated, with users getting a warning to
switch to True. After the switch, we could (optionally) deprecate the
argument, arriving, after a further two releases, at the same place:
scikit-image no longer rescales data automatically, there are no
unnecessary keyword arguments lingering all over the API.

Of course, this kind of operation would have to be done simultaneously over all
of the above proposed changes.

Ultimately, the core team felt that this approach generates more work for both
the scikit-image developers and the developers of downstream libraries, for
dubious benefit: ultimately, later versions of scikit-image will still be
incompatible with prior versions, although over a longer time scale.

A new package name
..................

Since the import name is changing, it would be possible to also change the package name from ``scikit-image`` to ``skimage2``.
This option was a previous proposal of this SKIP.
It shares many of the same strengths as the current proposal – chiefly – the new ``skimage2`` namespace.
This option also requires informing users about the new package.
Similarly to the suggestion here, we could raise a warning when the old package is imported.
It could advise users to install the new package.

However, managing and releasing two packages from the same repository is problematic.
At the same time, introducing a new repository would eventually leave behind issues and pull requests and would also make it prohibitively difficult to implement one API as a wrapper around the other.

Not making the proposed API changes
...................................

Another possibility is to reject backwards incompatible API changes outright,
except in extreme cases. The core team feels that this is essentially
equivalent to pinning the library at 0.19.

"scikit-image2" as the new package name
.......................................

The authors acknowledge that the new names should be chosen with care to keep
the disruption to scikit-image's user base and community as small as possible.
However, to protect users without upper version constraints from accidentally
upgrading to the new API, the package name ``scikit-image`` must be changed.
Changing the import name ``skimage`` is similarly advantageous because it allows
using both APIs in the same environment.

This document suggests just ``skimage2`` as the single new name for
scikit-image's API version 2.0, both for the import name and the name on PyPI,
conda-forge and elsewhere. The following arguments were given in favor of this:

- Only one new name is introduced with the project thereby keeping the number of
  associated names as low as possible.
- With this change, the import and package name match.
- Users might be confused whether they should install ``scikit-image2`` or
  ``scikit-image-2``. It was felt that ``skimage2`` avoids this confusion.
- Users who know what ``skimage`` is and see ``skimage2`` in an install
  instruction somewhere, will likely be able to infer that it is a newer version
  of the package.
- It is unlikely that users will be aware of the new API 2.0 but not of the new
  package name. A proposed release of scikit-image 1.1 might point users to
  ``skimage2`` during the installation and update process and thereby clearly
  communicate the successors name.

The following arguments were made against naming the package ``skimage2``:

- According to the "Principle of least astonishment", ``scikit-image2`` might be
  considered the least surprising evolution of the package name.
- It breaks with the convention that is followed by other scikits including
  scikit-image. (It was pointed out that this convention has not been true for
  some time and introducing a version number in the name is a precedent anyway.)

The earlier section "Related Work" describes how other projects dealt with
similar problems.

Discussion
----------

This SKIP is the result of many evolving discussions among the core team, with fellow projects, and with our user base:

- :ref:`SKIP-3 <skip_3_transition_v1>` was an earlier iteration of this SKIP.
  See the "Resolution" section of that document for further background on the motivation for this SKIP.
- `A pragmatic pathway towards skimage2 <https://discuss.scientific-python.org/t/a-pragmatic-pathway-towards-skimage2/530>`_
- Many discussions happened in `issues and pull requests tagged as "Path to skimage2" <https://github.com/scikit-image/scikit-image/pulls?q=label%3A%22%3Ahiking_boot%3A+Path+to+skimage2%22+>`_.

Resolution
----------



References and Footnotes
------------------------

All SKIPs should be declared as dedicated to the public domain with the CC0
license [1]_, as in `Copyright`, below, with attribution encouraged with CC0+BY
[2]_.

.. [1] CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
   https://creativecommons.org/publicdomain/zero/1.0/
.. [2] https://dancohen.org/2013/11/26/cc0-by/
.. [3] https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.0.0.html#backwards-incompatible-api-changes
.. [4] https://docs.scipy.org/doc/scipy/reference/release.1.0.0.html
.. [5] https://github.com/scipy/scipy/pull/12862
.. [6] https://semver.org/
.. [7] https://github.com/scikit-image/scikit-image/issues/5439

Copyright
---------

This document is dedicated to the public domain with the Creative Commons CC0
license [1]_. Attribution to this source is encouraged where appropriate, as per
CC0+BY [2]_.
