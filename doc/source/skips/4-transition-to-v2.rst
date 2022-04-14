.. _skip_4_transition_v2:

==========================================
SKIP 4 — Transitioning to scikit-image 2.0
==========================================

:Author: Juan Nunez-Iglesias <juan.nunez-iglesias@monash.edu>
:Status: Draft
:Type: Standards Track
:Created: 2022-04-08
:Resolved: 
:Resolution: 
:Version effective: None

Abstract
--------

scikit-image is preparing to release version 1.0. This `was seen
<skip_3_transition_v1>_` as an opportunity to clean up the API, including
backwards incompatible changes. Some of these changes involve changing return
values without changing function signatures, which can ordinarily only be done
by adding an otherwise useless keyword argument (such as
``new_return_style=True``) whose default value changes over several releases.
The result is *still* a backwards incompatible change, but made over a longer
time period.

Despite being in beta and in a 0.x series of releases, scikit-image is used
extremely broadly, and any backwards incompatible changes are likely to be
disruptive. Given the rejection of `SKIP-3 <skip_3_transition_v1>_`, this
document proposes an alternative pathway to create a new API. The new pathway
involves the following steps:

- Any pending deprecations that were sheduled for v0.20 and v0.21 are 
  finalised (the new API suggested by deprecation messages in v0.19 becomes
  the only API).
- This is released as 1.0.
- At this point, main changes the package and import names to skimage2, and the
  API is free to evolve.

Further motivation for the API changes is explained below, and largely
duplicated from `SKIP-3 <skip_3_transition_v1>_`.

Motivation and Scope
--------------------

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

To make this transition with a minimum amount of user disruption, this SKIP
proposes releasing a new library, skimage2, that would replace the existing
library, *but only if users explicitly opt-in*. Additionally, by releasing a
new library, users could depend *both* on scikit-image (1.0) and on skimage2,
allowing users to migrate their code progressively.

Related Work
------------

`pandas` released 1.0.0 in January 2020, including many backwards-incompatible
API changes [3]_. `scipy` released version 1.0 in 2017, but, given its stage of
maturity and position at the base of the scientific Python ecosystem, opted not
to make major breaking changes [4]_. However, SciPy has adopted a policy of
adding upper-bounds on dependencies [5]_, acknowledging that the ecosystem as a
whole makes backwards incompatible changes on a 2 version deprecation cycle.

Several libraries have successfully migrated their user community to a new
namespace with a version number on it, such as OpenCV (imported as ``cv2``) and
BeautifulSoup (imported as ``bs4``), Jinja (``jinja2``) and psycopg (currently
imported as ``psycopg2``). Further afield, R's ggplot is used as ``ggplot2``.

Implementation
--------------

The details of the proposal are as follows:

- scikit-image 0.19 will be followed by scikit-image 1.0. Every deprecation
  message will be removed from 1.0, and the API will be considered the
  scikit-image 1.0 API.
- After 1.0, the main branch will be changed to (a) change the import name to
  skimage2, (b) change the package name to skimage2, and (c) change the version
  number to 2.0-dev.
- There will be *no* scikit-image package on PyPI with version 2.0. Users who
  ``pip install scikit-image`` will always get the 1.0 version of the package.
- After consensus has been reached on the new API, skimage2 will be released.
- scikit-image 1.0.x will receive critical bug fixes for an unspecified period
  of time, depending on the severity of the bug and the amount of effort
  involved.

Backward compatibility
----------------------

This proposal breaks backward compatibility in numerous places in the library.
However, it does so in a new namespace, so that this proposal does not raise
backward compatibilty concerns for our users. That said, the authors will
attempt to limit the number of backward incompatible changes to those likely to
substantially improve the overall user experience. It is anticipated that
porting `skimage` code to `skimage2` will be a straightforward process
and we will publish a user guide for making the transition by the time of
the `skimage2` release.

Alternatives
------------

Releasing the new API in the same package using semantic versioning
...................................................................

This is `SKIP-3 <skip_3_transition_v1>_`, which was rejected after discussion
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

A single package containing both versions
.........................................

Since the import name is changing, it would be possible to make a single
package with both the ``skimage`` and ``skimage2`` namespaces shipping
together, at least for some time. This option is attractive but it implies
longer-term maintenance of the 1.0 namespace, for which we might lack
maintainer time, or a long deprecation cycle for the 1.0 namespace, which would
ultimately result in a lot of unhappy users getting deprecation messages from
their scikit-image use.

Not making the proposed API changes
...................................

Another possibility is to reject backwards incompatible API changes outright,
except in extreme cases. The core team feels that this is essentially
equivalent to pinning the library at 0.19.

Discussion
----------

This SKIP is the result of discussion of `SKIP-3 <skip_3_transition_v1>_`. See
the "Resolution" section of that document for further background on the
motivation for this SKIP.

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
