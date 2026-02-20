.. _skip_3_transition_v1:

==========================================
SKIP 3 — Transitioning to scikit-image 1.0
==========================================

:Author: Juan Nunez-Iglesias <juan.nunez-iglesias@monash.edu>
:Status: Final
:Type: Standards Track
:Created: 2021-07-15
:Resolved: 2021-09-13
:Resolution: Rejected
:Version effective: None

Abstract
--------

scikit-image is preparing to release version 1.0. This is potentially an
opportunity to clean up the API, including backwards incompatible changes. Some
of these changes involve changing return values without changing function
signatures, which can ordinarily only be done by adding an otherwise useless
keyword argument (such as ``new_return_style=True``) whose default value
changes over several releases. The result is *still* a backwards incompatible
change, but made over a longer time period.

Despite being in beta and in a 0.x series of releases, scikit-image is used
extremely broadly, and any backwards incompatible changes are likely to be
disruptive. This SKIP proposes a process to ensure that the community is aware
of upcoming changes, and can adapt their libraries *or* their declared
scikit-image version dependencies accordingly.

Motivation and Scope
--------------------

scikit-image has grown organically over the past 12 years, with functionality
being added by a broad community of contributors from different backgrounds.
This has resulted in various parts of the API being inconsistent: for example,
``skimage.transform.warp`` inverts the order of coordinates, so that a
translation of (45, 32) actually moves the values in a NumPy array by 32 along
the 0th axis, and 45 along the 1st, *but only in 2D*.

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

Given the accumulation of potential API changes that have turned out to be too
burdensome and noisy to fix with a standard deprecation cycle, principally
because they involve changing function outputs for the same inputs, it makes
sense to make all those changes in a transition to version 1.0 -- semantic
versioning, which we use, explicitly allows breaking API changes on major
version updates [6]_. However, we must acknowledge that (1) an enormous number
of projects depend on scikit-image and would thus be affected by backwards
incompatible changes, and (2) it is not yet common practice in the scientific
Python community to put upper version bounds on dependencies, so it is very
unlikely that anyone used ``scikit-image<1.*`` in their dependency list (though
this is slowly changing [5]_).

Given the above, we need to come up with a way to notify all our users that
this change is coming, while also allowing them to silence any warnings once
they have been noted.

Detailed description
--------------------

It is beyond the scope of this document to list all of the proposed API changes
for scikit-image 1.0, many of which have yet to be decided upon. Indeed, the
scope and ambition of the 1.0 transition could grow if this SKIP is accepted.
The SKIP instead proposes a mechanism for warning users about upcoming breaking
changes. A meta-issue tracking the proposed changes can be found on GitHub,
scikit-image/scikit-image#5439 [7]_. Some examples are briefly included below
for illustrative purposes:

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
  specific task.

The question is, how do we make this transition while causing as little
disruption as possible?

This document proposes releasing 0.19 as the final 0.x series release, then
immediately releasing a nearly identical 0.20 release that warns users about
breaking changes in 1.0, thus giving them an opportunity to pin their
scikit-image dependency to 0.19.x. The warning would also point users to a
transition guide to prepare their code for 1.0. See `Implementation` for
details.

This approach ensures that all users get ample warning, and a chance to ensure
that their scripts and libraries will continue to work after 1.0 is released.
Users who don't have the time or inclination to make the transition will be
able to pin their dependencies correctly. Those who prefer to be on the cutting
edge will also be able to plan around the 1.0 release and update their code
correctly, in sync with scikit-image.

Related Work
------------

``pandas`` released 1.0.0 in January 2020, including many backwards-incompatible
API changes [3]_. `scipy` released version 1.0 in 2017, but, given its stage of
maturity and position at the base of the scientific Python ecosystem, opted not
to make major breaking changes [4]_. However, SciPy has adopted a policy of
adding upper-bounds on dependencies [5]_, acknowledging that the ecosystem as a
whole makes backwards incompatible changes on a 2 version deprecation cycle.

Implementation
--------------

The details of the proposal are as follows:

- scikit-image 0.19 will be the final *true* 0.x release. It contains some new
  features, bug fixes, and several API changes following on from deprecations
  in 0.17.
- shortly after 0.19, we release 0.20, which is identical except that it emits
  a warning at import time. The warning reads something like the following:
  "scikit-image 1.0 will be released later this year and will contain breaking
  changes. To ensure your code keeps running, please install
  ``scikit-image<=0.19.*``. To silence this warning but still depend on
  scikit-image after 1.0 is released, install ``scikit-image!=0.20.*``." The
  warning also contains a link for further details, and instructions for
  managing the dependency in both conda and pip environments.
- After 0.20, we make all the API changes we need, without deprecation cycles.
  Importantly, for every API change, we add a line to a "scikit-image 1.0
  transition guide" in the documentation, which maps every changed
  functionality in the library from its old form to its new form. These changes
  are tracked on a GitHub issue [7]_ and in the 1.0 milestone [8]_.
- Once the transition has happened in the repository, we release 1.0.0a0, an
  alpha release which contains a global warning pointing to the transition
  guide, as well as all of the new functionality. We also release 0.21, which
  contains the same warning but is functionally identical to 0.19. This gives
  authors who chose to pin to ``scikit-image!=0.20.*`` a chance to make the
  migration to 1.0.
- After at least one month, we release 1.0.
- We continue to maintain a 0.19.x branch with bug fixes for a year, in order
  to give users time to transition to the new API.

Backward compatibility
----------------------

This proposal breaks backwards compatibility in numerous places in the library.

Alternatives
------------

New package naming
..................

Instead of breaking compatibility in the ``scikit-image`` package, we could
leave that package at 0.19, and release a *new* package, e.g.
``scikit-image1``, which starts at 1.0 and imports as ``skimage1``. This would
obviate the need for users to pin their scikit-image version — users depending
on skimage 0.x would be able to use that library "in perpetuity."

Ultimately, the core developers felt that this approach could unnecessarily
fragment the community, between those that continue using 0.19 and those that
shift to 1.0. Ultimately, the transition of downstream code to 1.0 would be
equally painful as the proposed approach, but the pressure to make the switch
would be decreased, as everyone installing ``scikit-image`` would still get the
old version.

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

Not making the proposed API changes
...................................

Another possibility is to reject backwards incompatible API changes outright,
except in extreme cases. The core team feels that this is essentially
equivalent to pinning the library at 0.19.

Discussion
----------

In early July 2021, the core team held a series of meetings to discuss this
approach. The minutes of this meeting are in the scikit-image meeting notes
repository [9]_.

Ongoing discussion will happen on the user forum [10]_, the
developer forum [11]_, and GitHub discussion [7]_. Specific links to relevant
posts will be added to this document before acceptance.

Resolution
----------

This SKIP was discussed most extensively in a thread on the mailing list in
July 2021 [12]_. In the end, many and core developers felt that this plan
posed too big a risk of either changing code behavior silently or eroding
goodwill in the community, or both. Matthew Brett wrote [13]_:

    I'm afraid I wasn't completely sure whether the 1.0 option would
    result in breaking what I call the Konrad Hinsen rule for scientific
    software:

    """
    Under (virtually) no circumstances should new versions of a scientific
    package silently give substantially different results for the same
    function / method call from a previous version of the package.
    """

Matthew further wrote [14]_ that if we *don't* break the Hinsen rule, but
instead break users' unpinned scripts, we will lose a lot of goodwill from the
community:

    If you make all these break (if they are lucky) or give completely
    wrong results, it's hard to imagine you aren't going to cause
    significant damage to the rest-of-iceberg body of users who are not on
    the mailing list.

Riadh Fezzani, one of our core developers, felt strongly that SemVer [6]_ was
sufficient to protect users [15]_:

    In scikit-image, we adopted the semantic versioning as it
    is largely adopted in the engineering community. This convention manages
    API breaking and that's what we are doing by releasing v1.0

Even taking this view, though, it cannot address the issue of external
scikit-image "documentation", such as a decade's worth of accumulated
StackOverflow answers, that would be made obsolete by a breaking 1.0 release,
as pointed out by Josh Warner [16]_:

    It's also worth considering that there is a substantial corpus of
    scikit-image teaching material out there. The majority we do not control,
    so cannot be updated or edited. The first hits on YouTube for tutorials
    are not the most recent, but older ones with lots of views.

Nor can it address the issue of *gradually* migrating a code base from the old
API to the new API, as pointed out by Tom Caswell [17]_:

    Put another way, you do not want to put a graduate student in the position
    of saying "I _want_ to use the new API, but I have 10k LoC of inherited
    code using the old API .....".

Ultimately, all these concerns add up to a compelling case to rejecting the
SKIP. Juan Nunez-Iglesias wrote on the mailing list [18]_:

    My proposal going forward is to reject SKIP-3 and create a SKIP-4 proposing
    the skimage2 package.

The SKIP is therefore rejected.

References and Footnotes
------------------------

All SKIPs should be declared as dedicated to the public domain with the CC0
license [1]_, as in `Copyright`, below, with attribution encouraged with CC0+BY
[2]_.

.. [1] CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
   https://creativecommons.org/publicdomain/zero/1.0/
.. [2] https://dancohen.org/2013/11/26/cc0-by/
.. [3] https://pandas.pydata.org/pandas-docs/stable/whatsnew/v1.0.0.html#backwards-incompatible-api-changes
.. [4] https://docs.scipy.org/doc/scipy/release.1.0.0.html
.. [5] https://github.com/scipy/scipy/pull/12862
.. [6] https://semver.org/
.. [7] https://github.com/scikit-image/scikit-image/issues/5439
.. [8] https://github.com/scikit-image/scikit-image/milestones/1.0
.. [9] https://github.com/scikit-image/meeting-notes/blob/main/2021/july-api-meetings.md
.. [10] https://forum.image.sc/tag/scikit-image
.. [11] https://discuss.scientific-python.org/c/contributor/skimage
.. [12] https://mail.python.org/archives/list/scikit-image@python.org/thread/DSV6PEYVJ4RZRUWWV5SBNF7FFRERTSCF/
.. [13] https://mail.python.org/archives/list/scikit-image@python.org/message/UYARUQM5LBWXIAWBAPNHIQIDRKUUDTEK/
.. [14] https://mail.python.org/archives/list/scikit-image@python.org/message/63ZGG7DY5SWVM62XASHMCPFAG6KPJCMT/
.. [15] https://mail.python.org/archives/list/scikit-image@python.org/message/HXI7YVCN6IFF5TL54JBP5QRUDHKTTYRR/
.. [16] https://mail.python.org/archives/list/scikit-image@python.org/message/HRZGMOJLD2WDIO3JXQV3PRWKIUOVOF7P/
.. [17] https://mail.python.org/archives/list/scikit-image@python.org/message/GFXBQYKDACDCH7BGNEGOU7LKHR2LPFX6/
.. [18] https://mail.python.org/archives/list/scikit-image@python.org/message/5J4W63BXFQTT4GHPTZFH3AM4QHAXOW5R/

Copyright
---------

This document is dedicated to the public domain with the Creative Commons CC0
license [1]_. Attribution to this source is encouraged where appropriate, as per
CC0+BY [2]_.
