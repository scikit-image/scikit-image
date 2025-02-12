scikit-image 0.25.2rc0
======================

We're happy to announce the release of scikit-image 0.25.2rc0!

Bug Fixes
---------

- Handle random degenerate case in ``skimage.graph.cut_normalized`` gracefully (`#7675 <https://github.com/scikit-image/scikit-image/pull/7675>`_).
- Copy keypoints if necessary to preserve contiguity (`#7692 <https://github.com/scikit-image/scikit-image/pull/7692>`_).
- Revert a previous fix to ``skimage.segmentation.watershed`` that unintentionally changed the algorithm's behavior for markers placed at maxima in the image. We decided that the behavior originally reported as a bug (gh-6632), is not actually one (`#7702 <https://github.com/scikit-image/scikit-image/pull/7702>`_).

Documentation
-------------

- Improve docstring for rolling_ball function (`#7682 <https://github.com/scikit-image/scikit-image/pull/7682>`_).

Infrastructure
--------------

- Only run the job if the PR got merged (vs merely closed) (`#7679 <https://github.com/scikit-image/scikit-image/pull/7679>`_).
- Fix typo in GH workflow (`#7681 <https://github.com/scikit-image/scikit-image/pull/7681>`_).
- Refactor GitHub's CI config and helper scripts (`#7672 <https://github.com/scikit-image/scikit-image/pull/7672>`_).
- Use pytest config in pyproject.toml in CI (`#7555 <https://github.com/scikit-image/scikit-image/pull/7555>`_).
- Lower CI build verbosity (`#7688 <https://github.com/scikit-image/scikit-image/pull/7688>`_).
- Port testing on Windows from Azure CI to GitHub's CI (`#7687 <https://github.com/scikit-image/scikit-image/pull/7687>`_).
- CI cleanup (`#7693 <https://github.com/scikit-image/scikit-image/pull/7693>`_).
- Simultaneously resolve all dependencies; add pip caching (`#7690 <https://github.com/scikit-image/scikit-image/pull/7690>`_).
- Reenable graph reproducibility test (`#7694 <https://github.com/scikit-image/scikit-image/pull/7694>`_).
- Give milestone labeler necessary permissions (`#7695 <https://github.com/scikit-image/scikit-image/pull/7695>`_).
- Milestone labeler permission not needed (`#7696 <https://github.com/scikit-image/scikit-image/pull/7696>`_).
- Fix 313t wheel build (`#7699 <https://github.com/scikit-image/scikit-image/pull/7699>`_).

Maintenance
-----------

- Include a missing image in meson.build so they are included in the wheel (`#7660 <https://github.com/scikit-image/scikit-image/pull/7660>`_).
- Add zizmor to pre-commit; address GH workflow issues raised (`#7662 <https://github.com/scikit-image/scikit-image/pull/7662>`_).

Contributors
------------

6 authors added to this release (alphabetically):

- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- Orion Poplawski (`@opoplawski <https://github.com/opoplawski>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

8 reviewers added to this release (alphabetically):

- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- Ralf Gommers (`@rgommers <https://github.com/rgommers>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

_These lists are automatically generated, and may not be complete or may contain
duplicates._
