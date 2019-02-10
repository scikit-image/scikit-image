Core Developer Guide
====================

Welcome, new core developer!  The core team appreciate the quality of
your work, and enjoy working with you; we have therefore invited you
to join us.  Thank you for your numerous contributions to the project
so far.

This document offers guidelines for your new role.  First and
foremost, you should familiarize yourself with the project's
`mission, vision, and values
<https://github.com/scikit-image/scikit-image/pull/3585>`__.  When in
doubt, always refer back here.

As a core team member, you gain the responsibility of shepherding
other contributors through the review process; here are some
guidelines.

All Contributors Are Treated The Same
-------------------------------------

You now have the ability to push changes directly to the master
branch, but should never do so; instead, continue making pull requests
as before and in accordance with the `general contributor guide
<http://scikit-image.org/docs/dev/contribute.html>`__.

As a core contributor, you gain the ability to merge or approve
other contributors' pull requests.  Much like nuclear launch keys, it
is a shared power: you must merge *only after* another core has
approved the pull request, *and* after you yourself have carefully
reviewed it.  (See `Reviewing`_ and especially `Merge Only Changes You
Understand`_ below.) To ensure a clean git history, use GitHub's
`Squash and Merge <https://help.github.com/articles/merging-a-pull-request/#merging-a-pull-request-on-github>`__
feature to merge, unless you have a good reason not to do so.

Reviewing
---------

How to Conduct A Good Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Always* be kind to contributors. Nearly all of `scikit-image` is
volunteer work, for which we are tremendously grateful. Provide
constructive criticism on ideas and implementations, and remind
yourself of how it felt when your own work was being evaluated as a
novice.

`scikit-image` strongly values mentorship in code review.  New users
often need more handholding, having little to no git
experience. Repeat yourself liberally, and, if you don’t recognize a
contributor, point them to our development guide, or other GitHub
workflow tutorials around the web. Do not assume that they know how
GitHub works (e.g., many don't realize that adding a commit
automatically updates a pull request). Gentle, polite, kind
encouragement can make the difference between a new core developer and
an abandoned pull request.

When reviewing, focus on the following:

1. **API:** The API is what users see when they first use
   `scikit-image`. APIs are difficult to change once released, so
   should be simple, `functional
   <https://en.wikipedia.org/wiki/Functional_programming>`__ (i.e. not
   carry state), consistent with other parts of the library, and
   should avoid modifying input variables.  Please familiarize
   yourself with the project's `deprecation policy <http://scikit-image.org/docs/dev/contribute.html#deprecation-cycle>`__.

2. **Documentation:** Any new feature should have a gallery
   example, that not only illustrates but explains it.

3. **The algorithm:** You should understand the code being modified or
   added before approving it.  (See `Merge Only Changes You
   Understand`_ below.) Implementations should do what they claim,
   and be simple, readable, and efficient.

4. **Tests:** All contributions to the library *must* be tested, and
   each added line of code should be covered by at least one test. Good
   tests not only execute the code, but explores corner cases.  It is tempting
   not to review tests, but please do so.

Other changes may be *nitpicky*: spelling mistakes, formatting,
etc. Do not ask contributors to make these changes, and instead
make the changes by `pushing to their branch
<https://help.github.com/articles/committing-changes-to-a-pull-request-branch-created-from-a-fork/>`__,
or using GitHub’s `suggestion
<https://help.github.com/articles/commenting-on-a-pull-request/>`__
`feature
<https://help.github.com/articles/incorporating-feedback-in-your-pull-request/>`__.
(The latter is preferred because it gives the contributor a choice in
whether to accept the changes.)

Unless you know that a contributor is experienced with git, don’t
ask for a rebase when merge conflicts arise. Instead, rebase the
branch yourself, force-push to their branch, and advise the contributor to force-pull.  If the contributor is
no longer active, you may take over their branch by submitting a new pull
request and closing the original. In doing so, ensure you communicate
that you are not throwing the contributor's work away!

Please add a note to a pull request after you push new changes; GitHub
does not send out notifications for these.

Merge Only Changes You Understand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Long-term maintainability* is an important concern.  Code doesn't
merely have to *work*, but should be *understood* by multiple core
developers.  Changes will have to be made in the future, and the
original contributor may have moved on.

Therefore, *do not merge a code change unless you understand it*. Ask
for help freely: we have a long history of consulting community
members, or even external developers, for added insight where needed,
and see this as a great learning opportunity.

While we collectively "own" any patches (and bugs!) that become part
of the code base, you are vouching for changes you merge.  Please take
that responsibility seriously.

Further resources
-----------------

As a core member, you should be familiar with community and developer
resources such as:

-  Our `contributor
   guide <http://scikit-image.org/docs/stable/contribute.html>`__
-  Our `community
   guidelines <https://scikit-image.org/community_guidelines.html>`__
-  `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ for Python style
-  `PEP257 <https://www.python.org/dev/peps/pep-0257/>`__ and the `NumPy
   documentation
   guide <https://docs.scipy.org/doc/numpy/docs/howto_document.html>`__
   for docstrings. (NumPy docstrings are a superset of PEP257. You
   should read both.)
-  The scikit-image `tag on
   StackOverflow <https://stackoverflow.com/questions/tagged/scikit-image>`__
-  The scikit-image `tag on
   forum.image.sc <https://forum.image.sc/tags/scikit-image>`__
-  Our `mailing
   list <https://mail.python.org/mailman/listinfo/scikit-image>`__
-  Our `chat room <https://skimage.zulipchat.com/>`__

You are not required to monitor all of the social resources.

Inviting New Core Members
-------------------------

Any core member may nominate other contributors to join the core team.
Nominations happen on a private email list,
skimage-core@python.org. As of this writing, there is no hard-and-fast
rule about who can be nominated; at a minimum, they should have: been
part of the project for at least six months, contributed
significant changes of their own, contributed to the discussion and
review of others' work, and collaborated in a way befitting our
community values.

Contribute To This Guide!
-------------------------

This guide reflects the experience of the current core developers.  We
may well have missed things that, by now, have become second
nature—things that you, as a new team member, will spot more easily.
Please ask the other core developers if you have any questions, and
submit a pull request with insights gained.

Conclusion
----------

We are excited to have you on board!  We look forward to your
contributions to the code base and the community.  Thank you in
advance!
