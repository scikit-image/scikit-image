(core_dev)=

# Core Developer Guide

Welcome, new core developer! The core team appreciate the quality of
your work, and enjoy working with you; we have therefore invited you
to join us. Thank you for your numerous contributions to the project
so far.

This document offers guidelines for your new role. First and
foremost, you should familiarize yourself with the project's
{doc}`mission, vision, and values <../about/values>`. When in
doubt, always refer back here.

As a core team member, you gain the responsibility of shepherding
other contributors through the review process; here are some
guidelines.

## All Contributors Are Treated The Same

You now have the ability to push changes directly to the main
branch, but should never do so; instead, continue making pull requests
as before and in accordance with the
{doc}`general contributor guide <contribute>`.

As a core contributor, you gain the ability to merge or approve
other contributors' pull requests. Much like nuclear launch keys, it
is a shared power: you must merge _only after_ another core has
approved the pull request, _and_ after you yourself have carefully
reviewed it. (See {ref}`sec:reviewing` and especially
{ref}`sec:understand` below.) To ensure a clean git history,
use GitHub's [Squash and Merge][gh_sqmrg]
feature to merge, unless you have a good reason not to do so.

[gh_sqmrg]: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/merging-a-pull-request#merging-a-pull-request-on-github

(sec:reviewing)=

## Reviewing

### How to Conduct A Good Review

_Always_ be kind to contributors. Nearly all of `scikit-image` is
volunteer work, for which we are tremendously grateful. Provide
constructive criticism on ideas and implementations, and remind
yourself of how it felt when your own work was being evaluated as a
novice.

`scikit-image` strongly values mentorship in code review. New users
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
   should be simple, [functional][wiki_functional] (i.e. not
   carry state), consistent with other parts of the library, and
   should avoid modifying input variables. Please familiarize
   yourself with the project's [deprecation policy][dep_pol]

2. **Documentation:** Any new feature should have a gallery
   example, that not only illustrates but explains it.

3. **The algorithm:** You should understand the code being modified or
   added before approving it. (See {ref}`sec:understand`
   below.) Implementations should do what they claim,
   and be simple, readable, and efficient.

4. **Tests:** All contributions to the library _must_ be tested, and
   each added line of code should be covered by at least one test. Good
   tests not only execute the code, but explores corner cases. It is tempting
   not to review tests, but please do so.

5. **Licensing:** New contributions should be available under the same license
   as or be compatible with {doc}`scikit-image's license <../license>`.
   Examples of BSD-compatible licenses are the [MIT License][mit_license] and
   [Apache License 2.0][apache_2-2]. When in doubt, ask the team for help.
   If you, the contributor, are not the copyright holder of the submitted
   code, please ask the original authors for approval and include their names
   in `LICENSE.txt`. You can use the other entries in that file as templates.

6. **Established methods:** In general, we are looking to include algorithms
   and methods which are established, well documented in the literature and
   widely used by the imaging community. While this is not a hard requirement,
   new contributions should be consistent with {doc}`our mission <../about/values>`.

[wiki_functional]: https://en.wikipedia.org/wiki/Functional_programming
[dep_pol]: https://scikit-image.org/docs/dev/development/contribute.html#deprecation-cycle
[mit_license]: https://spdx.org/licenses/MIT.html
[apache_2-2]: https://spdx.org/licenses/Apache-2.0.html

Other changes may be _nitpicky_: spelling mistakes, formatting,
etc. Do not ask contributors to make these changes, and instead
make the changes by [pushing to their branch][gh_push]
or using GitHub’s [suggestion][gh_suggest] [feature][gh_feedback].
(The latter is preferred because it gives the contributor a choice in
whether to accept the changes.)

[gh_push]: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/committing-changes-to-a-pull-request-branch-created-from-a-fork
[gh_suggest]: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request
[gh_feedback]: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request

Our default merge policy is to squash all PR commits into a single
commit. Users who wish to bring the latest changes from `main`
into their branch should be advised to merge, not to rebase. Even
when merge conflicts arise, don’t ask for a rebase unless you know
that a contributor is experienced with git. Instead, rebase the branch
yourself, force-push to their branch, and advise the contributor on
how to force-pull. If the contributor is no longer active, you may
take over their branch by submitting a new pull request and closing
the original. In doing so, ensure you communicate that you are not
throwing the contributor's work away!

Please add a note to a pull request after you push new changes; GitHub
does not send out notifications for these.

(sec:understand)=

### Merge Only Changes You Understand

_Long-term maintainability_ is an important concern. Code doesn't
merely have to _work_, but should be _understood_ by multiple core
developers. Changes will have to be made in the future, and the
original contributor may have moved on.

Therefore, _do not merge a code change unless you understand it_. Ask
for help freely: we have a long history of consulting community
members, or even external developers, for added insight where needed,
and see this as a great learning opportunity.

While we collectively "own" any patches (and bugs!) that become part
of the code base, you are vouching for changes you merge. Please take
that responsibility seriously.

In practice, if you are the second core developer reviewing and approving a
given pull request, you typically merge it (again, using GitHub's Squash and
Merge feature) in the wake of your approval. What are the exceptions to this
process? If the pull request has been particularly controversial or the
subject of much debate (e.g., involving API changes), then you would want to
wait a few days before merging. This waiting time gives others a chance to
speak up in case they are not fine with the current state of the pull request.
Another exceptional situation is one where the first approving review happened
a long time ago and many changes have taken place in the meantime.

When squashing commits GitHub concatenates all commit messages.
Please edit the resulting message so that it gives a concise, tidy
overview of changes. For example, you may want to grab the
description from the PR itself, and delete lines such as "pep8 fix",
"apply review comments", etc. Please retain all Co-authored-by
entries.

## Closing issues and pull requests

Sometimes, an issue must be closed that was not fully resolved. This can be
for a number of reasons:

- the person behind the original post has not responded to calls for
  clarification, and none of the core developers have been able to reproduce
  their issue;
- fixing the issue is difficult, and it is deemed too niche a use case to
  devote sustained effort or prioritize over other issues; or
- the use case or feature request is something that core developers feel
  does not belong in scikit-image,

among others. Similarly, pull requests sometimes need to be closed without
merging, because:

- the pull request implements a niche feature that we consider not worth the
  added maintenance burden;
- the pull request implements a useful feature, but requires significant
  effort to bring up to scikit-image's standards, and the original
  contributor has moved on, and no other developer can be found to make the
  necessary changes; or
- the pull request makes changes that do not align with our values, such as
  increasing the code complexity of a function significantly to implement a
  marginal speedup,

among others.

All these may be valid reasons for closing, but we must be wary not to alienate
contributors by closing an issue or pull request without an explanation. When
closing, your message should:

- explain clearly how the decision was made to close. This is particularly
  important when the decision was made in a community meeting, which does not
  have as visible a record as the comments thread on the issue itself;
- thank the contributor(s) for their work; and
- provide a clear path for the contributor or anyone else to appeal the
  decision.

These points help ensure that all contributors feel welcome and empowered to
keep contributing, regardless of the outcome of past contributions.

## Further resources

As a core member, you should be familiar with community and developer
resources such as:

- Our {doc}`contributor guide <contribute>`
- Our [community guidelines](https://scikit-image.org/community_guidelines.html)
- [PEP8](https://www.python.org/dev/peps/pep-0008/) for Python style
- [PEP257](https://www.python.org/dev/peps/pep-0257/) and the
  [NumPy documentation guide][numpydoc]
  for docstrings. (NumPy docstrings are a superset of PEP257. You
  should read both.)
- The scikit-image [tag on StackOverflow][so_tag]
- The scikit-image [tag on forum.image.sc](https://forum.image.sc/tags/scikit-image)
- Our [developer forum][ml]
- Our [chat room](https://skimage.zulipchat.com/)

[numpydoc]: https://docs.scipy.org/doc/numpy/docs/howto_document.html
[so_tag]: https://stackoverflow.com/questions/tagged/scikit-image
[ml]: https://discuss.scientific-python.org/c/contributor/skimage

You are not required to monitor all of the social resources.

## Inviting New Core Members

Any core member may nominate other contributors to join the core team.
Nominations happen on a private email list,
<skimage-core@discuss.scientific-python.org>. As of this writing, there is no hard-and-fast
rule about who can be nominated; at a minimum, they should have: been
part of the project for at least six months, contributed
significant changes of their own, contributed to the discussion and
review of others' work, and collaborated in a way befitting our
community values.

## Contribute To This Guide!

This guide reflects the experience of the current core developers. We
may well have missed things that, by now, have become second
nature—things that you, as a new team member, will spot more easily.
Please ask the other core developers if you have any questions, and
submit a pull request with insights gained.

## Conclusion

We are excited to have you on board! We look forward to your
contributions to the code base and the community. Thank you in
advance!
