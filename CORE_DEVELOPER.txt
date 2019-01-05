So you became a scikit-image core developer
===========================================

Thank you for your numerous contributions to the project so far. The core
team has been so impressed by your work that they invited you to join them!
Congratulations and thank you again!

This document lays out some guidelines to keep in mind while fulfilling your
role as a core developer. As you know, anyone can contribute to scikit-image
at any time via a GitHub pull request. As a core team member, your added
responsibility is to shepherd those contributions to fulfill the [mission,
vision, and values][1] of the scikit-image project.

This document sets out some guidelines for how best to do that.

The development model stays the same
------------------------------------

Although you now have the ability to push changes directly to the master
branch, all core developers have agreed to forgo this privilege, and instead
make changes via one of two modes:

1. Submit your own code as a pull request, and work with two other core
   developers to merge it. This is the same process as all contributors.
2. Merge or approve somebody else's GitHub pull request. This is the new power
   you get by becoming a core developer, and, much like nuclear launch keys,
   it is a shared power: you must only merge *after* another core has approved
   the PR, and after you yourself have carefully reviewed it as well. (See
   "Reviewing" and especially "If you merge it, you buy it", below.)

Reviewing
---------

## About reviews

Always be kind to contributors, past and present. Nearly all of scikit-image
comes from volunteer work, not directly supported, so its very existence is
something to be grateful for and even marvel at. This is true even if you
disagree with some prior API decision or are frustrated by a bug that crept in
despite the review process. Feel free to criticize ideas and implementations,
but never be insulting to the authors of the code being modified or the code
being contributed.

scikit-image strongly values mentorship in code review, and this often means
handholding new contributors, who might have little to no git experience,
through the revision process. Repeat yourself liberally, and, if you don't
recognize a contributor, point them to our development guide, or other GitHub
workflow tutorials around the web. Don't assume that they know that making
another commit will automatically update a pull request. Gentle, polite, kind
encouragement can be the difference between gaining a new core developer or
gaining a new abandoned pull request. (We have lots of those but we are not
collecting them!)

When reviewing, try to focus most on the API, that is, the part of the pull
request that new users of scikit-image will see. This is difficult and time
consuming to change once it has been released, so it's the most important thing
to get right from the beginning. Is the API as simple as possible? Is it
[functional][15] rather than stateful? Does it avoid modifying input variables?
Is it consistent with other, similar functions in scikit-image?

Next in importance, the documentation: any new feature must come with a gallery
example that shows it off. Make sure that the example doesn't just use the new
functionality, but explains it as deeply as possible. Would a new user get the
idea of the function straight away while reading the example?

Next, the algorithm itself: make sure you understand what the code being
modified or added does. (See "if you merge it, you buy it", below.) Does the
algorithm do what it claims to do in the documentation? Does it do so in a
reasonably effective and readable way, or can you immediately see how to
simplify it?

In all of the above cases, you should provide guidance to the contributor so
that they can meet your requirements. Contributions to scikit-image should meet
all of the above requirements, and come with tests (though we have automated
tools to check for testing, generally, so it is less of a worry for reviewers).

There are further changes that are *nitpicky*: spelling mistakes, formatting, a
missing space. As much as possible, you should avoid asking contributors to
change these things, and instead either push to their branch ([Ref1][2],
[Ref2][3]) or, better yet, use GitHub's [suggestion][4] [feature][5], to make
the changes yourself. (The latter is preferred because it gives the contributor
the ability to see the change and either learn from it in accepting it, or
defend their original code.) The exception to this is in contributions from
other core contributors: they have demonstrated they have the skills and
confidence to fend for themselves. ;) On the other hand, nitpicky reviews can
be disproportionately discouraging to new contributors, who are rightfully not
impressed that their contribution would be held up over such trivialities.

Similarly, unless you have a strong indication that the contributor is
experienced with git, don't ask them to rebase when there are merge conflicts.
Instead, get the code to the shape you want it in, and then rebase yourself and
either force-push to their branch, or submit a new pull request to supercede
the existing one. (When closing the existing one, make sure to communicate that
you are not throwing the contributor's work away!)

After you have made modifications to a contributor's pull request, make sure
you point them out, as the contributor might miss them, and miss out on a
learning opportunity for future contributions.

## Other core developers

When you are reviewing a pull request from another core developer, you can
usually be more relaxed about finishing their work for them. They have been
through the gauntlet a few times to become core developers, so they know how it
works and have demonstrated enough persistence that they are unlikely to give
up after you remind them to put spaces around an equals sign.

However, you should submit *all* nitpicky comments in a single review, to
minimize the number of iterations that the contributor has to go through before
getting their contribution merged.

## If you merge it, you buy it

A core issue in scikit-image development is *long-term maintainability*. Code
in scikit-image doesn't have to *just* work, it must be understandable by a
large proportion of the core team, because refactoring will inevitably happen
in the future (whether for API changes, performance improvements, new features,
or bug fixes), and this is impossible if you don't understand the code. The
original contributor, in most cases, will have moved on and will be unable to
help at that point.

Therefore, whether you are the first or second reviewer of a pull request, you
should make sure that you *deeply understand* every aspect of it. *If you don't
understand it, don't merge it!* Even if it passes the tests. Instead, ask for
help! scikit-image has a long history of core developers asking contributors or
even external developers for help understanding things. Each case is an
opportunity to grow as an individual and as a team.

We should emphasize that the above relationship to "your" code is one of
*responsibility*, not ownership. Once merged, the code in scikit-image belongs
to the scikit-image contributors as a collective, not to the original author or
to the reviewers.

Further resources
-----------------

As a core member, you should be more familiar than most with all of the
resources and community engagement forums that guide scikit-image development.
Specifically:

- Our [contributor guide][6]
- Our [community guidelines][7]
- [PEP8][8] for style
- [PEP257][9] and the [NumPy documentation guide][10] for docstrings. (NumPy
  docstrings are a superset of PEP257. You should read both.)
- The scikit-image [tag on StackOverflow][11]
- The scikit-image [tag on forum.image.sc][12]
- Our [mailing list][13]
- Our [chat room][14]


Inviting new core members
-------------------------

Any core member may nominate other contributors to join the core team.
Nomination happens on a private email list, skimage-core@python.org. As of this
writing, there is no hard-and-fast rule about who can be nominated, but they
should have at least one accepted, substantial pull request to the library.
"Substantial" is a fuzzy term that includes a large changeset (not just a few
typos), and some back-and-forth discussion and iteration on the contents. It is
also preferable if the nominee has participated in discussion and even review
of others' pull requests.

Contribute to this guide!
-------------------------

This guide was written by core developers after quite some time and experience
being one, so it might be missing critical information that has become second
nature after we acquired it. Therefore, as a newly invited member, you are best
placed to identify gaps in this guide. If you have any questions, please ask
the rest of the team, then submit a pull request to incorporate the answers
into this guide!

Now, we are looking forward to continuing to work with you in the future, and
to seeing your reviews of still newer contributors!

[1]: https://github.com/scikit-image/scikit-image/pull/3585
[2]: https://help.github.com/articles/committing-changes-to-a-pull-request-branch-created-from-a-fork/
[3]: https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/
[4]: https://help.github.com/articles/commenting-on-a-pull-request/
[5]: https://help.github.com/articles/incorporating-feedback-in-your-pull-request/
[6]: http://scikit-image.org/docs/stable/contribute.html
[7]: https://scikit-image.org/community_guidelines.html
[8]: https://www.python.org/dev/peps/pep-0008/
[9]: https://www.python.org/dev/peps/pep-0257/
[10]: https://docs.scipy.org/doc/numpy/docs/howto_document.html
[11]: https://stackoverflow.com/questions/tagged/scikit-image
[12]: https://forum.image.sc/tags/scikit-image
[13]: https://mail.python.org/mailman/listinfo/scikit-image
[14]: https://skimage.zulipchat.com/
[15]: https://en.wikipedia.org/wiki/Functional_programming
