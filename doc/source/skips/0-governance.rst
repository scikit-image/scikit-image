.. _governance:

===========================================
scikit-image governance and decision-making
===========================================

The purpose of this document is to formalize the governance process used by the
scikit-image project, to clarify how decisions are made and how the various
elements of our community interact.

This is a consensus-based community project. Anyone with an interest in the
project can join the community, contribute to the project design, and
participate in the decision making process. This document describes how that
participation takes place, how to find consensus, and how deadlocks are
avoided.

Roles And Responsibilities
==========================

Contributors
------------
Contributors are community members who contribute in concrete ways to the
project. Anyone can become a contributor, and contributions can take many forms
– not only code – as detailed in the
:ref:`contributors guide <howto_contribute>`.

Core developers
---------------
Core developers are community members that have demonstrated continued
commitment to the project through ongoing engagement with the community. They
have shown they can be trusted to maintain scikit-image with care. Being a core
developer allows contributors to more easily carry on with their project
related activities by giving them direct access to the project’s repository and
is represented as being an organization member on the scikit-image
`GitHub organization <https://github.com/orgs/scikit-image/people>`_.
Core developers are expected to review code contributions, can merge approved
pull requests, can cast votes for and against merging a pull-request, and can
be involved in deciding major changes to the API. Core developers are expected
to adhere to the :ref:`core developer guide <core_dev>`.

New core developers can be nominated by any existing core developer. Once they
have been nominated, there will be a vote by the current core developers.
Voting on new core developers is one of the few activities that takes place on
the project's private management list. While it is expected that most votes
will be unanimous, a two-thirds majority of the cast votes is enough. The vote
needs to be open for at least 1 week.

Core developers that have not contributed to the project (commits or GitHub
comments) in the past 12 months will be asked if they want to become emeritus
core developers and rescind their commit and voting rights until they become
active again.

Technical Committee
-------------------
The Technical Committee (TC) members are core developers who have additional
responsibilities to ensure the smooth running of the project. TC members are
expected to participate in strategic planning, and approve changes to the
governance model. The purpose of the TC is to ensure a smooth progress from the
big-picture perspective. Changes that impact the full project require a
synthetic analysis and a consensus that is both explicit and informed. When the
core developer community (including the TC members) fails to reach such a
consensus in a reasonable timeframe, the TC is the entity to resolve the issue.

Membership of the TC is by nomination by a core developer. A nomination will
result in discussion that cannot take more than a month and then a vote by
the core developers that will stay open for a week. TC membership votes are
subject to a two-third majority of all cast votes as well as a simple majority
approval of all the current TC members. TC members who do not actively engage
with the TC duties are expected to resign.

The initial Technical Committee of scikit-image consists of …

Decision Making Process
=======================
Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the project `mailing list <mailto:scikit-image@python.org>`_
and the `issue tracker <https://github.com/scikit-image/scikit-image/issues>`_.
Occasionally, sensitive discussion may occur on a private list.

Scikit-image uses a “consensus seeking” process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
At any point during the discussion, any core-developer can call for a vote,
which will conclude one month from the call for the vote. Any vote must be
backed by a :ref:`scikit-image proposal (SKIP) <skip>`. If no option can gather
two thirds of the votes cast, the decision is escalated to the TC, which in
turn will use consensus seeking with the fallback option of a simple majority
vote if no consensus can be found within a month. This is what we hereafter may
refer to as “the decision making process”.

Decisions (in addition to adding core developers and TC membership as above)
are made according to the following rules:

- **Minor documentation changes**, such as typo fixes, or addition / correction of a
  sentence (but no change of the scikit-image.org landing page or the “about”
  page), require approval by a core developer *and* no disagreement or requested
  changes by a core developer on the issue or pull request page (lazy
  consensus). Core developers are expected to give “reasonable time” to others
  to give their opinion on the pull request if they’re not confident others
  would agree.

- **Code changes and major documentation changes** require agreement by *two*
  core developers *and* no disagreement or requested changes by a core developer
  on the issue or pull-request page (lazy consensus).

- **Changes to the API principles and changes to dependencies or supported
  versions** happen via a :ref:`skip` and follow the decision-making process
  outlined above.

- **Changes to the governance model** use the same decision process outlined above.

If a veto -1 vote is cast on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected using
the decision making procedure outlined above.

.. _skip:

Improvement proposals (SKIPs)
=============================
For all votes, a formal proposal must have been made public and discussed before the
vote. The lifetime of a SKIP is as follows:

- A proposal is brought up as either a GitHub issue or a post to the mailing
  list.
- After sufficient discussion, the core advocate(s) of the proposal must consolidate
  the discussion into a single document, with appropriate references to the
  original discussion, representing the pros and cons brought up by each
  participant.
- Such proposal must be a consolidated document, in the form of a
  ‘SciKit-Image Proposal’ (SKIP), rather than a long discussion on a GitHub issue or
  the mailing list.
- To submit a SKIP, you should copy the
  `SKIP template <https://github.com/scikit-image/scikit-image/tree/master/doc/source/skips/template.rst>`_,
  and give it a new name in the same directory, for example,
  ``35-currying-all-functions.rst``. You should then fill in each section with
  appropriate links to prior discussions. Finally, you should submit the added
  file as a pull request (see the :ref:`contributing guide <howto_contribute>`).

