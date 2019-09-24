.. _governance:

:Author: Juan Nunez-Iglesias <juan.nunez-iglesias@monash.edu>
:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Author: Josh Warner
:Author: François Boulogne
:Author: Emmanuelle Gouillart
:Author: Mark Harfouche
:Author: Lars Grüter
:Author: Egor Panfilov
:Status: Draft
:Type: Process
:Created: 2019-07-02
:Resolved:
:Resolution:
:Version effective: 0.16

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
resolved.

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
have shown they can be trusted to maintain scikit-image with care. Becoming a
core developer allows contributors to merge approved pull requests, cast votes
for and against merging a pull-request, and be involved in deciding major
changes to the API, and thereby more easily carry on with their project related
activities. Core developers appear as organization members on the scikit-image
`GitHub organization <https://github.com/orgs/scikit-image/people>`_. Core
developers are expected to review code contributions while adhering to the
:ref:`core developer guide <core_dev>`.

New core developers can be nominated by any existing core developer.
Discussion about new core developer nominations is one of the few activities
that takes place on the project's private management list. The decision to
invite a new core developer must be made by “lazy consensus”, meaning unanimous
agreement by all responding existing core developers. Invitation must take
place at least one week after initial nomination, to allow existing members
time to voice any objections.

Steering Council
----------------
The Steering Council (SC) members are core developers who have additional
responsibilities to ensure the smooth running of the project. SC members are
expected to participate in strategic planning, approve changes to the
governance model, and make decisions about funding granted to the project
itself. (Funding to community members is theirs to pursue and manage.) The
purpose of the SC is to ensure smooth progress from the big-picture
perspective. Changes that impact the full project require a synthetic analysis
and a consensus that is both explicit and informed. When the core developer
community (including the SC members) fails to reach such a consensus in a
reasonable timeframe, the SC is the entity that resolves the issue.

The steering council is fixed in size to five members. This may be expanded in
the future. The initial steering council of scikit-image consists of Stéfan
van der Walt, Juan Nunez-Iglesias, Emmanuelle Gouillart, Josh Warner, and
Zachary Pincus. The SC membership is revisited every January. SC members who do
not actively engage with the SC duties are expected to resign. New members are
added by nomination by a core developer. Nominees should have demonstrated
long-term, continued commitment to the project and its :ref:`values`. A
nomination will result in discussion that cannot take more than a month and
then admission to the SC by consensus.


Decision Making Process
=======================
Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the project `mailing list <mailto:scikit-image@python.org>`_
and the `issue tracker <https://github.com/scikit-image/scikit-image/issues>`_.
Occasionally, sensitive discussion may occur on a private list.

Decisions should be made in accordance with the :ref:`mission, vision and
values <values>` of the scikit-image project.

Scikit-image uses a “consensus seeking” process for making decisions. The group
tries to find a resolution that has no open objections among core developers.
Core developers are expected to distinguish between fundamental objections to a
proposal and minor perceived flaws that they can live with, and not hold up the
decision-making process for the latter.  If no option can be found without
objections, the decision is escalated to the SC, which will itself use
consensus seeking to come to a resolution. In the unlikely event that there is
still a deadlock, the proposal will move forward if it has the support of a
simple majority of the SC. Any vote must be backed by a :ref:`scikit-image
proposal (SKIP) <skip>`.

Decisions (in addition to adding core developers and SC membership as above)
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

- **Changes to the API principles** require a :ref:`skip` and follow the
  decision-making process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a :ref:`skip` and follow the decision-making process outlined above,
  *unless* there is unanimous agreement from core developers on the change.

If a veto is cast on a lazy consensus, the proposer can appeal to the
community and core developers and the change can be approved or rejected using
the decision making procedure outlined above.

.. _skip:

Improvement proposals (SKIPs)
=============================

For all votes, a formal proposal must have been made public and discussed
before the vote. After discussion has taken place, the key advocate of the
proposal must create a consolidated document summarizing the discussion, called
a SKIP, on which the core team votes. The lifetime of a SKIP detailed in
:ref:`skip0`.

A list of all existing SKIPs is available :ref:`here <skip_list>`.

Copyright
=========

This document is based on the `scikit-learn governance document
<https://scikit-learn.org/stable/governance.html>`_ and is placed in the public
domain.
