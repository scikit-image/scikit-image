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
itself. (Funding to community members is theirs to pursue and
manage.) The purpose of the SC is to ensure smooth progress from the
big-picture perspective. Changes that impact the full project require a
synthetic analysis and a consensus that is both explicit and informed. When the
core developer community (including the SC members) fails to reach such a
consensus in a reasonable timeframe, the SC is the entity to resolve the issue.

Membership of the SC is by nomination by a core developer. A nomination will
result in discussion that cannot take more than a month and then admission to
the SC by consensus. SC members who do not actively engage
with the SC duties are expected to resign.

The initial Steering Council of scikit-image consists of Stéfan van der
Walt, Juan Nunez-Iglesias, and Emmanuelle Gouillart.

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
If no option can be found without objections, the decision is escalated to the
SC, which will itself use consensus seeking to come to a resolution. In the
unlikely event that there is still a deadlock, the proposal will move forward
if it has the support of a simple majority of the SC. Any vote must be
backed by a :ref:`scikit-image proposal (SKIP) <skip>`.

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

- **Changes to the API principles and changes to dependencies or supported
  Python versions** require a :ref:`skip` and follow the decision-making
  process outlined above.

- **Changes to this governance model or our mission, vision, and values**
  require a :ref:`skip` and follow the decision-making process outlined above.

If a veto is cast on a lazy consensus, the proposer can appeal to the
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
- To submit a SKIP, you should copy the `SKIP template
  <https://github.com/scikit-image/scikit-image/tree/master/doc/source/skips/template.rst>`_,
  and give it a new name in the same directory, for example,
  ``35-currying-all-functions.rst``. You should then fill in each section with
  appropriate links to prior discussions. Finally, you should submit the added
  file as a pull request (see the :ref:`contributing guide <howto_contribute>`).
- The SKIP pull request may be merged as “Accepted” before implementation, or
  the implementation may happen on the same PR. Upon implementation, the SKIP
  status should be updated to “Final”.

For a more detailed overview of the SKIP process, see :ref:`skip0`.

A list of all existing SKIPs is available :ref:`here <skip_list>`.

Copyright
=========

This document is based on the `scikit-learn governance document
<https://scikit-learn.org/stable/governance.html>`_ and is placed in the public
domain.
