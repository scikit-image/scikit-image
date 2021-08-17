=======================================
SKIP 2 — scikit-image mission statement
=======================================

:Author: Juan Nunez-Iglesias <juan.nunez-iglesias@monash.edu>
:Author: Stéfan van der Walt <stefanv@berkeley.edu>
:Author: Josh Warner
:Author: François Boulogne
:Author: Emmanuelle Gouillart
:Author: Mark Harfouche
:Author: Lars Grüter
:Author: Egor Panfilov
:Author: Gregory Lee
:Status: Active
:Type: Process
:Created: 2018-12-08
:Resolved:
:Resolution:
:skimage-Version: 0.16

Abstract
--------

scikit-image should adopt the document below as its mission statement. This
statement will feature prominently in the scikit-image home page and readme,
as well as the contributor and core developer guides. Decisions about the API
and the future of the library would be referenced against this document. (See
:ref:`governance`.)

In July 2018, I (Juan) published a blog post that broadly outlined what I would
want from a roadmap for scikit-image [1]_, but requested comments from the
community before it would be finalized. I consider that we have collected
comments for long enough and can move forward with formal adoption. Most
comments were positive, so below I'll just summarize the “negative” comments
under “rejected ideas”.

Detailed description
--------------------

(Or: What problem does this proposal solve?)

Over the past few years, scikit-image has been slightly “adrift”, with new and
old contributors coming in, adding what small bits they need at the time, and
disappearing again for a while. This is *fine* and we want to encourage more of
it, but it also lacks direction. Additionally, without a concerted roadmap to
concentrate effort, many of these contributions just fall by the wayside, as it
is difficult for new contributors to live up to our stringent (and largely
unwritten) standards of code.

Implementation
--------------

Our mission
***********

scikit-image aims to be the reference library for scientific image analysis in
Python. We accomplish this by:

- being **easy to use and install**. We are careful in taking on new
  dependencies, and sometimes cull existing ones, or make them optional. All
  functions in our API have thorough docstrings clarifying expected inputs and
  outputs.
- providing a **consistent API**. Conceptually identical arguments have the
  same name and position in a function signature.
- **ensuring correctness**. Test coverage is close to 100% and code is reviewed by
  at least two core developers before being included in the library.
- **caring for users’ data**. We have a functional [2]_ API and don't modify
  input arrays unless explicitly directed to do so.
- promoting **education in image processing**, with extensive pedagogical
  documentation.

Our values
**********

- We are inclusive. We continue to welcome and mentor newcomers who are
  making their first contribution.
- We are community-driven. Decisions about the API and features are driven by
  our users' requirements, not by the whims of the core team. (See
  :ref:`governance`.)
- We serve scientific applications primarily, over “consumer” image editing in
  the vein of Photoshop or GIMP. This often means prioritizing n-dimensional
  data support, and rejecting implementations of “flashy” filters that have
  little scientific value.
- We value simple, readable implementations over getting every last ounce of
  performance. Readable code that is easy to understand, for newcomers and
  maintainers alike, makes it easier to contribute new code as well as prevent
  bugs. This means that we will prefer a 20% slowdown if it reduces lines of
  code two-fold, for example.
- We value education and documentation. All functions should have NumPy-style
  docstrings [3]_, preferably with examples, as well as gallery
  examples that showcase how that function is used in a scientific application.
  Core developers take an active role in finishing documentation examples.
- We don't do magic. We use NumPy arrays instead of fancy façade objects
  [#np]_, and we prefer to educate users rather than make decisions on their
  behalf.  This does not preclude sensible defaults. [4]_

This document
*************

Much in the same way that the Zen of Python [5]_ and PEP8 guide style and
implementation details in most Python code, this guide is meant to guide
decisions about the future of scikit-image, be it in terms of code style,
whether to accept new functionality, or whether to take on new dependencies,
among other things.

References
**********

To find out more about the history of this document, please read the following:

- Original blog post [1]_
- The GitHub issue [6]_
- The image.sc forum post [7]_
- The SKIP GitHub pull request [8]_

.. rubric:: Footnotes

.. [#np] The use of NumPy arrays was the most supported of the statement's
   components, together with the points about inclusivity, mentorship, and
   documentation. We had +1s from Mark Harfouche, Royi Avital, and Greg Lee,
   among others.

Backward compatibility
----------------------

This SKIP formalizes what had been the unwritten culture of scikit-image, so it
does not raise any backward compatibility concerns.

Alternatives
------------

Two topics in the original discussion were ultimately rejected, detailed below:

Handling metadata
*****************

In my original post, I suggested that scikit-image should have some form of
metadata handling before 1.0. Among others, Mark Harfouche, Curtis Rueden, and
Dan Allan all advised that (a) maybe scikit-image doesn't *need* to handle
metadata, and can instead focus on being a robust lower-level library that
another like XArray can use to include metadata handling, and (b) anyway,
metadata support can be added later without breaking the 1.0 API. I think these
are very good points and furthermore metadata handling is super hard and I
don't mind keeping this off our plate for the moment.

Magical thinking
****************

Philipp Hanslovsky suggested [9]_ that, regarding "doing magic", it is
advisable in some contexts, and a good solution is to provide a magic layer
built on top of the non-magical one. I agree with this assessment, but, until
1.0, scikit-image should remain the non-magic layer.

Discussion
----------

See References below.

References
----------

.. [1] https://ilovesymposia.com/2018/07/13/the-road-to-scikit-image-1-0/
.. [2] https://en.wikipedia.org/wiki/Functional_programming
.. [3] https://docs.scipy.org/doc/numpy/docs/howto_document.html
.. [4] https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/4
.. [5] https://www.python.org/dev/peps/pep-0020/
.. [6] https://github.com/scikit-image/scikit-image/issues/3263
.. [7] https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099
.. [8] https://github.com/scikit-image/scikit-image/pull/3585
.. [9] https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/3
.. [10] CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
   https://creativecommons.org/publicdomain/zero/1.0/
.. [11] https://dancohen.org/2013/11/26/cc0-by/

Copyright
---------

This document is dedicated to the public domain with the Creative Commons CC0
license [10]_. Attribution to this source is encouraged where appropriate, as per
CC0+BY [11]_.
