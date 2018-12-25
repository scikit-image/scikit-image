####################################
SKIP: scikit-image mission statement
####################################

Introduction
============

I propose that the scikit-image project adopts the document below as its
mission statement. This statement would feature prominently in the scikit-image
home page and readme, as well as the contributor and core developer guides.
Decisions about the API and the future of the library would be referenced
against this document.

Background
==========

In July, I published a blog post that broadly outlined what I would want from a
roadmap for scikit-image [1]_, but requested comments from the community before
it would be finalized. I consider that we have collected comments for long
enough and can move forward with formal adoption. Most comments were positive,
so below I'll just summarize the "negative" comments under "rejected ideas".

What problem does this proposal solve?
======================================

Over the past few years, scikit-image has been slightly "adrift", with new and
old contributors coming in, adding what small bits they need at the time, and
disappearing again for a while. This is *fine* and we want to encourage more of
it, but it also lacks direction. Additionally, without a concerted roadmap to
concentrate effort, many of these contributions just fall by the wayside, as it
is difficult for new contributors to live up to our stringent (and largely
unwritten) standards of code.

The statement
=============

Our mission
-----------

scikit-image aims to be the essential toolkit for scientific image analysis in
Python.  We will accomplish this by:

- being **easy to use**. We are careful in taking on new dependencies, and
  sometimes cull existing ones, or make them optional.
- providing a **consistent API**. Conceptually identical arguments have the
  same name and position in a function signature.
- **ensuring accuracy**. Test coverage is close to 100% and code is reviewed by
  at least two core developers before being included in the library.
- **caring about data**. We have a functional [2]_ API and don't modify input
  arrays unless explicitly directed to do so.

Our values
----------

- We value elegant implementations that are easy to understand for newcomers
  over getting every last ounce of performance. This means that we will prefer
  a 20% slowdown if it reduces lines of code two-fold.
- We serve scientific applications primarily, over "consumer" image editing in
  the vein of Photoshop or GIMP. This often means prioritizing n-dimensional
  data support, and rejecting implementations of "flashy" filters that have
  little scientific value.
- We are inclusive. We will continue to welcome and mentor newcomers who are
  making their first contribution.
- We require excellent documentation. All functions should have docstrings,
  preferably with doctest examples, as well as gallery examples that showcase
  how that function is used in a scientific application. Good documentation is
  hard, so this requirement can stall many contributions. In the future, core
  developers will take a more active role in finishing the final documentation
  example.
- We don't do magic. We use NumPy arrays instead of fancy façade objects^, and
  we prefer to educate users rather than make decisions on their behalf. This
  does not preclude sensible defaults, but it does preclude *value-dependent*
  behavior that is not controllable by the user. [3]_
- We are community-driven. Decisions about the API and features are driven by
  our users' requirements, not by the whims of the core team.

This document
-------------

Much in the same way that the Zen of Python [4]_ and PEP8 guide style and
implementation details in most Python code, this guide is meant to guide any
decisions about the future of scikit-image, be it in terms of code style,
whether to accept new functionality, or whether to take on new dependencies,
among other things.

References
----------

To find out more about the history of this document, please read the following:

- Original blog post
- GitHub post
- image.sc forum post
- This GitHub pull request

Footnotes
---------

^ The use of NumPy arrays was the most supported of the statement's components,
together with the points about inclusivity, mentorship, and documentation.  We
had +1s from Mark Harfouche, Royi Avital, and Greg Lee, among others.

Rejected ideas
==============

Handling metadata
-----------------

In my original post, I suggested that scikit-image should have some form of
metadata handling before 1.0. Among others, Mark Harfouche, Curtis Rueden, and
Dan Allan all advised that (a) maybe scikit-image doesn't *need* to handle
metadata, and can instead focus on being a robust lower-level library that
another like XArray can use to include metadata handling, and (b) anyway,
metadata support can be added later without breaking the 1.0 API. I think these
are very good points and furthermore metadata handling is super hard and I
don't mind keeping this off our plate for the moment.

Magical thinking
----------------

Philipp Hanslovsky suggested [5]_ that, regarding "doing magic", it is
advisable in some contexts, and a good solution is to provide a magic layer
built on top of the non-magical one. I agree with this assessment, but, until
1.0, I'm happy for scikit-image to remain the non-magic layer.

References
==========

..[1]: https://ilovesymposia.com/2018/07/13/the-road-to-scikit-image-1-0/
..[2]: https://en.wikipedia.org/wiki/Functional_programming
..[3]: https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/4
..[4]: https://www.python.org/dev/peps/pep-0020/
..[5]: https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/3
