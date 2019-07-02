.. _values:

Our mission
-----------

scikit-image aims to be the essential toolkit for scientific image analysis in
Python. We accomplish this by:

- being **easy to use**. We are careful in taking on new dependencies, and
  sometimes cull existing ones, or make them optional.
- providing a **consistent API**. Conceptually identical arguments have the
  same name and position in a function signature.
- **ensuring accuracy**. Test coverage is close to 100% and code is reviewed by
  at least two core developers before being included in the library.
- **caring about users’ data**. We have a functional [2]_ API and don't modify
  input arrays unless explicitly directed to do so.

Our values
----------

- We are inclusive. We continue to welcome and mentor newcomers who are
  making their first contribution.
- We are community-driven. Decisions about the API and features are driven by
  our users' requirements, not by the whims of the core team. (See
  :ref:`governance`.)
- We serve scientific applications primarily, over “consumer” image editing in
  the vein of Photoshop or GIMP. This often means prioritizing n-dimensional
  data support, and rejecting implementations of “flashy” filters that have
  little scientific value.
- We value elegant implementations over getting every last ounce of
  performance. Readable code that is easy to understand, for newcomers and
  maintainers alike, makes it easier to contribute new code as well as prevent
  bugs. This means that we will prefer a 20% slowdown if it reduces lines of
  code two-fold, for example.
- We require excellent documentation. All functions should have docstrings,
  preferably with doctest examples, as well as gallery examples that showcase
  how that function is used in a scientific application. Good documentation is
  hard, so this requirement can stall many contributions. Core developers take
  an active role in finishing final documentation examples.
- We don't do magic. We use NumPy arrays instead of fancy façade objects^, and
  we prefer to educate users rather than make decisions on their behalf. This
  does not preclude sensible defaults, but it does preclude *value-dependent*
  behavior that is not controllable by the user. [3]_

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

- Original blog post [1]_
- The GitHub issue [5]_
- The image.sc forum post [6]_
- The SKIP GitHub pull request [7]_

Footnotes
---------

^ The use of NumPy arrays was the most supported of the statement's components,
together with the points about inclusivity, mentorship, and documentation.  We
had +1s from Mark Harfouche, Royi Avital, and Greg Lee, among others.

References
----------

.. [1] https://ilovesymposia.com/2018/07/13/the-road-to-scikit-image-1-0/
.. [2] https://en.wikipedia.org/wiki/Functional_programming
.. [3] https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/4
.. [4] https://www.python.org/dev/peps/pep-0020/
.. [5] https://github.com/scikit-image/scikit-image/issues/3263
.. [6] https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099
.. [7] https://github.com/scikit-image/scikit-image/pull/3585
