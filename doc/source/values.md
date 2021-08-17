# Our mission

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
- **caring for users’ data**. We have a [functional API][functional] and don't modify
  input arrays unless explicitly directed to do so.
- promoting **education in image processing**, with extensive pedagogical
  documentation.

(sec:values)=
Our values
----------

- We are inclusive. We continue to welcome and mentor newcomers who are
  making their first contribution.
- We are community-driven. Decisions about the API and features are driven by
  our users' requirements, not by the whims of the core team. (See
  {ref}`governance`.)
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
  [docstrings][numpydoc], preferably with examples, as well as gallery
  examples that showcase how that function is used in a scientific application.
  Core developers take an active role in finishing documentation examples.
- We don't do magic. We use NumPy arrays instead of fancy façade objects
  [^np], and we prefer to educate users rather than make decisions on their
  behalf.  This does not preclude [sensible defaults][defaults].

This document
-------------

Much in the same way that the [Zen of Python][zen] and PEP8 guide style and
implementation details in most Python code, this guide is meant to guide
decisions about the future of scikit-image, be it in terms of code style,
whether to accept new functionality, or whether to take on new dependencies,
among other things.

References
----------

To find out more about the history of this document, please read the following:

- [Original blog post][blog]
- [The GitHub issue][issue]
- [The image.sc forum post][forum]
- [The SKIP GitHub pull request][skip_pr]

% Links
% -----

[functional]: https://en.wikipedia.org/wiki/Functional_programming
[blog]: https://ilovesymposia.com/2018/07/13/the-road-to-scikit-image-1-0/
[numpydoc]: https://docs.scipy.org/doc/numpy/docs/howto_document.html
[defaults]: https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099/4
[zen]: https://www.python.org/dev/peps/pep-0020/
[issue]: https://github.com/scikit-image/scikit-image/issues/3263
[forum]: https://forum.image.sc/t/request-for-comment-road-to-scikit-image-1-0/20099
[skip_pr]: https://github.com/scikit-image/scikit-image/pull/3585
[cc0]: https://creativecommons.org/publicdomain/zero/1.0/
[ccby]: https://dancohen.org/2013/11/26/cc0-by/

Copyright
---------

This document is dedicated to the public domain with the Creative Commons CC0
[license][cc0]. Attribution to this source is encouraged where appropriate, as per
[CC0+BY][ccby].

[^np]: The use of NumPy arrays was the most supported of the statement's
       components, together with the points about inclusivity, mentorship, and
       documentation. We had +1s from Mark Harfouche, Royi Avital, and Greg Lee,
       among others.
