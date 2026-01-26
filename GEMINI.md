I am working on the code in this repository, a branch of https://github.com/scikit-image/scikit-image.

The code uses two conventions for specifying and returning coordinates in images.

As background, consider some image, `img`, loaded as a Numpy array, shape (M,
N). The coordinate conventions tell us the meaning of the coordinate `(p, q)`
into this image, `img`.

One convention I will call the _Numpy convention_, also called the "ij"
convention. This simply uses array indexing, so the first coordinate value,
`p`, gives the position in terms of the first axis in `img`, and `q` gives the
position in terms of the second axis. Thus the value of `img` at `(p, q)` is
simply `img[p, q]` in Python / Numpy code. Notice that the term "row" in this
convention means a slice through the _second axis_. Thus row `r` is given by
(Numpy) `img[r, :]`. Similarly, "column" in this convention means a slice
through the _first axis_. Column `c` in this convention would be (Numpy)
`img[:, c]`.

A second convention I will call the imaging convention. Here `p` gives the
position on the _second_ Numpy axis, and `q` gives the position on the _first_
Numpy axis. The value of `img` at `(p, q)` for this convention is `img[q, p]`
in Python / Numpy. I will also refer to this convention as the "xy"
convention. simply `img[p, q]` in Python / Numpy code. In this convention, ow
`r` is given by (Numpy) `img[:, r]`. Column `c` in this convention would
be (Numpy) `img[c, :]`.

Just to emphasize again, "row" and "column" have opposite meanings in "ij" and
"xy" convention.

This also means that "rc" as a coordinate convention is ambiguous, although, in
this codebase, it often means ij convention.

Note too that the current code was not adapted to this convention. Thus `x`
and `y` in function / method names, or in docstrings do not _necessarily_ imply
the code is using "xy" convention. Please investigate the code implementation,
and uses of the code elsewhere in the codebase as evidence, and provide this
evidence in your summaries. I'm particularly interested in snippets of code
that call particular functions or methods, where the snippets demonstrate that
the convention is ij or xy.

For more background, please review the open issue at
<https://github.com/scikit-image/scikit-image/issues/7728>, and any documents
linked from that document.

Please provide a summary of your findings, for each function or method that you
identify, giving evidence, with links to the relevant code, starting from the
base repository on Github, at https://github.com/scikit-image/scikit-image
