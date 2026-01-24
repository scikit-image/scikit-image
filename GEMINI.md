I am working on the code in this repository, a branch of https://github.com/scikit-image/scikit-image.

The code uses two conventions for specifying and returning coordinates in images.

As background, consider some image, `img`, loaded as a Numpy array, shape (M,
N).  The coordinate conventions tell us the meaning of the coordinate `(p, q)`
into this image, `img`.

One convention I will call the *Numpy convention*, also called the "ij"
convention. This simply uses array indexing, so the first coordinate value,
`p`, gives the position in terms of the first axis in `img`, and `q` gives the
position in terms of the second axis.  Thus the value of `img` at `(p, q)` is
simply `img[p, q]` in Python / Numpy code.  Notice that the term "row" in this
convention means a slice through the *second axis*.  Thus row `r` is given by
(Numpy) `img[r, :]`.  Similarly, "column" in this convention means a slice
through the *first axis*.  Column `c` in this convention would be (Numpy)
`img[:, c]`.

A second convention I will call the imaging convention.   Here `p` gives the
position on the *second* Numpy axis, and `q` gives the position on the *first*
Numpy axis.  The value of `img` at `(p, q)` for this convention is `img[q, p]`
in Python / Numpy.  I will also refer to this convention as the "xy"
convention. simply `img[p, q]` in Python / Numpy code. In this convention, ow
`r` is given by (Numpy) `img[:, r]`.  Column `c` in this convention would
be (Numpy) `img[c, :]`.

Just to emphasize again, "row" and "column" have opposite meanings in "ij" and "xy" convention.

Please review the open issue at
<https://github.com/scikit-image/scikit-image/issues/7728>, and any documents
linked from that document, for more detail.
