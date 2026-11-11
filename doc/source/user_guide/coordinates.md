---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(coordinates)=

# Specifying image coordinates

## Summary

The following applies to a grayscale images — that is, the image array has no color channel axis.

If we have a 2D grayscale image `img` then we refer to the pixel at row
position `i` and column position `j` as being at coordinate `(i, j)`. We say
that the image, in general, has shape `(I, J)`.

Scikit-image represents a grayscale 2D image as a 2D array. Thus the pixel
value at `i, j` is the result of Numpy indexing into the array with `img[i,
j]`.

For a 3D grayscale image `img3d`, then we name the coordinates of three axes
`i, j, k`, and the image array has shape `(I, J, K)`. It will depend on the
particular image whether the first axis is best thought of as "rows" or
"columns" or "planes".

A multichannel (color) image has an extra axis containing the color channel
information. Such an axis will typically be last, but routines that consider
color channels have a `channel_axis` keyword argument to specify the axis in
the array that holds the color information. Thus, in the typical case, a 2D
image with color channels will be shape `(I, J, C)`, where `C` is the number
of channels, and `C` is typically (e.g. for RGB) equal to 3. However, if
your image array has `channel_axis=0`, then the image array would be shape
`(C, I, J)`.

## Background

A coordinate identifies a pixel. A coordinate consists of a sequence of
numbers, one for each image axis, specifying the pixel location on the matching
image axis. For example, for a 2D image, the coordinate is a length 2 sequence,
giving a number identifying the locations on the first and second image axes.

This page sets out the coordinate systems that we use in scikit-image, the
names we give to image axes, and the rules the library follows when a function
takes or returns coordinates.

The page assumes you have read {ref}`numpy_images`. That page, in turn,
explains the fundamental principle in scikit-image — that we use Numpy
arrays to represent images. This choice is the basis for the way we think of
image coordinates; coordinates are indices into arrays. When we index into an
image, we use array coordinates. That is, to get the pixel value in row `i`
and column `j`, the coordinate is `i, j`, and the value at that coordinate
(in code) is `image[i, j]`.

For those of you who are used to Numpy arrays, or who think naturally in terms
of matrices, this will be an obvious choice. But, there is another common
choice in image processing, and that is Cartesian (x, y) indexing, where "x"
refers to columns in the image, and "y" refers to rows. Therefore to get the
pixel value in column `x`, and row `y`, in Cartesian coordinates, we would
ask for `image{x, y}`.

(curlies-for-cartesian)=

:::{note} Curly brackets for Cartesian indexing

Notice we used `{x, y}` rather than `[x, y]` above. We'll continue to
use curly brackets for Cartesian indexing to remind ourselves that this is
notation, not code.
:::

Cartesian indexing is very common in descriptions of image processing
algorithms in papers, books and web pages. It is also the way that Matplotlib
thinks of positions on a plot.

In what follows, we'll set out the scikit-image approach, of using Numpy /
matrix indexing for images, and contrast with Cartesian indexing.

After that, we discuss the various ways that scikit-image describes and names
axes in images, and justify our choices.

## Numpy / matrix indexing

Here we recap and extend the exposition in {ref}`numpy_images`. Let us start
with an array that defines an image:

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

img_2d = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  9, 99, 99, 94,  0],
                   [ 0,  0,  0, 25, 99, 99, 79,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0, 56, 99, 99, 49,  0],
                   [ 0,  0,  0, 73, 99, 99, 31,  0],
                   [ 0,  0,  0, 91, 99, 99, 13,  0],
                   [ 0,  0,  9, 99, 99, 94,  0,  0],
                   [ 0,  0, 27, 99, 99, 77,  0,  0],
                   [ 0,  0, 45, 99, 99, 59,  0,  0],
                   [ 0,  0, 63, 99, 99, 42,  0,  0],
                   [ 0,  0, 80, 99, 99, 24,  0,  0],
                   [ 0,  1, 96, 99, 99,  6,  0,  0],
                   [ 0, 16, 99, 99, 88,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0]])
img_2d.shape
```

:::{note} Terminology

To specify the number of rows and columns in an image, we use `I` for
the number of rows and `J` for the number of columns. So we can specify
the shape of a 2D grayscale image in general, as shape `(I, J)`. Here
`I` is 15 and `J` is 8.
:::

You can probably see by squinting at the values in the array above that this
two-dimensional array represents a grayscale image of an italic lower-case "i".
Here is the array displayed as an image:

```{code-cell} python
plt.imshow(img_2d, cmap="gray");
```

Numpy indexing into these image arrays is as for any other Numpy array. To
get the pixel in the second (position 1) row, and fourth (position 3) column:

```{code-cell} python
img_2d[1, 3]
```

:::{note} Terminology

To generalize, to get a pixel value at row position `i` and column
position `j` in an image / array `image`: `image[i, j]`.
:::

:::{admonition} 0-based indexing and matrix indexing

Because this is Numpy array indexing, and Numpy indexing is 0-based (not
1-based), the top-left pixel value is `img_2d[0, 0]`. This is a convention
choice by Python, and Numpy. In this sense, it differs from standard matrix
/ linear algebra notation, that typically prefers 1-based indexing.
:::

Because indexing is nothing other than Numpy indexing, we can use the usual
indexing expressions:

```{code-cell} python
img_2d[2]  # Values in third row of the image.
```

```{code-cell} python
img_2d[:, 3]  # Values in fourth column of the image.
```

and so on.

To summarize, for Numpy indexing:

- The first coordinate (`i`) specifies positions from top to bottom in the
  image (the image row).
- The second coordinate (`j`) specifies positions from left to right in the
  image (the image column).

## Matplotlib / Cartesian indexing

Above, we have used a coordinate pair `i, j` to index into an image. For
example, to get the value in the second row and fourth column, `i=1, j=3` and
`img_2d[i, j]`. The first coordinate in the pair specifies the row position and
the second specifies the column position.

Cartesian indexing is an alternative convention for defining what the first
and second coordinate values mean, in an indexing expression. In Cartesian
indexing, a coordinate pair `x, y` specifies the **column** at position
`x` and the **row** at position `y`.

Cartesian indexing, like Numpy / matrix indexing, needs to specify whether it
is 0-based, or 1-based --- but we'll assume 0-base here to reduce confusion.

Therefore, in Cartesian indexing, to get the value at row position 1 and
column position 3, we would use `img_2d{3, 1}`. To summarize, in Cartesian
indexing:

- The first coordinate (`x`) specifies positions from left to right in the
  image (the image column).
- The second coordinate (`y`) specifies positions from top to bottom in the
  image (the image row).

Cartesian indexing matches the indexing we use for on graph axes, where we
think of the x-axis (the first axis) as being left-to-right. Then consider the
way we typically display images in e.g. Matplotlib. Consider the display of the image of the italic "i" above, repeated below. Notice the image axes for this Matplotlib default image display, that have their origin at 0, 0 at the top-left of the image.

We've also placed a red dot at position `{2, 5}` _in Matplotlib (image axis)
coordinates_.

```{code-cell} python
plt.imshow(img_2d, cmap="gray");
plt.plot(5, 2, color='red', marker='o')
```

Notice that Matplotlib considers the first axis to be left-to-right (hence
column 5 on this plot), and the second axis to be top-to-bottom (hence row 2 on
this plot). This is Cartesian indexing. Because it is natural to think in
this convention when discussing displayed images, it is a very common
convention in papers and textbooks describing image algorithms.

## Coordinate conventions in scikit-image

Versions of sckit-image before version 2 used both Numpy and Matplotlib
(Cartesian) conventions, because Cartesian conventions are so common in
descriptions of algorithms.

Version 2 onwards standardize to using Numpy conventions.

## Rows and columns beyond 2D

"Row" and "column" describe a _displayed_ _2D image_. Once an image has three
axes, or more, the relationship between axes and rows, columns becomes more
ambiguous.

For example, consider the following array that contains a 3D image of cells.

See the docstring for `ski.data.cell3d` for more details on this array.

```{code-cell} python
import skimage as ski

# The full data has shape (n_planes, n_channels, n_rows, n_columns).
img_4d = ski.data.cells3d()
# Slice the 4D array to get a 3D array (and a 3D image).
# Channel 1 contains imaging of nuclei.
img_3d = img_4d[:, 1]
img_3d.shape
```

In fact, in this case, the first axis contains the _planes_ of the image
— that is, the microscope took planar (2D) images, and the array we have is
the result of taking these 2D images, and stacking them over the first
dimension. For example, this was the first planar 2D image from the whole sequence:

```{code-cell} python
first_plane = img_3d[0, :, :]

plt.imshow(first_plane, cmap='gray')
```

We bring this up, because it may now be obvious that, for a 3D image, it is no
longer true that the first axis is always "rows" in the image, and the second
"columns". And, in fact, in this case, the first axis could be thought of as
containing "planes", the second "rows" and the third "columns".

For this reason, scikit-image, in docstrings and documentation, refers to
coordinates in grayscale 3D images as `(i, j, k)` — meaning `i` is the
coordinate on the first axis (whatever that represents), and so on. Similarly
we write the shape of a 3D (grayscale) image, in general, as `(I, J, K)`.

Put another way, the term "row" is a statement about our intended
two-dimensional display of the image, so it is ambiguous for a volume. So we
avoid the terms "row" and "column" for 3D (volume) image.

## Color channels, semantic and positional labels

You will notice that, so far, we have avoided using labels for axes that
attempt to give some meaning to the axes. Labels that contain meaning or
interpretation of the axis are — semantic labels. For example, "row" is
a semantic label, because it tells us that the labeled axis will be displayed
as rows in a 2D display. But `i` and `j` and `k` are not semantic; they just mean the first, second and third axes in a 3D (grayscale) image.

However, color images have an extra axis that contains the color information. For example, consider the `cat` image:

pcc

`row`, `column` and `plane` are _semantic_ labels. Each says what an
axis means for display. `i`, `j` and `k` are _positional_ labels. Each
names an axis by its position in the array, and says nothing about meaning.

% Draft note. Source: coordinate_review.md "m and n are ambiguous labels"
% (L267-291) and "Semantic and not-semantic labels" (L292-300).
% State plainly that `M` and `N` are ambiguous in current docstrings:
% they mean (row, col) in some places and (i, j) in others. Prefer `i` and
% `j`.

## Convention fixes the mapping

A semantic label only identifies an axis if the axis order is fixed by
convention. `scikit-image` puts the plane axis first, then rows, then
columns. Other fields order them differently.

% Draft note. Source: coordinate_review.md "Convention and semantic labels"
% (L301-320) and "Skimage and row, column" (L321-341).
% The brain-imaging counter-example is useful: there the plane axis is last.
% Colour channels are the convention most readers already know.

(numpy-images-coordinate-conventions)=

### The conventions

% Draft note. Moved here from numpy_images.rst, together with the label
% above, which `data_types.rst` and `development/guidelines.md` both
% reference. Rename the label only if those two are updated with it.

Two-dimensional (2D) grayscale images are indexed by rows and columns
(abbreviated to either `(row, col)` or `(r, c)`), with the lowest element
`(0, 0)` at the top-left corner. In various parts of the library, you will
also see `rr` and `cc` refer to lists of row and column coordinates. We
distinguish this convention from `(x, y)`, which commonly denote standard
Cartesian coordinates, where `x` is the horizontal coordinate, `y` - the
vertical one, and the origin is at the bottom left (Matplotlib axes, for
example, use this convention).

In the case of multichannel images, any dimension (array axis) can be used for
color channels, and is denoted by `channel` or `ch`.

For volumetric (3D) images, such as videos, magnetic resonance imaging (MRI)
scans, confocal microscopy, etc., we refer to the leading dimension as
`plane`, abbreviated as `pln` or `p`.

These conventions are summarized below:

:::{table} _Dimension name and order conventions in scikit-image_

| Image type                | Coordinates         |
| ------------------------- | ------------------- |
| 2D grayscale              | (row, col)          |
| 2D multichannel (eg. RGB) | (row, col, ch)      |
| 3D grayscale              | (pln, row, col)     |
| 3D multichannel           | (pln, row, col, ch) |

:::

Note that the position of `ch` is controlled by the `channel_axis`
argument.

We can supplement the above table for time-varying data:

:::{table} _Addendum to dimension names and orders in scikit-image_

| Image type     | coordinates            |
| -------------- | ---------------------- |
| 2D color video | (t, row, col, ch)      |
| 3D color video | (t, pln, row, col, ch) |

:::

## Color channels and `channel_axis`

No axis is a color axis by default. Functions that accept multichannel data
take a `channel_axis` argument that names the axis holding the channels.

% Draft note. Source: coordinate_review.md "Color and `channel_axis`"
% (L344-371). That section is a porting checklist; only the user-facing rule
% belongs here. Give the default: `channel_axis=-1` for functions that
% require multichannel data, `channel_axis=None` otherwise.

## Coordinates as arguments

A function takes coordinates in one of two shapes. Some take one argument per
axis, such as `line(r0, c0, r1, c1)`. Others take a single sequence holding
one value per axis, such as `disk(center, radius)`.

% Draft note. Source: coordinate_review.md "Coordinate arguments as
% sequences" (L372-521). Say which pattern applies where, and in what order
% the sequence holds its values. Do not reproduce the porting survey.

## Angles

Angles are in radians.

% Draft note. Source: coordinate_review.md "Radians and degrees" (L522-539).
% List the functions that take degrees instead: `transform.rotate`, and
% `theta` in `radon`, `iradon` and `iradon_sart`. Check this list
% against the code before publishing; the source is a porting document and
% proposes changing some of them.

:::{seealso}

- {ref}`numpy_images` — Images as NumPy arrays, and the cost of axis order.
- {ref}`geometrical_transform` — Transformations that map one set of coordinates to another.

:::
