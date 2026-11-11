(coordinates)=

# Specifying coordinates

A coordinate identifies a pixel. A coordinate consists of a sequence of
numbers, one for each axis, specifying the pixel location on each axis of the
image. For example, for a 2D image, the coordinate is a length 2 sequence, giving a number identifying the locations on the first and second image axis.

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

Cartesian indexing is common to the point of being universal in description of
image processing algorithms in papers, books and web pages. It is also the
way that Matplotlib thinks of positions on a plot.

In what follows, we'll set out the scikit-image approach, of using Numpy /
matrix indexing for images, and contrast with Cartesian indexing.

After that, we discuss the various ways that scikit-image describes and names
axes in images, and justify our choices.

## Numpy / matrix indexing

Here we recap and extend the exposition in {ref}`numpy_images`. Let us start
with an array that defines an image:

```{eval-rst}
.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> img_2d = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
   ...                    [ 0,  0,  0,  9, 99, 99, 94,  0],
   ...                    [ 0,  0,  0, 25, 99, 99, 79,  0],
   ...                    [ 0,  0,  0,  0,  0,  0,  0,  0],
   ...                    [ 0,  0,  0, 56, 99, 99, 49,  0],
   ...                    [ 0,  0,  0, 73, 99, 99, 31,  0],
   ...                    [ 0,  0,  0, 91, 99, 99, 13,  0],
   ...                    [ 0,  0,  9, 99, 99, 94,  0,  0],
   ...                    [ 0,  0, 27, 99, 99, 77,  0,  0],
   ...                    [ 0,  0, 45, 99, 99, 59,  0,  0],
   ...                    [ 0,  0, 63, 99, 99, 42,  0,  0],
   ...                    [ 0,  0, 80, 99, 99, 24,  0,  0],
   ...                    [ 0,  1, 96, 99, 99,  6,  0,  0],
   ...                    [ 0, 16, 99, 99, 88,  0,  0,  0],
   ...                    [ 0,  0,  0,  0,  0,  0,  0,  0]])
   >>> img_2d.shape
   (15, 8)
```

:::{note} Terminology

To specify the number of rows and columns in an image, we use `I` for
the number of rows and `J` for the number of columns. So we can specify
the shape of a 2D grayscale image in general, as shape `(I, J)`. Here
`I` is 15 and `J` is 8.
:::

You can probably see from the values in the array above that this
two-dimensional array represents a grayscale image of an italic lower-case
"i". Here is the array displayed as an image:

```{eval-rst}
.. plot::
   :context:
   :include-source:
   :caption: ``img_2d`` displayed with the first axis running top to bottom.

   >>> _ = plt.imshow(img_2d, cmap="gray")
```

Numpy indexing into these image arrays is as for any other Numpy array. To
get the pixel in the second (position 1) row, and fourth (position 3) column:

```python
>>> img_2d[1, 3]
np.int64(9)
```

:::{note} Terminology

To generalize, to get a pixel value at row position `i` and column
position `j` in an image / array `image`: `image[i, j]`.
:::

:::{admonition} 0-based indexing and matrix indexing

Because this is Numpy array indexing, and Numpy indexing is 0-based (not
1-based), the top-left pixel value is `img_2d[0, 0]`. This is a
convention choice by Python, and Numpy. In this sense, it differs from
standard matrix / linear algebra notation, that typically prefers 1-based
indexing.
:::

Because indexing is nothing other than Numpy indexing, we can use the usual
indexing expressions:

```python
>>> img_2d[2]  # Values in third row of the image.
array([ 0,  0,  0, 25, 99, 99, 79,  0])
>>> img_2d[:, 3]  # Values in fourth column of the image.
array([ 0,  9, 25,  0, 56, 73, 91, 99, 99, 99, 99, 99, 99, 99,  0])
```

and so on.

To summarize, for Numpy indexing:

- The first coordinate (`i`) specifies positions from top to bottom in the
  image (the image row).
- The second coordinate (`j`) specifies positions from left to right in the
  image (the image column).

## Cartesian indexing

Above, we have used a coordinate pair `i, j` to index into an image. For
example, to get the value in the second row and fourth column, `i=1, j=3`
and `img_2d[i, j]`. The first coordinate in the pair specifies the row
position and the second specifies the column position.

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

As for that page, we start by considering a 2D grayscale image, represented by
a Numpy array with two dimensions.

In what follows, we discuss arrays and axes in general, so we begin by giving
the terminology we use for specifying axes, and length of axes.

## Terminology

An upper-case letter gives the length of an axis. The matching lower-case
letter gives an index into that axis. An array of shape `(I, J)` has
indices `i` from 0 to `I - 1`, and `j` from 0 to `J - 1`. We name an
axis by its lower-case letter, so we write "the `i` axis".

In the simple case of a 2D grayscale (no color) image, we represent that image
with a Numpy array of shape `(I, J)`.

## Two coordinate systems

The same pair of numbers names two different pixels, depending on the
coordinate system.

% Draft note. Source: coordinate_review.md "Background" (L23-53).
% State the problem with one coordinate, (5, 10), before splitting into the
% two subsections below.

### The array coordinate system

The array, or `IJ`, coordinate system gives the position along the first
array axis, then the position along the second. It is the system NumPy
indexing uses:

```python
>>> import skimage as ski
>>> img = ski.data.camera()
>>> img[5, 10]
np.uint8(198)
```

The names `i` and `j` carry no meaning about display. They name array
axes only.

% Draft note. Source: coordinate_review.md "The array coordinate system"
% (L71-106). Needs the displayed image, with the first axis running top to
% bottom and the second left to right. Use `.. plot::` rather than
% `plt.imshow`, because this page is built by Sphinx, not executed as a
% notebook.

### The imaging coordinate system

The imaging, or `XY`, coordinate system reverses the order. The first
number gives the position along the _second_ array axis, and the second
number the position along the _first_:

```python
>>> img[10, 5]
np.uint8(201)
```

`x` runs left to right in the displayed image. `y` runs top to bottom.
Both names refer to the display, not to the array.

% Draft note. Source: coordinate_review.md "The imaging coordinate system"
% (L107-144). Explain why the system is common: it matches graph axes,
% except that y runs downwards for images. Matplotlib axes use (x, y) with
% the origin at the bottom left.

So `(5, 10)` means `img[5, 10]` in array coordinates and `img[10, 5]`
in imaging coordinates. You cannot read a coordinate without knowing its
system.

## Rows and columns

A row runs left to right in the displayed image. A column runs top to
bottom:

```python
>>> row_5 = img[5, :]
>>> col_10 = img[:, 10]
```

For a 2D grayscale image, row-column (`RC`) coordinates and `IJ`
coordinates are the same.

% Draft note. Source: coordinate_review.md "Row and columns" (L145-195).
% Give `skimage.draw.line` as the worked example: it takes (r0, c0, r1, c1)
% and returns (rr, cc) for direct indexing.

## Rows and columns beyond 2D

"Row" and "column" describe a _displayed_ 2D image. Once an array has three
spatial axes, the row axis is no longer the first axis, and which axis holds
the rows depends on how you choose to display the volume:

```python
>>> img_4d = ski.data.cells3d()
>>> img_4d.shape
(60, 2, 256, 256)
```

% Draft note. Source: coordinate_review.md "Rows and columns in 3D"
% (L196-266). Show both readings of the same volume: `img_3d[middle, :, :]`
% makes axes 1 and 2 the rows and columns; `img_3d[:, :, middle]` makes axes
% 0 and 1 the rows and columns. The point is that "row" is a statement about
% display, so it is ambiguous for a volume.

## Semantic and positional labels

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
| :::                       |

Note that the position of `ch` is controlled by the `channel_axis`
argument.

We can supplement the above table for time-varying data:

:::{table} _Addendum to dimension names and orders in scikit-image_

| Image type     | coordinates            |
| -------------- | ---------------------- |
| 2D color video | (t, row, col, ch)      |
| 3D color video | (t, pln, row, col, ch) |
| :::            |

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
