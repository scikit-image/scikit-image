﻿# Acknowledgements

scikit-image is a joint effort, created by a large community of contributors.
For a full list of contributors, please visit
[our GitHub repo](https://github.com/scikit-image/scikit-image/graphs/contributors)
or use `git` in the source repository as follows:

```
git shortlog --summary --numbered

```

Previously, we asked authors to add their names to this file whenever
they made a contribution. Because these additions were not made
consistently, we now refer to the git commit log as the ultimate
record of code contribution.

Please note that, on a project as large as this, there are _many_
different ways to contribute, of which code is only one. Other
contributions include community & project management, code review,
answering questions on forums, and web design. We are grateful for
each and every contributor, regardless of their role.

## Historical credits list

- Stefan van der Walt
  Project coordination

- Nicolas Pinto
  Colour spaces and filters, and image resizing.
  Shape views: `util.shape.view_as_windows` and `util.shape.view_as_blocks`
  Montage helpers: `util.montage`.

- Damian Eads
  Morphological operators

- Mahipal Raythattha
  Documentation infrastructure

- S. Chris Colbert
  OpenCV wrappers, Scivi, Qt and Gtk gui bits, fast Hough transform,
  and much more.

- Holger Rapp
  OpenCV functions and better OSX library loader

- Ralf Gommers
  Image IO, color spaces, plots in documentation, cleaner API docs

- Helge Reikeras
  Logic around API docs generation

- Tony Yu
  Reading of paletted images; build, bug and doc fixes.
  Code to generate skimage logo.
  Otsu thresholding, histogram equalisation, template matching, and more.

- Zachary Pincus
  Tracing of low cost paths, FreeImage I/O plugin, iso-contours,
  and more.

- Almar Klein
  Binary heap class and other improvements for graph algorithms
  Lewiner variant of marching cubes algorithm

- Lee Kamentsky and Thouis Jones of the CellProfiler team, Broad Institute, MIT
  Constant time per pixel median filter, edge detectors, and more.

- Dan Farmer
  Incorporating CellProfiler's Canny edge detector, ctypes loader with Windows
  support.

- Pieter Holtzhausen
  Incorporating CellProfiler's Sobel edge detector, build and bug fixes.
  Radon transform, template matching.

- Emmanuelle Gouillart
  Total variation noise filtering, integration of CellProfiler's
  mathematical morphology tools, random walker segmentation,
  tutorials, and more.

- Maël Primet
  Total variation noise filtering

- Martin Bergholdt
  Fix missing math.h functions in Windows 7 + MSVCC.

- Neil Muller
  Numerous fixes, including those for Python 3 compatibility,
  QT image reading.

- The IPython team
  From whom we borrowed the github+web tools / style.

- Kyle Mandli
  CSV to ReST code for feature comparison table.

- The Scikit Learn team
  From whom we borrowed the example generation tools.

- Andreas Mueller
  Example data set loader. Nosetest compatibility functions.
  Quickshift image segmentation, Felzenszwalbs fast graph based segmentation.

- Yaroslav Halchenko
  For sharing his expert advice on Debian packaging.

- Brian Holt
  Histograms of Oriented Gradients

- David-Warde Farley, Sturla Molden
  Bresenheim line drawing, from snippets on numpy-discussion.

- Christoph Gohlke
  Windows packaging and Python 3 compatibility.

- Neil Yager
  Skeletonization and grey level co-occurrence matrices.

- Nelle Varoquaux
  Renaming of the package to `skimage`.
  Harris corner detector

- W. Randolph Franklin
  Point in polygon test.

- Gaël Varoquaux
  Harris corner detector

- Nicolas Poilvert
  Shape views: `util.shape.view_as_windows` and `util.shape.view_as_blocks`
  Image resizing.

- Johannes Schönberger
  Drawing functions, adaptive thresholding, regionprops, geometric
  transformations, LBPs, polygon approximations, web layout, and more.

- Pavel Campr
  Fixes and tests for Histograms of Oriented Gradients.

- Joshua Warner
  Multichannel random walker segmentation, unified peak finder backend,
  n-dimensional array padding, marching cubes, bug and doc fixes.

- Petter Strandmark
  Perimeter calculation in regionprops.

- Olivier Debeir
  Rank filters (8- and 16-bits) using sliding window.

- Luis Pedro Coelho
  imread plugin

- Steven Silvester, Karel Zuiderveld
  Adaptive Histogram Equalization

- Anders Boesen Lindbo Larsen
  Dense DAISY feature description, circle perimeter drawing.

- François Boulogne
  Drawing: Andres Method for circle perimeter, ellipse perimeter,
  Bezier curve, anti-aliasing.
  Circular and elliptical Hough Transforms
  Thresholding
  Various fixes

- Thouis Jones
  Vectorized operators for arrays of 16-bit ints.

- Xavier Moles Lopez
  Color separation (color deconvolution) for several stainings.

- Jostein Bø Fløystad
  Tomography: radon/iradon improvements and SART implementation
  Phase unwrapping integration

- Matt Terry
  Color difference functions

- Eugene Dvoretsky
  Yen, Ridler-Calvard (ISODATA) threshold implementations.

- Riaan van den Dool
  skimage.io plugin: GDAL

- Fedor Morozov
  Drawing: Wu's anti-aliased circle

- Michael Hansen
  novice submodule

- Munther Gdeisat
  Phase unwrapping implementation

- Miguel Arevallilo Herraez
  Phase unwrapping implementation

- Hussein Abdul-Rahman
  Phase unwrapping implementation

- Gregor Thalhammer
  Phase unwrapping integration

- François Orieux
  Image deconvolution http://research.orieux.fr

- Vighnesh Birodkar
  Blob Detection

- Axel Donath
  Blob Detection

- Adam Feuer
  PIL Image import and export improvements

- Rebecca Murphy
  astronaut in examples

- Geoffrey French
  skimage.filters.rank.windowed_histogram and plot_windowed_histogram example.

- Alexey Umnov
  skimage.draw.ellipse bug fix and tests.

- Ivana Kajic
  Updated description and examples in documentation for gabor filters

- Matěj Týč
  Extended the image labelling implementation so it also works on 3D images.

- Salvatore Scaramuzzino
  RectTool example

- Kevin Keraudren
  Fix and test for feature.peak_local_max

- Jeremy Metz
  Adaptation of ImageJ Autothresholder.Li, fixed Qhull error QH6228

- Mike Sarahan
  Sub-pixel shift registration

- Jim Fienup, Alexander Iacchetta
  In-depth review of sub-pixel shift registration

- Damian Eads
  Structuring elements in morphology module.

- Egor Panfilov
  Inpainting with biharmonic equation

- Evgeni Burovski
  Adaptation of ImageJ 3D skeletonization algorithm.

- Alex Izvorski
  Color spaces for YUV and related spaces

- Thomas Lewiner
  Design and original implementation of the Lewiner marching cubes algorithm

- Jeff Hemmelgarn
  Minimum threshold

- Kirill Malev
  Frangi and Hessian filters implementation

- Abdeali Kothari
  Alpha blending to convert from rgba to rgb

- Jeyson Molina
  Niblack and Sauvola Local thresholding

- Scott Sievert
  Wavelet denoising

- Gleb Goussarov
  Chan-Vese Segmentation

- Kevin Mader
  Montage improvements

- Matti Eskelinen
  ImageCollection improvements

- David Volgyes
  Unsharp masking

- Lars Grüter
  Flood-fill based local maxima detection

- Solutus Immensus
  Histogram matching

- Laurent P. René de Cotret
  Implementation of masked image translation registration

- Mark Harfouche
  Enabled GIL free operation of many algorithms implemented in Cython.
  Maintenance of the build and test infrastructure.

- Taylor D. Scott
  Simplified `_upsampled_dft` and extended `register_translation` to nD images.

- David J. Mellert
  Polar and log-polar warping, nD windows

- Sebastian Wallkötter
  morphology.rolling_ball and morphology.rolling_ellipsoid
