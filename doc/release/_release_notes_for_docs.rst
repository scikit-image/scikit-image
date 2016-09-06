We're happy to announce the release of scikit-image v0.12!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org

and our gallery of examples

http://scikit-image.org/docs/dev/auto_examples/

Highlights and new features
---------------------------

For this release, we merged over 200 pull requests with bug fixes,
cleanups, improved documentation and new features.  Highlights
include:

- 3D skeletonization (#1923)
- Parallelization framework using ``dask``:``skimage.util.apply_parallel``
  (#1493)
- Laplacian operator ``filters.laplace`` (#1763)
- Synthetic 2-D and 3-D binary data with rounded blobs (#1485)
- Plugin for ``imageio`` library (#1575)
- Inpainting algorithm (#1804)
- New handling of background pixels for ``measure.label``: 0-valued
  pixels are considered as background by default, and the label of
  background pixels is 0.
- Partial support of 3-D images for ``skimage.measure.regionprops``
  (#1505)
- Multi-block local binary patterns (MB-LBP) for texture classification (#1536)
- Seam Carving (resizing without distortion) (#1459)
- Simple image comparison metrics (PSNR, NRMSE) (#1897)
- Region boundary based region adjacency graphs (RAG) (#1495)
- Construction of RAG from label images (#1826)
- ``morphology.remove_small_holes`` now complements
  ``morphology.remove_small_objects`` (#1689)
- Faster cython implementation of ``morphology.skeletonize``
- More appropriate default weights in
  ``restoration.denoise_tv_chambolle`` and ``feature.peak_local_max``
- Correction of bug in the computation of the Euler characteristic in
  ``measure.regionprops``.
- Replace jet by viridis as default colormap
- Alpha layer support for ``color.gray2rgb``
- The measure of structural similarity (``measure.compare_ssim``) is now
  n-dimensional and supports color channels as well.
- We fixed an issue related to incorrect propagation in plateaus in
  ``segmentation.watershed``

Documentation:

- New organization of gallery of examples in sections
- More frequent (automated) updates of online documentation

API Changes
-----------

- ``equalize_adapthist`` now takes a ``kernel_size`` keyword argument,
  replacing  the ``ntiles_*`` arguments.
- The functions ``blob_dog``, ``blob_log`` and ``blob_doh`` now return
  float  arrays instead of integer arrays.
- ``transform.integrate`` now takes lists of tuples instead of integers
  to define the window over which to integrate.
- `reverse_map` parameter in `skimage.transform.warp` has been removed.
- `enforce_connectivity` in `skimage.segmentation.slic` defaults to ``True``.
- `skimage.measure.fit.BaseModel._params`,
  `skimage.transform.ProjectiveTransform._matrix`,
  `skimage.transform.PolynomialTransform._params`,
  `skimage.transform.PiecewiseAffineTransform.affines_*` attributes
  have been removed.
- `skimage.filters.denoise_*` have moved to `skimage.restoration.denoise_*`.

Deprecations
------------

- ``filters.gaussian_filter`` has been renamed ``filters.gaussian``
- ``filters.gabor_filter`` has been renamed ``filters.gabor``
- ``restoration.nl_means_denoising`` has been renamed
  ``restoration.denoise_nl_means``
- ``measure.LineModel`` was deprecated in favor of ``measure.LineModelND``
- ``measure.structural_similarity`` has been renamed
  ``measure.compare_ssim``
- ``data.lena`` has been deprecated, and gallery examples use instead the
  ``data.astronaut()`` picture.

Contributors to this release
----------------------------
(Listed alphabetically by last name)

- K.-Michael Aye
- Connelly Barnes
- Sumit Binnani
- Vighnesh Birodkar
- François Boulogne
- Matthew Brett
- Steven Brown
- Arnaud De Bruecker
- Olivier Debeir
- Charles Deledalle
- endolith
- Eric Lubeck
- Victor Escorcia
- Ivo Flipse
- Joel Frederico
- Cédric Gilon
- Christoph Gohlke
- Korijn van Golen
- Emmanuelle Gouillart
- J. Goutin
- Blake Griffith
- M. Hawker
- Jonathan Helmus
- Dhruv Jawali
- Lee Kamentsky
- Kevin Keraudren
- Julius Bier Kirkegaard
- David Koeller
- Gustav Larsson
- Gregory R. Lee
- Guillaume Lemaitre
- Benny Lichtner
- Himanshu Mishra
- Juan Nunez-Iglesias
- Ömer Özak
- Leena P.
- Michael Pacer
- Daniil Pakhomov
- David Perez-Suarez
- Egor Panfilov
- David PS
- Sergio Pascual
- Ariel Rokem
- Nicolas Rougier
- Christian Sachs
- Kshitij Saraogi
- Martin Savc
- Johannes Schönberger
- Arve Seljebu
- Tim Sheerman-Chase
- Scott Sievert
- Steven Silvester
- Alexandre Fioravante de Siqueira
- Daichi Suzuo
- Noah Trebesch
- Pratap Vardhan
- Gael Varoquaux
- Stefan van der Walt
- Joshua Warner
- Josh Warner
- Warren Weckesser
- Daniel Wennberg
- John Wiggins
- Robin Wilson
- Olivia Wilson

