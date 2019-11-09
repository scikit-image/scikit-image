"""Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""

import os as _os

import numpy as _np
from warnings import warn

from ..util.dtype import img_as_bool
from ._binary_blobs import binary_blobs
from ._detect import lbp_frontal_face_cascade_filename

import os.path as osp
data_dir = osp.abspath(osp.dirname(__file__))

__all__ = ['data_dir',
           'load',
           'astronaut',
           'binary_blobs',
           'brick',
           'camera',
           'cell',
           'checkerboard',
           'chelsea',
           'clock',
           'coffee',
           'coins',
           'colorwheel',
           'grass',
           'gravel',
           'horse',
           'hubble_deep_field',
           'immunohistochemistry',
           'lbp_frontal_face_cascade_filename',
           'lfw_subset',
           'logo',
           'microaneurysms',
           'moon',
           'page',
           'text',
           'retina',
           'rocket',
           'rough_wall',
           'shepp_logan_phantom',
           'stereo_motorcycle']


def load(f, as_gray=False):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.

    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.

    Notes
    -----
    This functions is deprecated and will be removed in 0.18.
    """
    warn('This function is deprecated and will be removed in 0.18. '
         'Use `skimage.io.load` or `imageio.imread` directly.',
         stacklevel=2)
    return _load(f, as_gray=as_gray)


def _load(f, as_gray=False):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.
    as_gray : bool, optional
        Whether to convert the image to grayscale.

    Returns
    -------
    img : ndarray
        Image loaded from ``skimage.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from ..io import imread
    return imread(_os.path.join(data_dir, f), plugin='pil', as_gray=as_gray)


def camera():
    """Gray-level "camera" image.

    Often used for segmentation and denoising examples.

    Returns
    -------
    camera : (512, 512) uint8 ndarray
        Camera image.
    """
    return _load("camera.png")


def astronaut():
    """Color image of the astronaut Eileen Collins.

    Photograph of Eileen Collins, an American astronaut. She was selected
    as an astronaut in 1992 and first piloted the space shuttle STS-63 in
    1995. She retired in 2006 after spending a total of 38 days, 8 hours
    and 10 minutes in outer space.

    This image was downloaded from the NASA Great Images database
    <https://flic.kr/p/r9qvLn>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    astronaut : (512, 512, 3) uint8 ndarray
        Astronaut image.
    """

    return _load("astronaut.png")


def brick():
    """Brick wall.

    Returns
    -------
    brick: (512, 512) uint8 image
        A small section of a brick wall.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Bricks25>`_ and licensed
    under the Creative Commons CC0 License.

    A perspective transform was then applied to the image, prior to
    rotating it by 90 degrees, cropping and scaling it to obtain the final
    image.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Bricks25/%5B2K%5DBricks25.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Bricks25.zip', 'bw') as f:
    ...     f.write(r.content)
    >>> with zipfile.ZipFile('[2K]Bricks25.zip') as z:
    ... z.extract('Bricks25_col.jpg')

    >>> from numpy.linalg import inv
    >>> from skimage.transform import rescale, warp, rotate
    >>> from skimage.color import rgb2gray
    >>> from imageio import imread, imwrite
    >>> from skimage import img_as_ubyte
    >>> import numpy as np


    >>> # Obtained playing around with GIMP 2.10 with their perspective tool
    >>> H = inv(np.asarray([[ 0.54764, -0.00219, 0],
    ...                     [-0.12822,  0.54688, 0],
    ...                     [-0.00022,        0, 1]]))


    >>> brick_orig = imread('Bricks25_col.jpg')
    >>> brick = warp(brick_orig, H)
    >>> brick = rescale(brick[:1024, :1024], (0.5, 0.5, 1))
    >>> brick = rotate(brick, -90)
    >>> imwrite('brick.png', img_as_ubyte(rgb2gray(brick)))
    """
    return _load("brick.png", as_gray=True)


def grass():
    """Grass.

    Returns
    -------
    grass: (512, 512) uint8 image
        Some grass.

    Notes
    -----
    The original image was downloaded from
    `DeviantArt <https://www.deviantart.com/linolafett/art/Grass-01-434853879>`__
    and licensed underthe Creative Commons CC0 License.

    The downloaded image was cropped to include a region of ``(512, 512)``
    pixels around the top left corner, converted to grayscale, then to uint8
    prior to saving the result in PNG format.

    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile
    >>> url = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/a407467e-4ff0-49f1-923f-c9e388e84612/d76wfef-2878b78d-5dce-43f9-be36->>     26ec9bc0df3b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2E0MDc0NjdlLTRmZjAtNDlmMS05MjNmLWM5ZTM4OGU4NDYxMlwvZDc2d2ZlZi0yODc4Yjc4ZC01ZGNlLTQzZjktYmUzNi0yNmVjOWJjMGRmM2IuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.98hIcOTCqXWQ67Ec5bM5eovKEn2p91mWB3uedH61ynI'
    >>> r = requests.get(url)
    >>> with open('grass_orig.jpg', 'bw') as f:
    ...     f.write(r.content)
    >>> grass_orig = imageio.imread('grass_orig.jpg')
    >>> grass = skimage.img_as_ubyte(skimage.color.rgb2gray(grass_orig[:512, :512]))
    >>> imageio.imwrite('grass.png', grass)
    """
    return _load("grass.png", as_gray=True)


def rough_wall():
    """Rough wall.

    Returns
    -------
    rough_wall: (512, 512) uint8 image
        Some rough wall.

    """
    from warnings import warn
    warn("The rough_wall dataset has been removed due to licensing concerns."
         "It has been replaced with the gravel dataset. This warning message"
         "will be replaced with an error in scikit-image 0.17.", stacklevel=2)
    return gravel()


def gravel():
    """Gravel

    Returns
    -------
    gravel: (512, 512) uint8 image
        Grayscale gravel sample.

    Notes
    -----
    The original image was downloaded from
    `CC0Textures <https://cc0textures.com/view.php?tex=Gravel04>`__ and
    licensed under the Creative Commons CC0 License.

    The downloaded image was then rescaled to ``(1024, 1024)``, then the
    top left ``(512, 512)`` pixel region  was cropped prior to converting the
    image to grayscale and uint8 data type. The result was saved using the
    PNG format.
    """

    """
    The following code was used to obtain the final image.

    >>> import sys; print(sys.version)
    >>> import platform; print(platform.platform())
    >>> import skimage; print(f"scikit-image version: {skimage.__version__}")
    >>> import numpy; print(f"numpy version: {numpy.__version__}")
    >>> import imageio; print(f"imageio version {imageio.__version__}")
    3.7.3 | packaged by conda-forge | (default, Jul  1 2019, 21:52:21)
    [GCC 7.3.0]
    Linux-5.0.0-20-generic-x86_64-with-debian-buster-sid
    scikit-image version: 0.16.dev0
    numpy version: 1.16.4
    imageio version 2.4.1

    >>> import requests
    >>> import zipfile

    >>> url = 'https://cdn.struffelproductions.com/file/cc0textures/Gravel04/%5B2K%5DGravel04.zip'
    >>> r = requests.get(url)
    >>> with open('[2K]Gravel04.zip', 'bw') as f:
    ...     f.write(r.content)

    >>> with zipfile.ZipFile('[2K]Gravel04.zip') as z:
    ...     z.extract('Gravel04_col.jpg')

    >>> from skimage.transform import resize
    >>> gravel_orig = imageio.imread('Gravel04_col.jpg')
    >>> gravel = resize(gravel_orig, (1024, 1024))
    >>> gravel = skimage.img_as_ubyte(skimage.color.rgb2gray(gravel[:512, :512]))
    >>> imageio.imwrite('gravel.png', gravel)
    """
    return _load("gravel.png", as_gray=True)


def text():
    """Gray-level "text" image used for corner detection.

    Notes
    -----
    This image was downloaded from Wikipedia
    <https://en.wikipedia.org/wiki/File:Corner.png>`__.

    No known copyright restrictions, released into the public domain.

    Returns
    -------
    text : (172, 448) uint8 ndarray
        Text image.
    """

    return _load("text.png")


def checkerboard():
    """Checkerboard image.

    Checkerboards are often used in image calibration, since the
    corner-points are easy to locate.  Because of the many parallel
    edges, they also visualise distortions particularly well.

    Returns
    -------
    checkerboard : (200, 200) uint8 ndarray
        Checkerboard image.
    """
    return _load("chessboard_GRAY.png")


def cell():
    """Cell floating in saline.

    This is a quantitative phase image retrieved from a digital hologram using
    the Python library ``qpformat``. The image shows a cell with high phase
    value, above the background phase.

    Because of a banding pattern artifact in the background, this image is a
    good test of thresholding algorithms. The pixel spacing is 0.107 µm.

    These data were part of a comparison between several refractive index
    retrieval techniques for spherical objects as part of [1]_.

    This image is CC0, dedicated to the public domain. You may copy, modify, or
    distribute it without asking permission.

    Returns
    -------
    cell : (660, 550) uint8 array
        Image of a cell.

    References
    ----------
    .. [1] Paul Müller, Mirjam Schürmann, Salvatore Girardo, Gheorghe Cojoc,
           and Jochen Guck. "Accurate evaluation of size and refractive index
           for spherical objects in quantitative phase imaging." Optics Express
           26(8): 10729-10743 (2018). :DOI:`10.1364/OE.26.010729`
    """
    return _load('cell.png')


def coins():
    """Greek coins from Pompeii.

    This image shows several coins outlined against a gray background.
    It is especially useful in, e.g. segmentation tests, where
    individual objects need to be identified against a background.
    The background shares enough grey levels with the coins that a
    simple segmentation is not sufficient.

    Notes
    -----
    This image was downloaded from the
    `Brooklyn Museum Collection
    <https://www.brooklynmuseum.org/opencollection/archives/image/51611>`__.

    No known copyright restrictions.

    Returns
    -------
    coins : (303, 384) uint8 ndarray
        Coins image.
    """
    return _load("coins.png")


def logo():
    """Scikit-image logo, a RGBA image.

    Returns
    -------
    logo : (500, 500, 4) uint8 ndarray
        Logo image.
    """
    return _load("logo.png")


def microaneurysms():
    """Gray-level "microaneurysms" image.

    Detail from an image of the retina (green channel).
    The image is a crop of image 07_dr.JPG from the
    High-Resolution Fundus (HRF) Image Database:
    https://www5.cs.fau.de/research/data/fundus-images/

    Notes
    -----
    No copyright restrictions. CC0 given by owner (Andreas Maier).

    Returns
    -------
    microaneurysms : (102, 102) uint8 ndarray
        Retina image with lesions.

    References
    ----------
    .. [1] Budai, A., Bock, R, Maier, A., Hornegger, J.,
           Michelson, G. (2013).  Robust Vessel Segmentation in Fundus
           Images. International Journal of Biomedical Imaging, vol. 2013,
           2013.
           :DOI:`10.1155/2013/154860`
    """
    return _load("microaneurysms.png")


def moon():
    """Surface of the moon.

    This low-contrast image of the surface of the moon is useful for
    illustrating histogram equalization and contrast stretching.

    Returns
    -------
    moon : (512, 512) uint8 ndarray
        Moon image.
    """
    return _load("moon.png")


def page():
    """Scanned page.

    This image of printed text is useful for demonstrations requiring uneven
    background illumination.

    Returns
    -------
    page : (191, 384) uint8 ndarray
        Page image.
    """
    return _load("page.png")


def horse():
    """Black and white silhouette of a horse.

    This image was downloaded from
    `openclipart <http://openclipart.org/detail/158377/horse-by-marauder>`

    No copyright restrictions. CC0 given by owner (Andreas Preuss (marauder)).

    Returns
    -------
    horse : (328, 400) bool ndarray
        Horse image.
    """
    return img_as_bool(_load("horse.png", as_gray=True))


def clock():
    """Motion blurred clock.

    This photograph of a wall clock was taken while moving the camera in an
    aproximately horizontal direction.  It may be used to illustrate
    inverse filters and deconvolution.

    Released into the public domain by the photographer (Stefan van der Walt).

    Returns
    -------
    clock : (300, 400) uint8 ndarray
        Clock image.
    """
    return _load("clock_motion.png")


def immunohistochemistry():
    """Immunohistochemical (IHC) staining with hematoxylin counterstaining.

    This picture shows colonic glands where the IHC expression of FHL2 protein
    is revealed with DAB. Hematoxylin counterstaining is applied to enhance the
    negative parts of the tissue.

    This image was acquired at the Center for Microscopy And Molecular Imaging
    (CMMI).

    No known copyright restrictions.

    Returns
    -------
    immunohistochemistry : (512, 512, 3) uint8 ndarray
        Immunohistochemistry image.
    """
    return _load("ihc.png")


def chelsea():
    """Chelsea the cat.

    An example with texture, prominent edges in horizontal and diagonal
    directions, as well as features of differing scales.

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Stefan van der Walt).

    Returns
    -------
    chelsea : (300, 451, 3) uint8 ndarray
        Chelsea image.
    """
    return _load("chelsea.png")


def coffee():
    """Coffee cup.

    This photograph is courtesy of Pikolo Espresso Bar.
    It contains several elliptical shapes as well as varying texture (smooth
    porcelain to course wood grain).

    Notes
    -----
    No copyright restrictions.  CC0 by the photographer (Rachel Michetti).

    Returns
    -------
    coffee : (400, 600, 3) uint8 ndarray
        Coffee image.
    """
    return _load("coffee.png")


def hubble_deep_field():
    """Hubble eXtreme Deep Field.

    This photograph contains the Hubble Telescope's farthest ever view of
    the universe. It can be useful as an example for multi-scale
    detection.

    Notes
    -----
    This image was downloaded from
    `HubbleSite
    <http://hubblesite.org/newscenter/archive/releases/2012/37/image/a/>`__.

    The image was captured by NASA and `may be freely used in the public domain
    <http://www.nasa.gov/audience/formedia/features/MP_Photo_Guidelines.html>`_.

    Returns
    -------
    hubble_deep_field : (872, 1000, 3) uint8 ndarray
        Hubble deep field image.
    """
    return _load("hubble_deep_field.jpg")


def retina():
    """Human retina.

    This image of a retina is useful for demonstrations requiring circular
    images.

    Notes
    -----
    This image was downloaded from
    `wikimedia <https://commons.wikimedia.org/wiki/File:Fundus_photograph_of_normal_left_eye.jpg>`.
    This file is made available under the Creative Commons CC0 1.0 Universal
    Public Domain Dedication.

    References
    ----------
    .. [1] Häggström, Mikael (2014). "Medical gallery of Mikael Häggström 2014".
           WikiJournal of Medicine 1 (2). :DOI:`10.15347/wjm/2014.008`.
           ISSN 2002-4436. Public Domain

    Returns
    -------
    retina : (1411, 1411, 3) uint8 ndarray
        Retina image in RGB.
    """
    return _load("retina.jpg")


def shepp_logan_phantom():
    """Shepp Logan Phantom.

    References
    ----------
    .. [1] L. A. Shepp and B. F. Logan, "The Fourier reconstruction of a head
           section," in IEEE Transactions on Nuclear Science, vol. 21,
           no. 3, pp. 21-43, June 1974. :DOI:`10.1109/TNS.1974.6499235`

    Returns
    -------
    phantom: (400, 400) float64 image
        Image of the Shepp-Logan phantom in grayscale.
    """
    return _load("phantom.png", as_gray=True)


def colorwheel():
    """Color Wheel.

    Returns
    -------
    colorwheel: (370, 371, 3) uint8 image
        A colorwheel.
    """
    return _load("color.png")


def rocket():
    """Launch photo of DSCOVR on Falcon 9 by SpaceX.

    This is the launch photo of Falcon 9 carrying DSCOVR lifted off from
    SpaceX's Launch Complex 40 at Cape Canaveral Air Force Station, FL.

    Notes
    -----
    This image was downloaded from
    `SpaceX Photos
    <https://www.flickr.com/photos/spacexphotos/16511594820/in/photostream/>`__.

    The image was captured by SpaceX and `released in the public domain
    <http://arstechnica.com/tech-policy/2015/03/elon-musk-puts-spacex-photos-into-the-public-domain/>`_.

    Returns
    -------
    rocket : (427, 640, 3) uint8 ndarray
        Rocket image.
    """
    return _load("rocket.jpg")


def stereo_motorcycle():
    """Rectified stereo image pair with ground-truth disparities.

    The two images are rectified such that every pixel in the left image has
    its corresponding pixel on the same scanline in the right image. That means
    that both images are warped such that they have the same orientation but a
    horizontal spatial offset (baseline). The ground-truth pixel offset in
    column direction is specified by the included disparity map.

    The two images are part of the Middlebury 2014 stereo benchmark. The
    dataset was created by Nera Nesic, Porter Westling, Xi Wang, York Kitajima,
    Greg Krathwohl, and Daniel Scharstein at Middlebury College. A detailed
    description of the acquisition process can be found in [1]_.

    The images included here are down-sampled versions of the default exposure
    images in the benchmark. The images are down-sampled by a factor of 4 using
    the function `skimage.transform.downscale_local_mean`. The calibration data
    in the following and the included ground-truth disparity map are valid for
    the down-sampled images::

        Focal length:           994.978px
        Principal point x:      311.193px
        Principal point y:      254.877px
        Principal point dx:      31.086px
        Baseline:               193.001mm

    Returns
    -------
    img_left : (500, 741, 3) uint8 ndarray
        Left stereo image.
    img_right : (500, 741, 3) uint8 ndarray
        Right stereo image.
    disp : (500, 741, 3) float ndarray
        Ground-truth disparity map, where each value describes the offset in
        column direction between corresponding pixels in the left and the right
        stereo images. E.g. the corresponding pixel of
        ``img_left[10, 10 + disp[10, 10]]`` is ``img_right[10, 10]``.
        NaNs denote pixels in the left image that do not have ground-truth.

    Notes
    -----
    The original resolution images, images with different exposure and
    lighting, and ground-truth depth maps can be found at the Middlebury
    website [2]_.

    References
    ----------
    .. [1] D. Scharstein, H. Hirschmueller, Y. Kitajima, G. Krathwohl, N.
           Nesic, X. Wang, and P. Westling. High-resolution stereo datasets
           with subpixel-accurate ground truth. In German Conference on Pattern
           Recognition (GCPR 2014), Muenster, Germany, September 2014.
    .. [2] http://vision.middlebury.edu/stereo/data/scenes2014/

    """
    return (_load("motorcycle_left.png"),
            _load("motorcycle_right.png"),
            _np.load(_os.path.join(data_dir, "motorcycle_disp.npz"))["arr_0"])


def lfw_subset():
    """Subset of data from the LFW dataset.

    This database is a subset of the LFW database containing:

    * 100 faces
    * 100 non-faces

    The full dataset is available at [2]_.

    Returns
    -------
    images : (200, 25, 25) uint8 ndarray
        100 first images are faces and subsequent 100 are non-faces.

    Notes
    -----
    The faces were randomly selected from the LFW dataset and the non-faces
    were extracted from the background of the same dataset. The cropped ROIs
    have been resized to a 25 x 25 pixels.

    References
    ----------
    .. [1] Huang, G., Mattar, M., Lee, H., & Learned-Miller, E. G. (2012).
           Learning to align from scratch. In Advances in Neural Information
           Processing Systems (pp. 764-772).
    .. [2] http://vis-www.cs.umass.edu/lfw/

    """
    return _np.load(_os.path.join(data_dir, 'lfw_subset.npy'))
