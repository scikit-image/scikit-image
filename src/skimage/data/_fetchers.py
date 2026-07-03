"""
Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php


"""

from _skimage2.data._fetchers import (
    astronaut as astronaut,
    brain as brain,
    brick as brick,
    camera as camera,
    cat as cat,
    cell as cell,
    cells3d as cells3d,
    checkerboard as checkerboard,
    chelsea as chelsea,
    clock as clock,
    coffee as coffee,
    coins as coins,
    colorwheel as colorwheel,
    data_dir as data_dir,
    legacy_data_dir as legacy_data_dir,
    download_all as download_all,
    eagle as eagle,
    file_hash as file_hash,
    grass as grass,
    gravel as gravel,
    horse as horse,
    hubble_deep_field as hubble_deep_field,
    human_mitosis as human_mitosis,
    immunohistochemistry as immunohistochemistry,
    kidney as kidney,
    lbp_frontal_face_cascade_filename as lbp_frontal_face_cascade_filename,
    lfw_subset as lfw_subset,
    lily as lily,
    logo as logo,
    microaneurysms as microaneurysms,
    moon as moon,
    nickel_solidification as nickel_solidification,
    page as page,
    palisades_of_vogt as palisades_of_vogt,
    protein_transport as protein_transport,
    retina as retina,
    rocket as rocket,
    shepp_logan_phantom as shepp_logan_phantom,
    skin as skin,
    stereo_motorcycle as stereo_motorcycle,
    text as text,
    vortex as vortex,
)  # noqa: F401

__all__ = [
    'astronaut',
    'brain',
    'brick',
    'camera',
    'cat',
    'cell',
    'cells3d',
    'checkerboard',
    'chelsea',
    'clock',
    'coffee',
    'coins',
    'colorwheel',
    'data_dir',
    'legacy_data_dir',
    'download_all',
    'eagle',
    'file_hash',
    'grass',
    'gravel',
    'horse',
    'hubble_deep_field',
    'human_mitosis',
    'immunohistochemistry',
    'kidney',
    'lbp_frontal_face_cascade_filename',
    'lfw_subset',
    'lily',
    'logo',
    'microaneurysms',
    'moon',
    'nickel_solidification',
    'page',
    'palisades_of_vogt',
    'protein_transport',
    'retina',
    'rocket',
    'shepp_logan_phantom',
    'skin',
    'stereo_motorcycle',
    'text',
    'vortex',
]

from _skimage2.data._fetchers import _image_fetcher  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
