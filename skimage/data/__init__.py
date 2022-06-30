from .._shared import lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules={},
    submod_attrs={
        '_binary_blobs': ['binary_blobs'],
        '_fetchers': ['create_image_fetcher', 'data_dir', 'download_all',
                      'file_hash', 'image_fetcher', 'astronaut',
                      'binary_blobs', 'brain', 'brick', 'camera', 'cat',
                      'cell', 'cells3d', 'checkerboard', 'chelsea', 'clock',
                      'coffee', 'coins', 'colorwheel', 'eagle', 'grass',
                      'gravel', 'horse', 'hubble_deep_field', 'human_mitosis',
                      'immunohistochemistry', 'kidney',
                      'lbp_frontal_face_cascade_filename', 'lfw_subset',
                      'lily', 'logo', 'microaneurysms', 'moon',
                      'nickel_solidification', 'page', 'protein_transport',
                      'retina', 'rocket', 'shepp_logan_phantom',
                      'skin', 'stereo_motorcycle', 'text', 'vortex']
    }
)
