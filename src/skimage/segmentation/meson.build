python_sources = [
    '__init__.pyi',
    'boundaries.py',
    'random_walker_segmentation.py',
    '_felzenszwalb.py',
    '__init__.py',
    '_quickshift.py',
    '_join.py',
    'slic_superpixels.py',
    'active_contour_model.py',
    'morphsnakes.py',
    '_slic.py',
    '_chan_vese.py',
    '_watershed.py',
    '_clear_border.py',
    '_felzenszwalb_cy.py',
    '_quickshift_cy.py',
    '_expand_labels.py',
    '_watershed_cy.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/segmentation'  # Folder relative to site-packages to install to
)
