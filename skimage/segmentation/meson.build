extensions = [
  '_felzenszwalb_cy',
  '_quickshift_cy',
  '_slic',
  '_watershed_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/segmentation'
  )
endforeach

python_sources = [
  '__init__.py',
  '_chan_vese.py',
  '_clear_border.py',
  '_expand_labels.py',
  '_felzenszwalb.py',
  '_join.py',
  '_quickshift.py',
  '_watershed.py',
  'active_contour_model.py',
  'boundaries.py',
  'morphsnakes.py',
  'random_walker_segmentation.py',
  'slic_superpixels.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/segmentation'  # Folder relative to site-packages to install to
)

subdir('tests')
