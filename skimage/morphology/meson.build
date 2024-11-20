_skeletonize_lee_cy = custom_target('_skeletonize_lee_cy',
  output: '_skeletonize_lee_cy.pyx',
  input: '_skeletonize_lee_cy.pyx.in',
  command: [
    py3, tempita, '@INPUT@', '-o', '@OUTDIR@'
  ]
)

extensions = [
  '_convex_hull',
  '_extrema_cy',
  '_flood_fill_cy',
  '_grayreconstruct',
  '_max_tree',
  '_misc_cy',
  '_skeletonize_various_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: ['../_shared'],
    dependencies: [np_dep],
    install: true,
    subdir: 'skimage/morphology'
  )
endforeach

py3.extension_module('_skeletonize_lee_cy',
  cython_gen_cpp.process(_skeletonize_lee_cy),
  cpp_args: cython_cpp_args,
  include_directories: ['../_shared'],
  dependencies: [np_dep],
  install: true,
  subdir: 'skimage/morphology'
)

python_sources = [
  '__init__.py',
  '_flood_fill.py',
  '_skeletonize.py',
  '_util.py',
  'binary.py',
  'convex_hull.py',
  'extrema.py',
  'footprints.py',
  'gray.py',
  'grayreconstruct.py',
  'isotropic.py',
  'max_tree.py',
  'misc.py',
]

data_files = [
  'ball_decompositions.npy',
  'disk_decompositions.npy'
]

py3.install_sources(
  python_sources + data_files,
  pure: false,                  # Will be installed next to binaries
  subdir: 'skimage/morphology'  # Folder relative to site-packages to install to
)

subdir('tests')
