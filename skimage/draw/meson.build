py3.extension_module('_draw',
  cython_gen.process('_draw.pyx'),
  c_args: cython_c_args,
  include_directories: ['../_shared'],
  dependencies: [np_dep],
  install: true,
  subdir: 'skimage/draw'
)

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_polygon2mask.py',
  '_random_shapes.py',
  'draw3d.py',
  'draw_nd.py',
  'draw.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/draw'  # Folder relative to site-packages to install to
)

subdir('tests')
