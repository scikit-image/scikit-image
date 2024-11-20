extensions = [
  '_hough_transform',
  '_radon_transform',
  '_warps_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: ['../_shared'],
    dependencies: [np_dep],
    install: true,
    subdir: 'skimage/transform'
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_geometric.py',
  '_thin_plate_splines.py',
  '_warps.py',
  'finite_radon_transform.py',
  'hough_transform.py',
  'integral.py',
  'pyramids.py',
  'radon_transform.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/transform'  # Folder relative to site-packages to install to
)

subdir('tests')
