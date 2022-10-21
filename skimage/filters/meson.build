extensions = [
  '_multiotsu'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/filters'
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_fft_based.py',
  '_gabor.py',
  '_gaussian.py',
  '_median.py',
  '_rank_order.py',
  '_sparse.py',
  '_unsharp_mask.py',
  '_window.py',
  'edges.py',
  'lpi_filter.py',
  'ridges.py',
  'thresholding.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/filters'  # Folder relative to site-packages to install to
)

subdir('rank')
subdir('tests')
