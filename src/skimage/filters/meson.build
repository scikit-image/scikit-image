python_sources = [
    '_gabor.py',
    'ridges.py',
    '__init__.pyi',
    '_sparse.py',
    '_median.py',
    '_multiotsu.py',
    '__init__.py',
    'thresholding.py',
    'edges.py',
    '_unsharp_mask.py',
    '_fft_based.py',
    '_rank_order.py',
    'lpi_filter.py',
    '_window.py',
    '_gaussian.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/filters'  # Folder relative to site-packages to install to
)
