python_sources = [
  '__init__.py',
  '__init__.pyi',
  'manual_segmentation.py',
  'trainable_segmentation.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/future'  # Folder relative to site-packages to install to
)

subdir('tests')
