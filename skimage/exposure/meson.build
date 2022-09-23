python_sources = [
  '__init__.py',
  '_adapthist.py',
  'exposure.py',
  'histogram_matching.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/exposure'  # Folder relative to site-packages to install to
)

subdir('tests')
