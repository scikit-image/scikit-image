python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_binary_blobs.py',
  '_fetchers.py',
  '_registry.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/data'  # Folder relative to site-packages to install to
)
