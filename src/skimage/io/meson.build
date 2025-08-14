python_sources = [
  '__init__.py',
  '_image_stack.py',
  '_io.py',
  'collection.py',
  'manage_plugins.py',
  'sift.py',
  'util.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/io'  # Folder relative to site-packages to install to
)

subdir('_plugins')
subdir('tests')
