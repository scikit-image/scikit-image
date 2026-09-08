python_sources = [
    'sift.py',
    'manage_plugins.py',
    '_image_stack.py',
    '_io.py',
    'util.py',
    '__init__.py',
    'collection.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/io'  # Folder relative to site-packages to install to
)

subdir('_plugins')
