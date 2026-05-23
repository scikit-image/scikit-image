python_sources = [
    'version.py',
    'copyfiles.py',
    '__init__.py',
    'gcc_build_bitness.py',
    'tempita.py',
    'cythoner.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/_build_utils'  # Folder relative to site-packages to install to
)
