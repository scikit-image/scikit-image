python_sources = [
    'pil_plugin.py',
    'imageio_plugin.py',
    '__init__.py',
    'fits_plugin.py',
    'gdal_plugin.py',
    'matplotlib_plugin.py',
    'tifffile_plugin.py',
    'simpleitk_plugin.py',
    'imread_plugin.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/io/_plugins'  # Folder relative to site-packages to install to
)
