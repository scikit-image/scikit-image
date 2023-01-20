extensions = [
  '_colormixer',
  '_histograms'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../../_shared'],
    install: true,
    subdir: 'skimage/io/_plugins'
  )
endforeach

python_sources = [
  '__init__.py',
  'fits_plugin.py',
  'gdal_plugin.py',
  'imageio_plugin.py',
  'imread_plugin.py',
  'matplotlib_plugin.py',
  'pil_plugin.py',
  'simpleitk_plugin.py',
  'tifffile_plugin.py'
]

ini_files = [
  'fits_plugin.ini',
  'gdal_plugin.ini',
  'imageio_plugin.ini',
  'imread_plugin.ini',
  'matplotlib_plugin.ini',
  'pil_plugin.ini',
  'simpleitk_plugin.ini',
  'tifffile_plugin.ini'
]

py3.install_sources(
  python_sources + ini_files,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/io/_plugins'  # Folder relative to site-packages to install to
)
