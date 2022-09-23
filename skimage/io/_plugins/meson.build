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
  'gtk_plugin.py',
  'imageio_plugin.py',
  'imread_plugin.py',
  'matplotlib_plugin.py',
  'pil_plugin.py',
  'q_color_mixer.py',
  'q_histogram.py',
  'simpleitk_plugin.py',
  'tifffile_plugin.py',
  'util.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/io/_plugins'  # Folder relative to site-packages to install to
)
