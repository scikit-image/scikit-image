extensions = [
  '_bilateral_cy',
  '_core_cy',
  '_core_cy_3d',
  '_generic_cy',
  '_percentile_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../../_shared'],
    install: true,
    subdir: 'skimage/filters/rank'
  )
endforeach

python_sources = [
  '__init__.py',
  '_percentile.py',
  '_bilateral.py',
  '_generic.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/filters/rank'  # Folder relative to site-packages to install to
)

subdir('tests')
