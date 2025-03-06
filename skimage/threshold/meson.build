extensions = [
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: ['../_shared'],
    dependencies: [np_dep],
    install: true,
    subdir: 'skimage/threshold'
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_thresholding.py'
]

py3.install_sources(
  python_sources,
  pure: false,                 # Will be installed next to binaries
  subdir: 'skimage/threshold'  # Folder relative to site-packages to install to
)

subdir('tests')
