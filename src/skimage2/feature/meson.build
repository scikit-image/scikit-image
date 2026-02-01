pyx_files = [
  ['_canny_cy', cython_gen],
]

foreach pyx_file: pyx_files
  py3.extension_module(pyx_file[0],
    pyx_file[1].process(pyx_file[0] + '.pyx'),
    c_args: cython_c_args,
    cpp_args: cython_cpp_args,
    dependencies: [np_dep],
    install: true,
    subdir: 'skimage2/feature'
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_canny.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage2/feature'  # Folder relative to site-packages to install to
)
