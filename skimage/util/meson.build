cpp_extensions = [
  '_remap'
]

foreach ext: cpp_extensions
  py3.extension_module(ext,
    cython_gen_cpp.process(ext + '.pyx'),
    cpp_args: cython_cpp_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/util'
  )
endforeach

python_sources = [
  '__init__.py',
  '_invert.py',
  '_label.py',
  '_map_array.py',
  '_montage.py',
  '_regular_grid.py',
  'apply_parallel.py',
  'arraycrop.py',
  'compare.py',
  'dtype.py',
  'lookfor.py',
  'noise.py',
  'shape.py',
  'unique.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/util'  # Folder relative to site-packages to install to
)

subdir('tests')
