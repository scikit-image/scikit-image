py3.extension_module('transform',
  cython_gen.process('transform.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'skimage/_shared'
)

py3.extension_module('fast_exp',
  cython_gen.process('fast_exp.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'skimage/_shared'
)

py3.extension_module('geometry',
  cython_gen.process('geometry.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'skimage/_shared'
)

py3.extension_module('interpolation',
  cython_gen.process('interpolation.pyx'),
  c_args: cython_c_args,
  include_directories: [incdir_numpy],
  install: true,
  subdir: 'skimage/_shared'
)

python_sources = [
  '__init__.py',
  '_dependency_checks.py',
  '_geometry.py',
  '_tempfile.py',
  '_warnings.py',
  'coord.py',
  'filters.py',
  'tester.py',
  'testing.py',
  'utils.py',
  'version_requirements.py',
  'fast_exp.h',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/_shared'  # Folder relative to site-packages to install to
)

subdir('tests')
