py3.extension_module('transform',
  cython_gen.process('transform.pyx'),
  c_args: cython_c_args,
  dependencies: [np_dep],
  install: true,
  subdir: '_skimage2/_shared'
)

py3.extension_module('fast_exp',
  cython_gen.process('fast_exp.pyx'),
  c_args: cython_c_args,
  dependencies: [np_dep],
  install: true,
  subdir: '_skimage2/_shared'
)

py3.extension_module('geometry',
  cython_gen.process('geometry.pyx'),
  c_args: cython_c_args,
  dependencies: [np_dep],
  install: true,
  subdir: '_skimage2/_shared'
)

py3.extension_module('interpolation',
  cython_gen.process('interpolation.pyx'),
  c_args: cython_c_args,
  dependencies: [np_dep],
  install: true,
  subdir: '_skimage2/_shared'
)

python_sources = [
  '__init__.py',
  '_dependency_checks.py',
  '_geometry.py',
  '_tempfile.py',
  '_warnings.py',
  'compat.py',
  'dtype.py',
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
  subdir: '_skimage2/_shared'  # Folder relative to site-packages to install to
)
