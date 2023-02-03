pyx_files = [
  ['corner_cy', cython_gen],
  ['censure_cy', cython_gen],
  ['orb_cy', cython_gen],
  ['_canny_cy', cython_gen],
  ['_texture', cython_gen],
  ['_hoghistogram', cython_gen],
  ['_sift', cython_gen],
  ['_cascade', cython_gen_cpp],
  ['_haar', cython_gen_cpp]
]

foreach pyx_file: pyx_files
  py3.extension_module(pyx_file[0],
    pyx_file[1].process(pyx_file[0] + '.pyx'),
    c_args: cython_c_args,
    cpp_args: cython_cpp_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/feature'
  )
endforeach

_brief_pythran = custom_target('_brief_pythran',
  output: ['brief_cy.cpp'],
  input: 'brief_pythran.py',
  command: [pythran, '-E', '@INPUT@', '-o', '@OUTPUT@']
)

py3.extension_module('brief_cy',
  [_brief_pythran],
  cpp_args: cpp_args_pythran,
  include_directories: [incdir_pythran, incdir_numpy],
  dependencies: [py3_dep],
  install: true,
  subdir: 'skimage/feature'
)

_hessian_det_appx_pythran = custom_target('_hessian_det_appx_pythran',
  output: ['_hessian_det_appx.cpp'],
  input: '_hessian_det_appx_pythran.py',
  command: [pythran, '-E', '@INPUT@', '-o', '@OUTPUT@']
)

py3.extension_module('_hessian_det_appx',
  [_hessian_det_appx_pythran],
  cpp_args: cpp_args_pythran,
  include_directories: [incdir_pythran, incdir_numpy],
  dependencies: [py3_dep],
  install: true,
  subdir: 'skimage/feature'
)

python_sources = [
  '__init__.py',
  '_basic_features.py',
  '_canny.py',
  '_daisy.py',
  '_hog.py',
  '_orb_descriptor_positions.py',
  'blob.py',
  'brief.py',
  'censure.py',
  'corner.py',
  'haar.py',
  'match.py',
  'orb.py',
  'peak.py',
  'sift.py',
  'template.py',
  'texture.py',
  'util.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/feature'  # Folder relative to site-packages to install to
)

py3.install_sources(
  ['orb_descriptor_positions.txt'],
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/feature'  # Folder relative to site-packages to install to
)

subdir('tests')
