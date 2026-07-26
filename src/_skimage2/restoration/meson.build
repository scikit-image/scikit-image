extensions = [
  '_denoise_cy',
  '_inpaint',
  '_nl_means_denoising',
  '_rolling_ball_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: ['../../_skimage2/_shared'],
    dependencies: [np_dep],
    install: true,
    subdir: '_skimage2/restoration'
  )
endforeach

wrap_extensions = [
  ['_unwrap_1d', []],
  ['_unwrap_2d', ['unwrap_2d_ljmu.c', 'unwrap_2d_ljmu.h', 'unwrap_common.h']],
  ['_unwrap_3d', ['unwrap_3d_ljmu.c', 'unwrap_3d_ljmu.h', 'unwrap_common.h']],
]

foreach ext: wrap_extensions
  py3.extension_module(ext[0],
    [cython_gen.process(ext[0] + '.pyx')] + ext[1],
    c_args: cython_c_args,
    include_directories: ['../../_skimage2/_shared'],
    dependencies: [np_dep],
    install: true,
    subdir: '_skimage2/restoration'
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_cycle_spin.py',
  '_denoise.py',
  'deconvolution.py',
  'inpaint.py',
  'j_invariant.py',
  'non_local_means.py',
  '_rolling_ball.py',
  'uft.py',
  'unwrap.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: '_skimage2/restoration'  # Folder relative to site-packages to install to
)
