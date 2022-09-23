extensions = [
  '_denoise_cy',
  '_inpaint',
  '_nl_means_denoising',
  '_rolling_ball_cy',
  '_unwrap_1d',
  '_unwrap_2d',
  '_unwrap_3d'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/restoration'
  )
endforeach

python_sources = [
  '__init__.py',
  '_cycle_spin.py',
  '_denoise.py',
  'deconvolution.py',
  'inpaint.py',
  'j_invariant.py',
  'non_local_means.py',
  'rolling_ball.py',
  'uft.py',
  'unwrap.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/restoration'  # Folder relative to site-packages to install to
)

subdir('tests')
