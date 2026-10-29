python_sources = [
    '_inpaint.py',
    'inpaint.py',
    'non_local_means.py',
    '_cycle_spin.py',
    '__init__.pyi',
    'j_invariant.py',
    'deconvolution.py',
    '__init__.py',
    'unwrap.py',
    '_unwrap_1d.py',
    '_unwrap_3d.py',
    '_unwrap_2d.py',
    'uft.py',
    '_rolling_ball.py',
    '_denoise_cy.py',
    '_nl_means_denoising.py',
    '_denoise.py',
    '_rolling_ball_cy.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/restoration'  # Folder relative to site-packages to install to
)
