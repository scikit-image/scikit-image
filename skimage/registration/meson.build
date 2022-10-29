python_sources = [
  '__init__.py',
  '_masked_phase_cross_correlation.py',
  '_optical_flow.py',
  '_optical_flow_utils.py',
  '_phase_cross_correlation.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/registration'  # Folder relative to site-packages to install to
)

subdir('tests')
