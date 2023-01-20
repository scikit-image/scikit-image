
python_sources = [
  '__init__.py',
  '_adapted_rand_error.py',
  '_contingency_table.py',
  '_structural_similarity.py',
  '_variation_of_information.py',
  'set_metrics.py',
  'simple_metrics.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/metrics'  # Folder relative to site-packages to install to
)

subdir('tests')
