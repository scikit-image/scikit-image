python_sources = [
    '__init__.pyi',
    '_structural_similarity.py',
    '__init__.py',
    '_adapted_rand_error.py',
    '_variation_of_information.py',
    '_contingency_table.py',
    'simple_metrics.py',
    'set_metrics.py',
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: 'skimage/metrics'  # Folder relative to site-packages to install to
)
