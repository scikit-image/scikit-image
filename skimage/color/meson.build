
python_sources = [
  '__init__.py',
  'adapt_rgb.py',
  'colorconv.py',
  'colorlabel.py',
  'delta_e.py',
  'rgb_colors.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/color'  # Folder relative to site-packages to install to
)

subdir('tests')
