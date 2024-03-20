python_sources = [
  '__init__.py',
  '__init__.pyi',
  '_binary_blobs.py',
  '_fetchers.py',
  '_registry.py',
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/data'  # Folder relative to site-packages to install to
)

py3.install_sources([
  # See legacy_datasets in _registry.py
  'README.txt',
  'astronaut.png',
  'brick.png',
  'camera.png',
  'chessboard_GRAY.png',
  'chessboard_RGB.png',
  'chelsea.png',
  'clock_motion.png',
  'coffee.png',
  'coins.png',
  'color.png',
  'cell.png',
  'grass.png',
  'gravel.png',
  'horse.png',
  'hubble_deep_field.jpg',
  'ihc.png',
  'lbpcascade_frontalface_opencv.xml',
  'lfw_subset.npy',
  'logo.png',
  'microaneurysms.png',
  'moon.png',
  'multipage.tif',
  'no_time_for_that_tiny.gif',
  'page.png',
  'text.png',
  'retina.jpg',
  'rocket.jpg',
  'phantom.png',
  'motorcycle_disp.npz',
  'motorcycle_left.png',
  'motorcycle_right.png',
  ],
  pure: false,
  subdir: 'skimage/data'
)

subdir('tests')
