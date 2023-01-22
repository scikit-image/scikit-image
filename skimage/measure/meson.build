extensions = [
  '_ccomp',
  '_find_contours_cy',
  '_marching_cubes_lewiner_cy',
  '_moments_cy',
  '_pnpoly'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/measure'
  )
endforeach

python_sources = [
  '__init__.py',
  '_blur_effect.py',
  '_colocalization.py',
  '_find_contours.py',
  '_label.py',
  '_marching_cubes_lewiner.py',
  '_marching_cubes_lewiner_luts.py',
  '_moments.py',
  '_moments_analytical.py',
  '_polygon.py',
  '_regionprops.py',
  '_regionprops_utils.py',
  'block.py',
  'entropy.py',
  'fit.py',
  'pnpoly.py',
  'profile.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/measure'  # Folder relative to site-packages to install to
)

subdir('tests')
