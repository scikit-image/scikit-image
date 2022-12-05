extensions = [
  '_ncut_cy'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../../_shared'],
    install: true,
    subdir: 'skimage/future/graph'
  )
endforeach

python_sources = [
  '__init__.py',
  '_ncut.py',
  'graph_cut.py',
  'graph_merge.py',
  'rag.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/future/graph'  # Folder relative to site-packages to install to
)

subdir('tests')
