extensions = [
  '_mcp',
  '_ncut_cy',
  '_spath',
  'heap'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy, '../_shared'],
    install: true,
    subdir: 'skimage/graph'
  )
endforeach

python_sources = [
  '__init__.py',
  '_graph.py',
  '_graph_cut.py',
  '_graph_merge.py',
  '_ncut.py',
  '_rag.py',
  'mcp.py',
  'spath.py'
]

py3.install_sources(
  python_sources,
  pure: false,             # Will be installed next to binaries
  subdir: 'skimage/graph'  # Folder relative to site-packages to install to
)

subdir('tests')
