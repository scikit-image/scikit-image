This is scikit-image's download cache for its example datasets.

Files are downloaded here on demand, the first time a function like
`skimage.data.kidney()` is called. Commonly-used datasets (e.g.
`skimage.data.astronaut()`) are instead bundled directly by the
`scikit-image-data` package, in its own install location, so those
specific files are never downloaded at all.

To download every dataset ahead of time, from a Python console:

  >>> from _skimage2.data import download_all
  >>> download_all()
