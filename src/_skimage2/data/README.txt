This is scikit-image's download cache for its example datasets.

Files are downloaded here on demand, the first time a function like
`skimage.data.astronaut()` is called. The optional `scikit-image-data`
package (installed via `pip install scikit-image[data]`) bundles a curated
subset of commonly-used datasets directly in its own install location, so
those specific files are never downloaded at all.

To download every dataset ahead of time, from a Python console:

  >>> from _skimage2.data import download_all
  >>> download_all()
