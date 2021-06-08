Data visualization
------------------

Data visualization takes an important place in image processing. Data can be
a simple unique 2D image or a more complex with multidimensional aspects: 3D
in space, timeslapse, multiple channels.

Therefore, the visualization strategy will depend on the data complexity and
a range of tools external to scikit-image can be used for this purpose.
Historically, scikit-image provided viewer tools but powerful packages
are now available and must be preferred.


Matplotlib
^^^^^^^^^^

`Matplotlib <https://matplotlib.org/>`__ is a library able to generate static
plots, which includes image visualization.

Plotly
^^^^^^

`Plotly <https://dash.plotly.com/>`__ is a plotting library relying on web
technologies with interaction capabilities.

Mayavi
^^^^^^

`Mayavi <https://docs.enthought.com/mayavi/mayavi/>`__ can be used to visualize
3D images.

Napari
^^^^^^

`Napari <https://napari.org/>`__ is a multi-dimensional image viewer. It’s
designed for browsing, annotating, and analyzing large multi-dimensional images.
