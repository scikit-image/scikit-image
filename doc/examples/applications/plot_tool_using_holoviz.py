"""
=======================================
Explore and annotate image data interactively with with HoloViz
=======================================

This tutorial shows how to build interactive dashboards and applications

using `Panel <https://panel.holoviz.org>`_ and `HoloViews <https://holoviews.org>`_.
These tools can make your analyses more interactive and, hence, powerful.

.. figure:: https://user-images.githubusercontent.com/42288570/146666639-02f0106f-a3e2-4306-9033-707554e4e61e.gif
   :alt: Screencast showcasing the example application.

   Screencast showcasing the example application.


You can develop using notebooks or .py files. You can use it with a very
broad range of Pythons Viz. You can use the tools in a notebook, your editor/ IDE or
deploy to a web server.

To run this example you would need to get some python packages
installed. You can create an environment from scratch using.

.. code:: bash

   conda create -n scikit-image-demo -c conda-forge jupyterlab panel holoviews scikit-image selenium \
   firefox geckodriver
   conda activate scikit-image-demo
   jupyter serverextension enable panel.io.jupyter_server_extension
"""


############################
# Import packages
# ==========================


from io import BytesIO

import holoviews as hv
import panel as pn
from skimage import data, filters

#############################################
# Activate the Panel and HoloViews extensions
# ==================================


pn.extension(sizing_mode="fixed")
hv.extension("bokeh")


##############################
# Start With an Image of Coins

image = data.coins()
bounds = (-1, -1, 1, 1)
height, width = image.shape
aspect_ratio = float(width) / float(height)

height = 400
width = int(height * aspect_ratio)

hv.Image(image, bounds=bounds).opts(
    height=height,
    width=width,
    cmap="binary_r",
    title="Before",
    active_tools=["box_zoom"],
    tools=["hover"],
)

##########################################
# Find the Edges Using the Sobel Algorithm

edges = filters.sobel(image) * 256

hv.Image(edges, bounds=bounds).opts(
    height=400,
    width=width,
    cmap="binary_r",
    title="After",
    active_tools=["box_zoom"],
    tools=["hover"],
)


#############################################################
# We Make an Interactive Tool to Try Out Different Color Maps
# ===========================================================
#
# First we create the *color map selection* widget.

cmaps = [cmap for cmap in hv.plotting.list_cmaps() if (cmap.endswith("_r") and cmap.islower())]
cmap = pn.widgets.Select(
    value="binary_r", options=cmaps, width=290, height=50, sizing_mode="fixed", name="Color Map"
)
cmap

###########################################
# Then we define the two interactive images

before_img = hv.Image(image, bounds=bounds).apply.opts(
    cmap=cmap, responsive=True, title="Before", active_tools=["box_zoom"], tools=["hover"]
)
after_img = hv.Image(edges, bounds=bounds).apply.opts(
    cmap=cmap,
    responsive=True,
    title="After",
    active_tools=["box_zoom"],
    tools=["hover"],
)


############################
# Finally we layout the tool

plots = pn.panel((before_img + after_img), sizing_mode="scale_both", aspect_ratio=aspect_ratio * 2)
layout = pn.Column(cmap, plots, sizing_mode="stretch_both")
layout

#################################
# We Make a Download Image Button
# ===============================

download = pn.widgets.FileDownload(
    filename="after.png", label="Download .png", button_type="success", sizing_mode="stretch_width"
)
download

####################################
# and define the *download callback*


def callback():
    download.loading = True

    height = 800
    width = int(aspect_ratio * height)
    plot = hv.Image(edges, bounds=bounds).opts(
        cmap=cmap.value, height=height, width=width, xaxis=None, yaxis=None
    )

    bytesio = BytesIO()
    hv.save(plot, bytesio, fmt="png")
    bytesio.seek(0)

    download.loading = False
    return bytesio


download.callback = callback


##################################
# and add the button to the layout

layout.append(download)


####################################
# We Wrap it Up as A Web Application
# ==================================
# We include the value of the color map in the url to make it easy to share a link with colleagues

###########################################################
# We sync the browser url arguments to the color map widget

if pn.state.location:
    pn.state.location.sync(cmap, {"value": "cmap"})

########################################
# We wrap it in a nicely styled template

pn.template.FastListTemplate(
    site="Scikit-Image, Panel and HoloViews",
    title="Finding Edges with the Sobel Filter",
    main=[layout],
    header_background="#292929",
).servable()

########################################
# You can serve the app using `panel serve ScikitImageSobelApp.ipynb`
# and the app will be available at
# http://localhost:5006/ScikitImageSobelApp.
# Add the ``--autoreload`` flag for hot reloading during development. You
# can also install the jupyterlab preview for hot reload inside Jupyter
# Lab during development. See
# https://blog.holoviz.org/panel_0.12.0.html#JupyterLab-previews
# 
# For more inspiration checkout `Panel <https://panel.holoviz.org>`_ and 
# `awesome-panel.org <https://awesome-panel.org>`_