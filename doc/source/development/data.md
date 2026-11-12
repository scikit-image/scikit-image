# Adding Data

While code is hosted on [GitHub](https://github.com/scikit-image/),
example datasets are on [GitLab](https://gitlab.com/scikit-image/data).
These are fetched with [pooch](https://github.com/fatiando/pooch)
when accessing `skimage.data.*`.

New datasets are submitted on GitLab and, once merged, the data
registry `skimage/data/_registry.py` in the main GitHub repository
can be updated.
