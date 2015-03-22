import numpy as np
"""
Haar Features
It returns the specific haar feature, evaluated in a 24*24 frame,
as a single array.

Input
-----
i : integral image
x : the x-co-ordinate
y : the y-co-ordinate
f : feature type
s : scale factor

Output
------
haar_features = computed haar value

References
----------
Viola, Jones: Robust Real-time Object Detection, IJCV 2001 See pages 1,3.

Example
-------
>>> import numpy as np
>>> from skimage import data
>>> from skimage.transform import integral_image
>>> from skimage.transform import haar_feature
>>> image=data.coins()
>>> im=integral_image(image)
>>> hf=haar_feature(im, 34, 45, 1, 1)

"""


def haar_feature(i, x, y, f, s):
    features = np.array([[2, 1], [1, 2], [3, 1], [1, 3], [2, 2]])
    h = features[f][0]*s
    w = features[f][1]*s
    if f == 0:
        bright = (i[x+h/2-1, y+w-1] + i[x-1, y-1]) - \
            (i[x-1, y+w-1] + i[x+h/2-1, y-1])
        dark = (i[x+h-1, y+w-1] + i[x+h/2-1, y-1]) - \
            (i[x+h/2-1, y+w-1] + i[x+h-1, y-1])
        
    elif f == 1:
        bright = (i[x+h-1, y+w/2-1] + i[x-1, y-1]) - \
            (i[x-1, y+w/2-1] + i[x+h-1, y-1])
        dark = (i[x+h-1, y+w-1] + i[x-1, y+w/2-1]) - \
            (i[x+h-1, y+w/2-1] + i[x-1, y+w-1])
    haar_feature = bright-dark
    return haar_feature
