from skimage import io, transform

s = 0.7

img = io.imread('scikit_image_logo.png')
h, w, c = img.shape

print "\nScaling down logo by %.1fx..." % s

img = transform.homography(img, [[s, 0, 0],
                                 [0, s, 0],
                                 [0, 0, 1]],
                           output_shape=(int(h*s), int(w*s), 4),
                           order=3)

io.imsave('scikit_image_logo_small.png', img)
