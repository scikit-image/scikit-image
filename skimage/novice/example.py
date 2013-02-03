from skimage import novice            # special submodule for beginners

picture = novice.open('sample.png')   # create a picture object from a file
print picture.format                  # pictures know their format...
print picture.path                    # ...and where they came from...
print picture.size                    # ...and their size
print picture.width                   # 'width' and 'height' also exposed
picture.size = (200, 250)             # changing size automatically resizes
for pixel in picture:                 # can iterate over pixels
    if ((pixel.red > 128) and         # pixels have RGB (values are 0-255)...
        (pixel.x < picture.width)):   # ...and know where they are
        pixel.red /= 2                # pixel is an alias into the picture
   
print picture.modified                # pictures know if their pixels are dirty
print picture.path                    # picture no longer corresponds to file
picture[0:20, 0:20] = (0, 0, 0)       # overwrite lower-left rectangle with black
picture.save('sample-bluegreen.jpg')  # guess file type from suffix
print picture.path                    # picture now corresponds to file
print picture.format                  # ...has a different format
print picture.modified                # ...and is now in sync
