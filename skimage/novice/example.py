from skimage.novice import picture  # special submodule for beginners

pic = picture.open('sample.png')    # create a picture object from a file
print pic.format                    # pictures know their format...
print pic.path                      # ...and where they came from...
print pic.size                      # ...and their size
print pic.width                     # 'width' and 'height' also exposed
pic.size = (200, 250)               # changing size automatically resizes
for pixel in pic:                   # can iterate over pixels
    if ((pixel.red > 128) and           # pixels have RGB (values are 0-255)...
        (pixel.x < pic.width)):     # ...and know where they are
        pixel.red /= 2                  # pixel is an alias into the pic
   
print pic.modified                  # pictures know if their pixels are dirty
print pic.path                      # picture no longer corresponds to file
pic[0:20, 0:20] = (0, 0, 0)         # overwrite lower-left rectangle with black
pic.save('sample-bluegreen.jpg')    # guess file type from suffix
print pic.path                      # picture now corresponds to file
print pic.format                    # ...has a different format
print pic.modified                  # ...and is now in sync
