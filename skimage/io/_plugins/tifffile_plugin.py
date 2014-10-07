try:
    from tifffile import imread, imsave
except ImportError:
    raise ImportError("The tifffile module could not be found.\n"
        "It can be obtained at "
        "<http://www.lfd.uci.edu/~gohlke/code/tifffile.py>\n")
