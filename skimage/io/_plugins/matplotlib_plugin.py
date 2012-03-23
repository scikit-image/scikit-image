import matplotlib.pyplot as plt


def imshow(image, fancy=False, **kwargs):
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    if fancy:
        from ... import viewer
        return viewer.ImageViewer(image, **kwargs)
    else:
        plt.imshow(image, **kwargs)


imread = plt.imread
show = plt.show

def _app_show():
    show()

