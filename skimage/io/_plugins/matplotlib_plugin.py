import matplotlib.pyplot as plt


def imshow(*args, **kwargs):
    if plt.gca().has_data():
        plt.figure()
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')
    plt.imshow(*args, **kwargs)

imread = plt.imread
show = plt.show


def _app_show():
    show()
