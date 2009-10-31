import plugin

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    plugin.register('matplotlib', show=plt.imshow, save=plt.imsave)
