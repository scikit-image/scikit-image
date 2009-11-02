import plugin

try:
    import matplotlib.pyplot as plt
except ImportError, e:
    print e
else:
    plugin.register('matplotlib', show=plt.imshow, save=plt.imsave)
