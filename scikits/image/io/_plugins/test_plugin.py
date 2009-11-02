import plugin

def save(fname, arr):
    return fname, arr

plugin.register('test', save=save)
