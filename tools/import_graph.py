import numpy as np
import sys
import importlib

modules = ['_shared', 'color', 'data', 'draw', 'exposure', 'feature', 'filters',
           'future', 'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
           'restoration', 'segmentation', 'transform', 'util']

# This is very weird, yes. I'm using a `npz` file to save index of the
# module we need to import next as well as the partial state of the import
# graph. `import_graph_caller` calls this repeatedly until we run out of
# indices. Obviously there are more elegant ways to do this.
try:
    res = np.load('import_graph.npz')
    A = res['A']
    k = res['k']
except:
    n = len(modules)
    A = np.zeros((n, n), dtype=int)
    k = 0

modules = ['skimage.' + module for module in modules]
module_numbers = {mod: k for k, mod in enumerate(modules)}

# Import the module and associated tests
importlib.import_module(modules[k])
importlib.import_module(modules[k] + ".tests")

# See which other modules ended up in `sys.modules`
skimage_subpackages = []
skimage_modules = [sys_module for sys_module in sys.modules
                   if 'skimage' in sys_module]
for module in modules:
    for skimage_module in skimage_modules:
        if module in skimage_modules:
            skimage_subpackages.append(module)
            break

# A simpler way of determining which modules are in `sys.modules`
# (sanity check)
skimage_subpackages2 = [sys_module for sys_module in sys.modules
                        if 'skimage' in sys_module and sys_module.count('.') == 1]
assert not (set(skimage_subpackages) - set(skimage_subpackages2))

# Encode which modules are in `sys.modules` in an adjacency matrix.
# Each row is the module we imported; each column is 1 if the corresponding
# module ended up in `sys.modules`.
for package in skimage_subpackages:
    A[k, module_numbers[package]] = 1
k += 1
np.savez('import_graph.npz', A=A, k=k)
