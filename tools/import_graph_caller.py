# This script prints the modules for which tests need to run
# given `git diff main` of a feature branch.
import numpy as np
import subprocess
from time import perf_counter
# All the modules we're considering. We might want to have separate logic for
# `tools`, etc.
modules = ['_shared', 'color', 'data', 'draw', 'exposure', 'feature', 'filters',
           'future', 'graph', 'io', 'measure', 'metrics', 'morphology', 'registration',
           'restoration', 'segmentation', 'transform', 'util']
modules = ['skimage/' + module for module in modules]
module_numbers = {mod: k for k, mod in enumerate(modules)}  # index in `A` below
n = len(modules)

tic = perf_counter()
while True:
    # Each time this runs, `import_graph.py` imports a different module
    # and records which other modules ended up in `sys.modules`.
    res = subprocess.run(['python', 'import_graph.py'])
    if res.returncode:
        break
toc = perf_counter()
print(toc-tic)
# It stores the information as a matrix in this file
res = np.load('import_graph.npz')
A = res['A']

# Now look for module names in output of `git diff main`
git_diff = subprocess.run('git diff main', capture_output=True,
                          text=True, shell=True).stdout
changed_modules = []
for module in modules:
    if module in git_diff:
        changed_modules.append(module)
        break

# For each module that changed, determine the modules for which we would
# need to run tests.
b = np.zeros(n, dtype=int)
for module in changed_modules:
    i = module_numbers[module]
    b |= A[:, i]

# Print the names of those modules
for i, j in enumerate(b):
    if j:
        print(modules[i])