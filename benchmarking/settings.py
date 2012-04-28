"""
Test suite definitions for vbench.

Global variables are written assuming that this test suite is one directory
above the main package (e.g. /path/to/skimage/vbench/)
"""
import os
import sys

from vbench.api import Benchmark, GitRepo
from datetime import datetime


__all__ = ['REPO_URL', 'VBENCH_PATH', 'REPO_PATH', 'DB_PATH', 'TMP_DIR',
           'SUITE_PATH', 'START_DATE', 'PREPARE', 'BUILD', 'dependencies',
           'benchmarks', 'by_module']


REPO_URL = 'git@github.com:scikits-image/scikits-image.git'
VBENCH_PATH = os.path.abspath(os.path.dirname(__file__))
REPO_PATH = os.path.dirname(VBENCH_PATH)
DB_PATH = os.path.join(VBENCH_PATH, 'benchmarks.db')
TMP_DIR = os.path.join(VBENCH_PATH, 'tmp/skimage')
SUITE_PATH = os.path.join(VBENCH_PATH, 'suite')

by_module = {}
benchmarks = []

sys.path.append(SUITE_PATH) # required for __import__
_suite = [m.rstrip('.py') for m in os.listdir(SUITE_PATH) if m.endswith('.py')]
for modname in _suite:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

for bm in benchmarks:
    assert(bm.name is not None)
sys.path.remove(SUITE_PATH)

# TODO: Cleaning out the repo may not be necessary
#       (Strange: removing clean up doesn't seem to change the run time)
PREPARE = """
python setup.py clean
"""

BUILD = """
python setup.py build_ext --inplace
"""

# These modules are available for import in any benchmarking module
dependencies = ['vb_common.py']

START_DATE = datetime(2011, 10, 26) # date Cython MD5 hash was implemented
# START_DATE = datetime(2012, 4, 23) # use recent date for testing

repo = GitRepo(REPO_PATH)

