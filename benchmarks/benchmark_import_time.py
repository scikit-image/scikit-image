from subprocess import run, PIPE
from sys import executable

class ImportSuite:
    """Benchmark the time it takes to import various modules"""
    def setup(self):
        pass

    def time_import_numpy(self):
        results = run(executable + ' -c "import numpy"',
            stdout=PIPE, stderr=PIPE, stdin=PIPE, shell=True)

    def time_import_skimage(self):
        results = run(executable + ' -c "import skimage"',
            stdout=PIPE, stderr=PIPE, stdin=PIPE, shell=True)

