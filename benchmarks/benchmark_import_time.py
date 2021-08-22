from subprocess import run, PIPE
from sys import executable

class ImportSuite:
    """Benchmark the time it takes to import various modules"""
    params = [
        'numpy',
        'skimage',
        'skimage.feature',
        'skimage.morphology',
        'skimage.color',
    ]
    param_names = ["package_name"]
    def setup(self, package_name):
        pass

    def time_import(self, package_name):
        results = run(executable + ' -c "import ' + package_name + '"',
            stdout=PIPE, stderr=PIPE, stdin=PIPE, shell=True)

