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
        'skimage.io',
    ]
    param_names = ["package_name"]

    def setup(self, package_name):
        pass

    def time_import(self, package_name):
        run(
            executable + ' -c "import ' + package_name + '"',
            capture_output=True,
            stdin=PIPE,
            shell=True,
        )
