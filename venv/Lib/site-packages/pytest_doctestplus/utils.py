import importlib.util

from importlib.metadata import distribution
from packaging.requirements import Requirement


class ModuleChecker:

    def find_module(self, module):
        """Search for modules specification."""
        try:
            return importlib.util.find_spec(module)
        except ImportError:
            return None

    def find_distribution(self, dist):
        """Search for distribution with specified version (eg 'numpy>=1.15')."""
        try:
            reqs = Requirement(dist)
            dist_meta = distribution(reqs.name)
        except Exception:
            return None
        else:
            if reqs.specifier.contains(dist_meta.version, prereleases=True):
                return dist_meta
            else:
                return None

    def check(self, module):
        """
        Return True if module with specified version exists.
        >>> ModuleChecker().check('foo>=1.0')
        False
        >>> ModuleChecker().check('pytest>1.0')
        True
        """
        mods = self.find_module(module) or self.find_distribution(module)
        return bool(mods)
