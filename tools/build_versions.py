#!/usr/bin/env python

from pathlib import Path
import pkg_resources
import re


def main():
    requirements_dir = Path(__file__).parent / '..' / 'requirements'

    for requirement_file in requirements_dir.glob('*.txt'):
        print(requirement_file.name)
        with open(str(requirement_file), 'r') as f:
            for req in f:
                # Remove trailing and leading whitespace.
                req = req.strip()

                if not req:
                    # skip empty or white space only lines
                    continue
                elif req.startswith('#'):
                    continue

                # Get the name of the package
                req = re.split('<|>|=|!|;', req)[0]
                try:
                    # use pkg_resources to reliably get the version at install
                    # time by package name. pkg_resources needs the name of the
                    # package from pip, and not "import".
                    # e.g. req is 'scikit-learn', not 'sklearn'
                    version = pkg_resources.get_distribution(req).version
                    print(req.rjust(20), version)
                except pkg_resources.DistributionNotFound:
                    print(req.rjust(20), 'is not installed')


if __name__ == '__main__':
    main()
