#!/usr/bin/env python

from pathlib import Path
import requirements
import pkg_resources


def main():
    requirements_dir = Path(__file__).parent / '..' / 'requirements'

    for requirement_file in requirements_dir.glob('*.txt'):
        print(requirement_file.name)
        with open(str(requirement_file), 'r') as fd:
            for req in requirements.parse(fd):
                try:
                    version = pkg_resources.get_distribution(req.name).version
                    print(req.name.rjust(20), version)
                except pkg_resources.DistributionNotFound:
                    print(req.name.rjust(20), 'is not installed')


if __name__ == '__main__':
    main()
