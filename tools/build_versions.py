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
                req = req.strip()
                if req[0] == '#':
                    continue
                req = re.split('<|>|=|!|;', req)[0]
                try:
                    version = pkg_resources.get_distribution(req).version
                    print(req.rjust(20), version)
                except pkg_resources.DistributionNotFound:
                    print(req.rjust(20), 'is not installed')


if __name__ == '__main__':
    main()
