#!/usr/bin/env python

from __future__ import print_function
import pkg_resources
import re
from os.path import dirname, abspath, basename
from glob import glob


def main():
    requirements_dir = dirname(abspath(__file__)) + '/../requirements'
    for requirement_file in glob(requirements_dir + '/*.txt'):
        print(basename(requirement_file))
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
