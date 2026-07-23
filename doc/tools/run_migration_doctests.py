#!/usr/bin/env python
"""Script to run migration doctests"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys

from migration_utils import get_doc_dicts, run_doctests


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    return parser


def main():
    parser = get_parser()
    parser.parse_args()  # Check arguments.
    doc_dict, _ = get_doc_dicts()
    success, msg = run_doctests(doc_dict)
    print(msg)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
