#!/usr/bin/env python
"""Script to build migration guide from docstrings"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import sys


from migration_utils import get_doc_dicts, write_migration, run_doctests


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument('migration_tpl', help='Path to migration template file')
    parser.add_argument('--out-rst', help='Path to output ReST file')
    parser.add_argument(
        '--doctest', action='store_true', help='Run discovered doctests'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    doc_dict, extra_dict = get_doc_dicts()
    write_migration(args.migration_tpl, {**doc_dict, **extra_dict}, args.out_rst)
    if args.doctest:
        success, msg = run_doctests(doc_dict)
        print(msg)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
