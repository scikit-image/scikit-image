#!/usr/bin/env python
"""Script to build migration guide from docstrings

The script harvests the migration docstrings from the scikit-image source, and
writes the migration guide ReST page, using the given Jinja2 template.
"""

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from jinja2 import Template

from migration_utils import get_doc_dicts, run_doctests


def write_migration(in_tpl, doc_dict, out_path=None):
    in_tpl = Path(in_tpl)
    if out_path is None:
        out_path = in_tpl.with_name(in_tpl.name.replace('.tpl', ''))
    out_path = Path(out_path)
    tpl = Template(in_tpl.read_text())
    render_dict = doc_dict.copy()
    out_str = tpl.render(advice_map=render_dict)
    # Check all keys have been consumed.
    if render_dict:
        raise RuntimeError('These keys not used in template:' + ', '.join(render_dict))
    Path(out_path).write_text(out_str)
    return out_path


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
    out_path = write_migration(
        args.migration_tpl, {**doc_dict, **extra_dict}, args.out_rst
    )
    print(f"Wrote to '{out_path}'")
    if args.doctest:
        success, msg = run_doctests(doc_dict)
        print(msg)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
