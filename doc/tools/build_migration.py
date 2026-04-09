#!/usr/bin/env python
"""Script to build migration guide from docstrings
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from os import environ
from pathlib import Path

from jinja2 import Template


class TrackerDict(dict):
    """ Dict that keeps check on keys that have been accessed.
    """

    def __init__(self, in_dict):
        self.remaining_keys = set(in_dict)
        super().__init__(in_dict)

    def __getitem__(self, key):
        self.remaining_keys.discard(key)
        return super().__getitem__(key)


def write_migration(in_tpl, doc_dict, out_path=None):
    in_tpl = Path(in_tpl)
    out_path = Path(out_path) if out_path else in_tpl.with_suffix('.md')
    tpl = Template(in_tpl.read_text())
    render_dict = TrackerDict(doc_dict)
    out_str = tpl.render(**render_dict)
    # Check all keys have been consumed.
    if render_dict.remaining_keys:
        raise RuntimeError('These keys not used in template:' +
                           ', '.join(render_dict.remaining_keys))
    Path(out_path.write_text(out_str))


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('migration_tpl',
                        help='Path to migration template file')
    parser.add_argument('--out-md',
                        help='Path to output markdown file')
    return parser


def get_doc_dict():
    environ['EAGER_IMPORT'] = "true"  # Turn off lazy-loading
    import skimage as ski
    from _skimage2.util.migration import ski2_migration_dec
    return ski2_migration_dec.migration_docs


def main():
    parser = get_parser()
    args = parser.parse_args()
    doc_dict = get_doc_dict()
    write_migration(args.migration_tpl, doc_dict, args.out_md)


if __name__ == "__main__":
    main()
