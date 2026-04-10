#!/usr/bin/env python
"""Script to build migration guide from docstrings"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from os import environ
from pathlib import Path
import sys

from jinja2 import Template


class TrackerDict(dict):
    """Dict that keeps check on keys that have been accessed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.not_accessed_keys = set(self)

    def __getitem__(self, key):
        self.not_accessed_keys.discard(key)
        return super().__getitem__(key)


def write_migration(in_tpl, doc_dict, out_path=None):
    in_tpl = Path(in_tpl)
    out_path = Path(out_path) if out_path else in_tpl.with_suffix('.md')
    tpl = Template(in_tpl.read_text())
    render_dict = TrackerDict(doc_dict)
    out_str = tpl.render(d=render_dict)
    # Check all keys have been consumed.
    if render_dict.not_accessed_keys:
        raise RuntimeError(
            'These keys not used in template:'
            + ', '.join(render_dict.not_accessed_keys)
        )
    Path(out_path).write_text(out_str)


def run_doctests(doctests):
    if not doctests:
        return 'No doctests found'
    msgs = []
    success = True
    for func_name, tests in doctests.items():
        sep = '-' * 10 + '\n'
        msgs.append(f'Running tests for `{func_name}`')
        for i, test in enumerate(tests):
            context = {}
            title = f'Test {i}'
            try:
                exec(test, locals=context)
            except Exception as e:
                msgs.append(f'{title} ... failed\n{sep}{test}\n{sep}'
                            f'with traceback: {e}\n{sep}')
                success = False
            else:
                msgs.append(f'{title} ... passed')
    return success, '\n'.join(msgs)


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument('migration_tpl', help='Path to migration template file')
    parser.add_argument('--out-md', help='Path to output markdown file')
    parser.add_argument('--doctest', action='store_true',
                        help='Run discovered doctests')
    return parser


def get_doc_dicts():
    environ['EAGER_IMPORT'] = "true"  # Turn off lazy-loading
    import skimage as ski  # noqa: F401
    from _skimage2.util.migration import ski2_migration_dec as sk2md

    return sk2md.migration_docs, sk2md.doctests


def main():
    parser = get_parser()
    args = parser.parse_args()
    doc_dict, doctests = get_doc_dicts()
    write_migration(args.migration_tpl, doc_dict, args.out_md)
    if args.doctest:
        success, msg = run_doctests(doctests)
        print(msg)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
