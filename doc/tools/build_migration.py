#!/usr/bin/env python
"""Script to build migration guide from docstrings"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import doctest
from functools import partial
from os import environ
from pathlib import Path
import sys

from jinja2 import Template

PARSER = doctest.DocTestParser()
RUNNER = doctest.DocTestRunner()


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
    if out_path is None:
        out_path = in_tpl.with_name(in_tpl.name.replace('.tpl', ''))
    out_path = Path(out_path)
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


def _append_msgs(msg_str, messages=[]):
    messages.append(msg_str)


def run_doctest(func_name, doc):
    test = PARSER.get_doctest(doc, {}, func_name, "(unknown)", 0)
    out_msgs = []
    result = RUNNER.run(test, out=partial(_append_msgs, messages=out_msgs))
    return (
        result.failed,
        f"Attempted: {result.attempted}\nFailed: {result.failed}"
        + ('\n' + out_msgs[0] if result.failed else ''),
    )


def run_doctests(doc_dict):
    msgs = []
    success = True
    for func_name, doc in doc_dict.items():
        header = f'Running tests for `{func_name}`'
        lines = '-' * len(header)
        msgs.append(f'{lines}\n{header}\n{lines}')
        n_failed, msg = run_doctest(func_name, doc)
        msgs.append(msg.strip())
        success = success and n_failed == 0
    return success, '\n'.join(msgs)


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


def get_doc_dicts():
    environ['EAGER_IMPORT'] = "true"  # Turn off lazy-loading
    import skimage as ski  # noqa: F401
    from skimage._migration import ski2_migration_decorator as sk2md

    return sk2md.migration_docs, sk2md.extra_params


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
