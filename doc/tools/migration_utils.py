"""Utilities for migration CLI tools"""

import doctest
from functools import partial
from os import environ

PARSER = doctest.DocTestParser()
RUNNER = doctest.DocTestRunner()


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


def get_doc_dicts():
    environ['EAGER_IMPORT'] = "true"  # Force full importing.
    import skimage as ski  # noqa: F401
    from skimage._migration import ski2_migration_decorator as sk2md

    return sk2md.migration_docs, sk2md.extra_params
