__all__ = ['python_to_notebook', 'Notebook']

import json
import copy
import warnings


sample = """{
    "metadata": {
    "name":""
    },
    "nbformat": 3,
    "nbformat_minor": 0,
    "worksheets": [
        {
            "cells": [
            {
                "cell_type": "code",
                "collapsed": false,
                "input": [
                    "%matplotlib inline"
                ],
                "language": "python",
                "metadata": {},
                "outputs": []
            }
            ],
          "metadata": {}
        }
    ]
}"""


def _squash_repeats(x):
    """Reduce repeating elements to a single occurrance.

    Parameters
    ----------
    x : list
        Input list.

    Returns
    -------
    list
        A copy of `x` with repeating elements squashed.

    Examples
    --------
    >>> input = [1, 2, 3, 3, 4, 5, 6, 6]
    >>> print _squash_repeats(input)
    [1, 2, 3, 4, 5, 6]

    """
    return [x[0]] + [x[i] for i in range(1, len(x)) if x[i] != x[i-1]]


class Notebook(object):
    """
    Notebook object for building an IPython notebook cell-by-cell.
    """

    def __init__(self):
        # cell type code
        self.cell_code = {
            'cell_type': 'code',
            'collapsed': False,
            'input': [
                '# Code Goes Here'
            ],
            'language': 'python',
            'metadata': {},
            'outputs': []
        }

        # cell type markdown
        self.cell_md = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                'Markdown Goes Here'
            ]
        }

        self.cell_type = {'input': self.cell_code, 'source': self.cell_md}
        self.valuetype_to_celltype = {'code': 'input', 'markdown': 'source'}

    def add_cell(self, value, cell_type='code'):
        """Add a notebook cell.

        Parameters
        ----------
        value : str
            Cell content.
        cell_type : {'code', 'markdown'}
            Type of content (default is 'code').

        """
        if cell_type in ['markdown', 'code']:
            key = self.valuetype_to_celltype[cell_type]
            cells = self.template['worksheets'][0]['cells']
            cells.append(copy.deepcopy(self.cell_type[key]))
            # assign value to the last cell
            cells[-1][key] = value
        else:
            warnings.warn('Ignoring unsupported cell type (%s)' % cell_type)

    def json(self):
        """Return a JSON representation of the notebook.

        Returns
        -------
        str
            JSON notebook.

        """
        return json.dumps(self.template, indent=2)


def python_to_notebook(example_file, notebook_path):
    """Convert a Python file to an IPython notebook.

    Parameters
    ----------
    example_file : str
        Path for source Python file.
    notebook_path : str
        Path for saving the notebook file (includes the filename).

    """
    nb = Notebook()
    with open(example_file, 'r') as pythonfile:
        nb.template = json.loads(sample)
        nb.code = pythonfile.readlines()
        # Add an extra newline at the end,
        # this aids in extraction of text segments
        nb.code.append('\n')

    # Newline separated portions in example file, are sections.
    # Code and markdown written together in such a section are further
    # treated as different segments. Each cell has content from one
    # segment.
    docstring = False
    source = []

    code = _squash_repeats(nb.code)

    for line in code:
        # A linebreak indicates a segment has ended.
        # If the text segment had only comments, ignore the blank source as
        # already added in cell type markdown
        if line == '\n':
            if source:
                # we've found text segments within the docstring
                if docstring:
                    nb.add_cell(source, 'markdown')
                else:
                    nb.add_cell(source, 'code')
                source = []
        # if it's a comment
        elif line.strip().startswith('#'):
            line = line.lstrip(' #')
            nb.add_cell(line, 'markdown')
        elif line == '"""\n':
            if not docstring:
                docstring = True
            # Indicates, completion of docstring
            # add whatever in source to markdown (cell type markdown)
            elif docstring:
                docstring = False
                # Write leftover docstring if any left
                if source:
                    nb.add_cell(source, 'markdown')
                    source = []
        else:
            # some text segment is continuing, so add to source
            source.append(line)

    with open(notebook_path, 'w') as output:
        output.write(nb.json())


def test_foo():
    assert 1==1


if __name__ == "__main__":
    import numpy.testing as npt
    npt.run_module_suite()
