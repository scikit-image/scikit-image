__all__ = ['python_to_notebook', 'Notebook']

import json
import copy

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


def _remove_consecutive_duplicates(inp):
    """Remove duplicates of elements appearing consecutively.

    Parameters
    ----------
    inp : list
        Input list.

    Returns
    -------
    modified_inp : list
        Output list, with no consecutive duplicates.

    Examples
    --------
    >>> input = [1, 2, 3, 3, 4, 5, 6, 6]
    >>> output = remove_consecutive_duplicates(input)
    >>> output
    [1, 2, 3, 4, 5, 6]

    """
    modified_inp = [inp[0]] + [inp[i] for i in range(1, len(inp)) if inp[i] != inp[i-1]]
    return modified_inp


class Notebook():
    """
    Notebook object for generating an IPython notebook from an example Python
    file.

    Parameters
    ----------
    example_file : str
        Path for example file.

    """

    def __init__(self, example_file):
        # Object variables, gives the ability to personalise per object
        # cell type code
        self.cell_code = {
            "cell_type": "code",
            "collapsed": False,
            "input": [
                "# Code Goes Here"
            ],
            "language": "python",
            "metadata": {},
            "outputs": []
        }

        # cell type markdown
        self.cell_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                'Markdown Goes Here'
            ]
        }

        self.cell_type = {'input': self.cell_code, 'source': self.cell_md}
        self.valuetype_to_celltype = {'code': 'input', 'markdown': 'source'}
        with open(example_file, 'r') as pythonfile:
            self.template = json.loads(sample)
            self.code = pythonfile.readlines()
            # Adds an extra newline at the end,
            # this aids in extraction of text segments
            self.code.append('\n')

    def add_cell(self, value, type_of_value='code'):
        """Adds a notebook cell.

        Parameters
        ----------
        value : str
            The actual content to be saved in the cell.
        type_of_value : {'code', 'markdown'}
            The type of content.
            The default value will add a cell of type code.

        """
        if type_of_value in ['markdown', 'code']:
            key = self.valuetype_to_celltype[type_of_value]
            self.template["worksheets"][0]["cells"].append(copy.deepcopy(self.cell_type[key]))
            # assign value to the last cell
            self.template["worksheets"][0]["cells"][-1][key] = value

    def json(self):
        """Dumps the template JSON to string.

        Returns
        -------
        str
            The template JSON converted to a string with a two char indent.

        """
        return json.dumps(self.template, indent=2)


def python_to_notebook(example_file, notebook_dir, notebook_path):
    """Convert a Python file to an IPython notebook.

    Parameters
    ----------
    example_file : str
        Path for source Python file.
    notebook_dir : str
        Directory for saving the notebook files.
    notebook_path : str
        Path for saving the notebook file (includes the filename).

    """
    nb = Notebook(example_file)

    # Newline separated portions in example file, are sections.
    # Code and markdown written together in such a section are further
    # treated as different segments. Each cell has content from one
    # segment.
    segment_has_begun = True
    docstring = False
    source = []

    modified_code = _remove_consecutive_duplicates(nb.code)

    for line in modified_code:
        # A linebreak indicates a segment has ended.
        # If the text segment had only comments, then source is blank,
        # So, ignore it, as already added in cell type markdown
        if line == "\n":
            if segment_has_begun is True and source:
                # we've found text segments within the docstring
                if docstring:
                    nb.add_cell(source, 'markdown')
                else:
                    nb.add_cell(source, 'code')
                source = []
        # if it's a comment
        elif line.strip().startswith('#'):
            line = line.strip(' #')
            nb.add_cell(line, 'markdown')
        elif line == '"""\n':
            if docstring is False:
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
