__all__ = ['python_to_notebook']

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


def remove_continuous_duplicates(code):
    """ Remove duplicates of elements appearing consecutively.

    Parameters
    ----------
    code : list of str

    Returns
    -------
    modified_code : list of str

    Notes
    -----
    We create a new list and add elements to it which do not have
    duplicates appearing consecutively.
    One use case here,
    'import xyz\n\n\n print 2' becomes 'import xyz\n print 2'

    """
    modified_code = []
    modified_code = [self.code[i] for i in range(len(self.code)) if i == 0 or self.code[i] != self.code[i-1]]
    return modified_code


class Notebook():
    """
    Notebook object for generating an IPython notebook,
    from an example Python file.

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

    def fetch_key(self, type_of_value):
        """ Find the key required for insertion into notebook.

        Parameters
        ----------
        type_of_value : str
            Type of data, to be inserted in a cell.

        Returns
        -------
        str
            Key which reflects what is the cell type.

        Notes
        -----
        type_of_value is either, 'code', which maps to cell of type
        code('input') or 'markdown', which maps to cell of type markdown('source').

        """
        return self.valuetype_to_celltype[type_of_value]

    def add_cell(self, segment_number, value, type_of_value='code'):
        """ Adds a notebook cell.

        Parameters
        ----------
        segment_number : int
            Newline separated sections in example file, are segments.
            Code and markdown written together in such a section are,
            treated as different segments. Each cell has content from
            one section.
        value : str
            The actual content to be saved in the cell.
        type_of_value : str, optional
            The type of content in the segment.
            The default value will add a cell of type code.

        Notes
        -----
        The cell is only added in the notebook if the segment is of type,
        markdown or code.

        """
        if type_of_value in ['markdown', 'code']:
            key = self.fetch_key(type_of_value)
            self.template["worksheets"][0]["cells"].append(copy.deepcopy(self.cell_type[key]))
            self.template["worksheets"][0]["cells"][segment_number][key] = value

    def json(self):
        """ Dumps the template JSON to string.

        Returns
        -------
        str
            The template JSON converted to a string with a two char indent.

        """
        return json.dump(self.template, indent=2)


def python_to_notebook(example_file, notebook_dir, notebook_path):
    """ Convert a Python file to an IPython notebook.

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

    segment_number = 0
    segment_has_begun = True
    docstring = False
    source = []

    modified_code = remove_continuous_duplicates(nb.code)

    for line in modified_code:
        # A linebreak indicates a segment has ended.
        # If the text segment had only comments, then source is blank,
        # So, ignore it, as already added in cell type markdown
        if line == "\n":
            if segment_has_begun is True and source:
                segment_number += 1
                # we've found text segments within the docstring
                if docstring is True:
                    nb.add_cell(segment_number, source, 'markdown')
                else:
                    nb.add_cell(segment_number, source, 'code')
                source = []
        # if it's a comment
        elif line.strip().startswith('#'):
            segment_number += 1
            line = line.strip(' #')
            nb.add_cell(segment_number, line, 'markdown')
        elif line == '"""\n':
            if docstring is False:
                docstring = True
            # Indicates, completion of docstring,
            # add whatever in source to markdown (cell type markdown)
            elif docstring is True:
                docstring = False
                # Write leftover docstring if any left
                if source:
                    segment_number += 1
                    nb.add_cell(segment_number, source, 'markdown')
                    source = []
        else:
            # some text segment is continuing, so add to source
            source.append(line)

    with open(notebook_path, 'w') as output:
        output.write(nb.json(notebook_path))
