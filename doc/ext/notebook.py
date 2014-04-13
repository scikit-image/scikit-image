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


class Notebook():
    """
    Notebook object for generating an IPython notebook
    from an example Python file.
    """

    def __init__(self, sample_notebook_path, example_file):
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
        self.keys = {'input_code': 'input', 'input_markdown': 'source'}
        with open(example_file, 'r') as pythonfile:
            self.template = json.loads(sample)
            self.code = pythonfile.readlines()
            # Adds an extra newline at the end,
            # this aids in extraction of text segments
            self.code.append('\n')

    def fetchkey(self, type_of_value):
        """ Returns the key required for insertion into notebook,
        based on the type of value.
        """
        return self.keys[type_of_value]

    def filter_continuous_duplication(self):
        """ Clusters multiple '\n's into one.
        For ex - 'import xyz\n\n\n print 2' becomes 'import xyz\n print 2' """
        modified_code = []
        modified_code = [self.code[i] for i in range(len(self.code)) if i == 0 or self.code[i] != self.code[i-1]]
        return modified_code

    def addcell(self, segment_number, value, type_of_value='input_code'):
        """ Adds a notebook cell, by updating the json template.
        Cell differs with type of value.
        """
        if type_of_value in ['input_markdown', 'input_code']:
            key = self.fetchkey(type_of_value)
            self.template["worksheets"][0]["cells"].append(copy.deepcopy(self.cell_type[key]))
            self.template["worksheets"][0]["cells"][segment_number][key] = value

    def json(self, notebook_path):
        """ Writes the template to file (json) """
        with open(notebook_path, 'w') as output:
            json.dump(self.template, output, indent=2)


def python_to_notebook(example_file, notebook_dir, notebook_path):
    """ Convert a Python file to an IPython notebook.

    Parameters
    ----------
    example_file : 'str'
        path for source Python file
    notebook_dir : 'str'
        directory for saving the notebook files
    notebook_path : 'str'
        path for saving the notebook file (includes the filename)
    """
    sample_notebook_path = notebook_dir.pjoin('sample.ipynb')

    nb = Notebook(sample_notebook_path, example_file)

    segment_number = 0
    segment_has_begun = True
    docstring = False
    source = []

    modified_code = nb.filter_continuous_duplication()

    for line in modified_code:
        # A linebreak indicates a segment has ended.
        # If the text segment had only comments, then source is blank,
        # So, ignore it, as already added in cell type markdown
        if line == "\n":
            if segment_has_begun is True and source:
                segment_number += 1
                # we've found text segments within the docstring
                if docstring is True:
                    nb.addcell(segment_number, source, 'input_markdown')
                else:
                    nb.addcell(segment_number, source, 'input_code')
                source = []
        # if it's a comment
        elif line.strip().startswith('#'):
            segment_number += 1
            line = line.strip(' #')
            nb.addcell(segment_number, line, 'input_markdown')
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
                    nb.addcell(segment_number, source, 'input_markdown')
                    source = []
        else:
            # some text segment is continuing, so add to source
            source.append(line)

    nb.json(notebook_path)
