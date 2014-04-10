import json
import copy

class Notebook():
    """Notebook object for generating an IPython notebook from an example file"""


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

        self.cell_type = {'input':self.cell_code, 'source': self.cell_md}
        with open(sample_notebook_path, 'r') as sample, open(example_file, 'r') as pythonfile:
            self.template = json.load(sample)
            self.code = pythonfile.readlines()
            # Adds an extra newline at the end, which aids in extraction of text segments
            self.code.append('\n')

    def getModifiedCode(self):
        """ Clusters multiple '\n's into one.
        For ex - 'import xyz\n\n\n print 2' becomes 'import xyz\n print 2' """
        modified_code = []
        modified_code = [self.code[i] for i in range(len(self.code)) if i==0 or self.code[i]!=self.code[i-1]]
        return modified_code

    def addcell(self, segment_number, type_of_value, value):
        """ Adds a notebook cell, by updating the json template. Cell differs with type of value """
        if type_of_value in ['source', 'input']:
            self.template["worksheets"][0]["cells"].append(copy.deepcopy(self.cell_type[type_of_value]))
            self.template["worksheets"][0]["cells"][segment_number][type_of_value] = value

    def json(self, notebook_path):
        """ Writes the template to file (json) """
        with open(notebook_path, 'w') as output:
            json.dump(self.template, output, indent=2)


def save_ipython_notebook(example_file, notebook_dir, notebook_path):
    """ Saves a Python file as an IPython notebook 

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

    modified_code = nb.getModifiedCode()

    for line in modified_code:
        # A linebreak indicates a segment has ended. If the text segment had only comments, then source is blank,
        # So, ignore it, as already added in cell type markdown
        if line == "\n":
            if segment_has_begun is True and source:
                segment_number += 1
                # we've found text segments within the docstring
                if docstring is True:
                    nb.addcell(segment_number, 'source', source)
                else:
                    nb.addcell(segment_number, 'input', source)
                source = []
        # if it's a comment
        elif line.strip().startswith('#'):
            segment_number += 1
            line = line.strip(' #')
            nb.addcell(segment_number, 'source', line)
        elif line == '"""\n':
            if docstring is False:
                docstring = True
            # Indicates, completion of docstring, add whatever in source to markdown (cell type markdown)
            elif docstring is True:
                docstring = False
                # Write leftover docstring if any left
                if source:
                    segment_number += 1
                    nb.addcell(segment_number, 'source', source)
                    source = []
        else:
            # some text segment is continuing, so add to source
            source.append(line)
    
    nb.json(notebook_path)
