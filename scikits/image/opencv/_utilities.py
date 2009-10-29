from textwrap import dedent
import numpy as np
# some utility functions for the opencv wrappers


# the doc decorator
class cvdoc(object):
    ''' a doc decorator which adds the docs for the opencv functions.
    It should be self-explanatory. See how the arguments are passed in e.g.
    opencv_cv.pyx
    '''

    SIGNATURE = '''Signature\n---------\n'''
    PARAMETERS = '''Parameters\n----------\n'''
    RETURNS = '''Returns\n-------\n'''
    NOTES = '''Notes\n-----\n'''
    EXAMPLES = '''Examples\n--------\n'''
    base_url = 'http://opencv.willowgarage.com/documentation/'
    branch_urls = {'cv': {'image': 'image_processing',
                          'structural': 'structural_analysis',
                          'calibration': 'camera_calibration_and_3D_reconstruction'
                          },
                   'cxcore': {},
                   'highgui': {}
                   }

    def __init__(self, description='', signature='', parameters='', returns='',
                 notes='', examples='', package='', group=''):
        self.description = str(description)
        self.signature = str(signature)
        self.parameters = str(parameters)
        self.returns = str(returns)
        self.notes = str(notes)
        self.examples = str(examples)
        self.package = str(package)
        self.group = str(group)
        self.doc = ''''''

    def __call__(self, func):
        # if key errors occur, fail silently
        try:
            self._add_description()
            self._add_signature()
            self._add_parameters()
            self._add_returns()
            self._add_notes()
            self._add_examples()
            self._add_url(func)
            np.add_docstring(func, self.doc)
            return func

        except KeyError:
            return func

    def _add_description(self):
        if self.description != '':
            self.doc += self.description + '\n\n'

    def _add_signature(self):
        if self.signature != '':
            self.doc += self.SIGNATURE + self.signature + '\n\n'

    def _add_parameters(self):
        if self.parameters != '':
            self.doc += self.PARAMETERS + self.parameters + '\n\n'

    def _add_returns(self):
        if self.returns != '':
            self.doc += self.RETURNS + self.returns + '\n\n'

    def _add_notes(self):
        if self.notes != '':
            self.doc += self.NOTES + self.notes + '\n\n'

    def _add_examples(self):
        if self.examples != '':
            self.doc += self.EXAMPLES + self.examples + '\n\n'

    def _add_url(self, func):
        name = func.__name__
        full_url = (self.base_url +
                    self.branch_urls[self.package][self.group] +
                    '.html' + '#' + name)
        message = dedent('''
            The OpenCV documentation for this fuction can
            be found at the following url:''')

        self.doc += message + '\n\n' + full_url + '\n'

