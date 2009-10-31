from textwrap import dedent
import numpy as np
# some utility functions for the opencv wrappers


# the doc decorator
class cvdoc(object):
    ''' a doc decorator which adds the docs for the opencv functions.
    It primarily serves to append the appropriate opencv doc url
    to each function.
    '''

    base_url = 'http://opencv.willowgarage.com/documentation/'
    branch_urls = {'cv': {'image': 'image_processing',
                          'structural': 'structural_analysis',
                          'calibration': 'camera_calibration_and_3d_reconstruction'
                          },
                   'cxcore': {},
                   'highgui': {}
                   }

    def __init__(self, package='', group='', doc=''):
        self.package = str(package)
        self.group = str(group)
        self.doc = str(doc)

    def __call__(self, func):
        # if key errors occur, fail silently
        try:
            self._add_url(func)
            np.add_docstring(func, self.doc)
            return func

        except KeyError:
            return func

    def _add_url(self, func):
        name = func.__name__
        full_url = (self.base_url +
                    self.branch_urls[self.package][self.group] +
                    '.html' + '#' + name)
        message = dedent('''
            The OpenCV documentation for this fuction can
            be found at the following url:''')

        self.doc += '\n\n' + message + '\n\n' + full_url + '\n'

