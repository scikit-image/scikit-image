import urllib.parse
import urllib.request

import os
import re
import tempfile
from contextlib import contextmanager


URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')


def is_url(filename):
    """Return True if string is an http or ftp path."""
    return (isinstance(filename, str) and
            URL_REGEX.match(filename) is not None)


@contextmanager
def file_or_url_context(resource_name):
    """Yield name of file from the given resource (i.e. file or url)."""
    if is_url(resource_name):
        url_components = urllib.parse.urlparse(resource_name)
        _, ext = os.path.splitext(url_components.path)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                u = urllib.request.urlopen(resource_name)
                f.write(u.read())
            # f must be closed before yielding
            yield f.name
        except OSError:
            raise ValueError('unreachable URL: {}'.format(resource_name))
        finally:
            os.remove(f.name)
    else:
        yield resource_name
