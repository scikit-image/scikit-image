import os
import warnings

qt_api = os.environ.get('QT_API')

if qt_api is None:
    try:
        import PySide
        qt_api = 'pyside'
    except ImportError:
        try:
            import PyQt4
            qt_api = 'pyqt'
        except ImportError:
            qt_api = 'none'
            # Note that we don't want to raise an error because that would
            # cause the TravisCI build to fail.
            warnings.warn("Could not import PyQt4: ImageViewer not available!")
    os.environ['QT_API'] = qt_api
