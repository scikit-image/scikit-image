This qt subpackage provides a wrapper to allow use of either PySide or PyQt4.
In addition, if neither package is available, some mock objects are created to
prevent errors in the TravisCI build. Only the objects used in the global
namespace need to be mocked (e.g., a Qt object that gets subclassed is used
in the global namespace).
