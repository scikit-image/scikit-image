class DefaultValue(object):
    """Class to provide a dummy default value other than None.

    This is useful for cases where it makes sense to pass 'None' as an
    actual value.
    """
    def __repr__(self):
        return 'None'

    def __str__(self):
        return 'None'


DEFAULT = DefaultValue()
