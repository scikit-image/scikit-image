class InheritedConfig(dict):
    """Configuration dictionary where non-existent keys can inherit values.

    This class allows you to define parameter names that can match exactly, but
    if it doesn't, parameter names will be searched based on key inheritance.
    For example, the key 'size.text' will default to 'size'.

    Note that indexing into the dictionary will raise an error if it doesn't
    match exactly, while `InheritedConfig.get` will look up values based on
    inheritance.

    Parameters
    ----------
    config_values : dict or list of (key, value) pairs
        Default values for a configuration, where keys are the parameter names
        and values are the associated value.
    cascade_map : dict
        Dictionary defining cascading defaults. If a parameter name is not
        found, indexing `cascade_map` with the parameter name will return
        the parameter to look for.
    kwargs : dict
        Keyword arguments for initializing dict.
    """

    _separator = '.'

    def __init__(self, config_values=None, **kwargs):
        assert 'config_values' not in kwargs

        if config_values is None:
            config_values = {}

        super(InheritedConfig, self).__init__(config_values, **kwargs)

    def get(self, key, default=None, _prev=None):
        """Return best matching config value for `key`.

        Get value from configuration. The search for `key` is in the following
        order:

            - `self` (Value in global configuration)
            - `default`
            - Alternate key specified by `self.cascade_map`

        This method supports the pattern commonly used for optional keyword
        arguments to a function. For example::

            >>> def print_value(key, **kwargs):
            ...     print kwargs.get(key, 0)
            >>> print_value('size')
            0
            >>> print_value('size', size=1)
            1

        Instead, you would create a config class and write::

            >>> config = InheritedConfig(size=0)
            >>> def print_value(key, **kwargs):
            ...     print kwargs.get(key, config.get(key))
            >>> print_value('size')
            0
            >>> print_value('size', size=1)
            1
            >>> print_value('non-existent')
            None
            >>> print_value('size.text')
            0

        See examples below for a demonstration of the cascading of
        configuration names.

        Parameters
        ----------
        key : str
            Name of config value you want.
        default : object
            Default value if `key` doesn't exist in instance.

        Examples
        --------
        >>> config = InheritedConfig(size=0)
        >>> config.get('size')
        0
        >>> top_choice={'size': 1}
        >>> top_choice.get('size', config.get('size'))
        1
        >>> config.get('non-existent', 'unknown')
        'unknown'
        >>> config.get('size.text')
        0
        >>> config.get('size.text', 2)
        2
        >>> top_choice.get('size', config.get('size.text'))
        1
        """
        if key in self.keys():
            return self[key]
        elif default is not None:
            return default
        elif self._separator in key:
            return self.get(self._parent(key))
        else:
            return None

    def _parent(self, key):
        """Return parent key."""
        parts = key.split(self._separator)
        return self._separator.join(parts[:-1])

    def __contains__(self, key):
        if key in self.keys():
            return True
        elif self._separator in key:
            return self.__contains__(self._parent(key))
        else:
            return False


if __name__ == '__main__':
    import doctest

    doctest.testmod()
