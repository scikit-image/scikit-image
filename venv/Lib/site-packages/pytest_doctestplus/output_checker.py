"""
Implements a replacement for `doctest.OutputChecker` that handles certain
normalizations of Python expression output.  See the docstring on
`OutputChecker` for more details.
"""

import doctest
import re
import math


# Much of this code, particularly the parts of floating point handling, is
# borrowed from the SymPy project with permission.  See
# licenses/SYMPY_LICENSE.rst for the full SymPy license.


FIX = doctest.register_optionflag('FIX')
FLOAT_CMP = doctest.register_optionflag('FLOAT_CMP')
REMOTE_DATA = doctest.register_optionflag('REMOTE_DATA')
IGNORE_OUTPUT = doctest.register_optionflag('IGNORE_OUTPUT')
IGNORE_OUTPUT_3 = doctest.register_optionflag('IGNORE_OUTPUT_3')
IGNORE_WARNINGS = doctest.register_optionflag('IGNORE_WARNINGS')
SHOW_WARNINGS = doctest.register_optionflag('SHOW_WARNINGS')

# These might appear in some doctests and are used in the default pytest
# doctest plugin. This plugin doesn't actually implement these flags but this
# allows them to appear in docstrings.
ALLOW_BYTES = doctest.register_optionflag('ALLOW_BYTES')
ALLOW_UNICODE = doctest.register_optionflag('ALLOW_UNICODE')


class OutputChecker(doctest.OutputChecker):
    """
    - Removes u'' prefixes on string literals
    - Ignores the 'L' suffix on long integers
    - In Numpy dtype strings, removes the leading pipe, i.e. '|S9' ->
      'S9'.  Numpy 1.7 no longer includes it in display.
    - Supports the FLOAT_CMP flag, which parses floating point values
      out of the output and compares their numerical values rather than their
      string representation.  This naturally supports complex numbers as well
      (simply by comparing their real and imaginary parts separately).
    """
    rtol = 1e-05
    atol = 1e-08

    _str_literal_re = re.compile(
        r"(\W|^)[uU]([rR]?[\'\"])", re.UNICODE)
    _byteorder_re = re.compile(
        r"([\'\"])[|<>]([biufcSaUV][0-9]+)([\'\"])", re.UNICODE)
    _fix_32bit_re = re.compile(
        r"([\'\"])([iu])[48]([\'\"])", re.UNICODE)
    _long_int_re = re.compile(
        r"([0-9]+)L", re.UNICODE)

    def __init__(self):
        exp = r'(?:e[+-]?\d+)'

        got_floats = (r'\s*([+-]?\d+\.\d*{0}?|'
                      r'[+-]?\.\d+{0}?|'
                      r'[+-]?\d+{0}|'
                      r'nan|'
                      r'[+-]?inf)').format(exp)

        # floats in the 'want' string may contain ellipses
        want_floats = got_floats + r'(\.{3})?'

        front_sep = r'\s|[*+-,<=(\[]'
        back_sep = front_sep + r'|[>j)\]}]'

        fbeg = fr'^{got_floats}(?={back_sep}|$)'
        fmidend = fr'(?<={front_sep}){got_floats}(?={back_sep}|$)'
        self.num_got_rgx = re.compile(fr'({fbeg}|{fmidend})')

        fbeg = fr'^{want_floats}(?={back_sep}|$)'
        fmidend = fr'(?<={front_sep}){want_floats}(?={back_sep}|$)'
        self.num_want_rgx = re.compile(fr'({fbeg}|{fmidend})')

        # As of 2023-09-26, Python base class has no init, but just in case
        # it acquires one.
        super().__init__()

    def do_fixes(self, want, got):
        want = re.sub(self._str_literal_re, r'\1\2', want)
        want = re.sub(self._byteorder_re, r'\1\2\3', want)
        want = re.sub(self._fix_32bit_re, r'\1\2\3', want)
        want = re.sub(self._long_int_re, r'\1', want)

        got = re.sub(self._str_literal_re, r'\1\2', got)
        got = re.sub(self._byteorder_re, r'\1\2\3', got)
        got = re.sub(self._fix_32bit_re, r'\1\2\3', got)
        got = re.sub(self._long_int_re, r'\1', got)

        return want, got

    def find_numbers(self, text):
        """
        Find float strings in text.
        >>> OutputChecker().find_numbers("1.1 foo abr 2.22")
        ['1.1', '2.22']
        """
        matches = self.num_want_rgx.finditer(text)
        return [match.group(1) for match in matches]

    def equal_floats(self, a, b):
        """
        Compare float strings.
        >>> OutputChecker().equal_floats('1.1', '1.10000000001')
        True
        >>> OutputChecker().equal_floats('1.1', '1.11')
        False
        """
        a, b = float(a), float(b)
        return isclose(a, b, rtol=self.rtol, atol=self.atol)

    def startswith(self, arr, prefix):
        """
        Check if array of str/floats starts with floats in prefix.
        >>> OutputChecker().startswith(['1', '2', '3'], ['1', '2.00000000001'])
        True
        >>> OutputChecker().startswith(['1', '2', '3'], ['1', '2.1'])
        False
        """
        if len(prefix) == 0:
            return True
        if len(arr) < len(prefix):
            return False
        for a, b in zip(arr, prefix):
            if not self.equal_floats(a, b):
                return False
        return True

    def endswith(self, arr, postfix):
        """
        Check if array of str/floats ends with floats in postfix.
        >>> OutputChecker().endswith(['1', '2', '3'], ['2', '3.00000000001'])
        True
        >>> OutputChecker().endswith(['1', '2', '3'], ['2', '3.1'])
        False
        """
        return self.startswith(arr[::-1], postfix[::-1])

    def find(self, arr, suffix, start, end):
        """
        Search for floats from suffix in arr.
        >>> OutputChecker().find(['1', '2', '3', '4'], ['2', '3.00000000001'], 0, 4)
        1
        >>> OutputChecker().find(['1', '2', '3', '4'], ['2', '3.1'], 0, 4)
        -1
        """
        if len(suffix) == 0:
            return start
        arr = arr[start:end]
        for i, a in enumerate(arr):
            # if current floats match...
            if self.equal_floats(a, suffix[0]):
                # ... then compare the rest of numbers from suffix
                if self.startswith(arr[i:], suffix):
                    return start + i
        return -1

    def partial_match(self, arr, chunks):
        """
        Check that each chunk in chunks is inside provided array of strings/floats.
        This is essentially list-with-floats equivalent of ellipsis matching.
        >>> OutputChecker().partial_match(
        ...   ['1', '2', '3', '4'],
        ...   [['1', '2'], ['4']],
        ... )
        True
        >>> OutputChecker().partial_match(
        ...   ['1', '2', '3', '4'],
        ...   [['1', '2'], []],
        ... )
        True
        >>> OutputChecker().partial_match(
        ...   ['1', '2', '3', '4'],
        ...   [['1', '2'], ['5']],
        ... )
        False
        """
        assert len(chunks) >= 2
        startpos, endpos = 0, len(arr)
        chunk = chunks[0]
        if chunk:  # starts with exact match
            if self.startswith(arr, chunk):
                startpos = len(chunk)
                del chunks[0]
            else:
                return False
        chunk = chunks[-1]
        if chunk:  # ends with exact match
            if self.endswith(arr, chunk):
                endpos -= len(chunk)
                del chunks[-1]
            else:
                return False

        if startpos > endpos:
            return False

        for chunk in chunks:
            startpos = self.find(arr, chunk, startpos, endpos)
            if startpos < 0:
                return False
            startpos += len(chunk)

        return True

    def normalize_floats(self, want, got, flags):
        """
        Alternative to the built-in check_output that also handles parsing
        float values and comparing their numeric values rather than their
        string representations.

        This requires rewriting enough of the basic check_output that, when
        FLOAT_CMP is enabled, it totally takes over for check_output.
        """

        # <BLANKLINE> can be used as a special sequence to signify a
        # blank line, unless the DONT_ACCEPT_BLANKLINE flag is used.
        if not (flags & doctest.DONT_ACCEPT_BLANKLINE):
            # Replace <BLANKLINE> in want with a blank line.
            want = re.sub(fr'(?m)^{re.escape(doctest.BLANKLINE_MARKER)}\s*?$',
                          '', want)
            # If a line in got contains only spaces, then remove the
            # spaces.
            got = re.sub(r'(?m)^\s*?$', '', got)

        # This flag causes doctest to ignore any differences in the
        # contents of whitespace strings. Note that this can be used
        # in conjunction with the ELLIPSIS flag.
        if flags & doctest.NORMALIZE_WHITESPACE:
            got = ' '.join(got.split())
            want = ' '.join(want.split())

        # Handle the common case first, for efficiency:
        # if they're string-identical, always return true.
        if got == want:
            return True

        got_ = self.num_got_rgx.sub('0.0', got)
        want_ = self.num_got_rgx.sub('0.0', want)
        # fail if strings with ellipsis and normalize floats are not equal
        if flags & doctest.ELLIPSIS:
            if not doctest._ellipsis_match(want_, got_):
                return False
        else:
            if not got_ == want_:
                return False

        # at this point we made sure that non-float parts of strings are equivalent
        # so now we need to compare each number

        numbers_got = self.find_numbers(got)
        numbers_want_chunks = [
            self.find_numbers(chunk)
            for chunk in want.split(doctest.ELLIPSIS_MARKER)
        ]
        if flags & doctest.ELLIPSIS and len(numbers_want_chunks) >= 2:
            return self.partial_match(numbers_got, numbers_want_chunks)

        # TODO parse integers as well ?
        # Parse floats and compare them.
        numbers_want = [f for chunk in numbers_want_chunks for f in chunk]  # flatten array
        if len(numbers_got) != len(numbers_want):
            return False
        for ng, nw in zip(numbers_got, numbers_want):
            if not self.equal_floats(ng, nw):
                return False

        return True

    def check_output(self, want, got, flags):
        if ((flags & IGNORE_OUTPUT) or (flags & IGNORE_OUTPUT_3)):
            return True

        if flags & FIX:
            want, got = self.do_fixes(want, got)

        if flags & FLOAT_CMP:
            return self.normalize_floats(want, got, flags)

        return super().check_output(want, got, flags)

    def output_difference(self, want, got, flags):
        if flags & FIX:
            want, got = self.do_fixes(want, got)

        return super().output_difference(want, got, flags)


try:
    import numpy

    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=True):
        return numpy.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
except ImportError:
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=True):
        return abs(a - b) <= atol + rtol * abs(b) or (equal_nan and math.isnan(a) and math.isnan(b))
