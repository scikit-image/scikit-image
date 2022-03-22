# -*- coding: utf-8 -*-

""" Create lookup tables for the marching cubes algorithm, by parsing
the file "LookUpTable.h". This prints a text to the stdout which
can then be copied to luts.py.

The luts are tuples of shape and base64 encoded bytes.

"""
import base64


def create_luts(fname):

    # Get the lines in the C header file
    text = open(fname,'rb').read().decode('utf-8')
    lines1 = [line.rstrip() for line in text.splitlines()]

    # Init lines for Python
    lines2 = []

    # Get classic table
    more_lines, ii = get_table(lines1, 'static const char casesClassic', 0)
    lines2.extend(more_lines)

    # Get cases table
    more_lines, ii = get_table(lines1, 'static const char cases', 0)
    lines2.extend(more_lines)

    # Get tiling tables
    ii = 0
    for casenr in range(99):
        # Get table
        more_lines, ii = get_table(lines1, 'static const char tiling', ii+1)
        if ii < 0:
            break
        else:
            lines2.extend(more_lines)

    # Get test tables
    ii = 0
    for casenr in range(99):
        # Get table
        more_lines, ii = get_table(lines1, 'static const char test', ii+1)
        if ii < 0:
            break
        else:
            lines2.extend(more_lines)

    # Get subconfig tables
    ii = 0
    for casenr in range(99):
        # Get table
        more_lines, ii = get_table(lines1, 'static const char subconfig', ii+1)
        if ii < 0:
            break
        else:
            lines2.extend(more_lines)

    return '\n'.join(lines2)


def get_table(lines1, needle, i):

    # Try to find the start
    ii = search_line(lines1, needle, i)
    if ii < 0:
        return [], -1

    # Init result
    lines2 = []

    # Get size and name
    front, dummu, back = lines1[ii].partition('[')
    name = front.split(' ')[-1].upper()
    size = int(back.split(']',1)[0])
    cdes = lines1[ii].rstrip(' {=')

    # Write name
    lines2.append('%s = np.array([' % name)

    # Get elements
    for i in range(ii+1, ii+1+9999999):
        line1 = lines1[i]
        front, dummy, back = line1.partition('*/')
        if not back:
            front, back = back, front
        line2 = '    '
        line2 += back.strip().replace('{', '[').replace('}',']').replace(';','')
        line2 += front.replace('/*','  #').rstrip()
        lines2.append(line2)
        if line1.endswith('};'):
            break

    # Close and return
    lines2.append("    , 'int8')")
    lines2.append('')
    #return lines2, ii+size

    # Execute code and get array as base64 text
    code = '\n'.join(lines2)
    code = code.split('=',1)[1]
    array = eval(code)
    array64 = base64.encodebytes(array.tostring()).decode('utf-8')
    # Reverse: bytes = base64.decodebytes(text.encode('utf-8'))
    text = '%s = %s, """\n%s"""' % (name, str(array.shape), array64)

    # Build actual lines
    lines2 = []
    #lines2.append( '# %s -> %s %s' % (cdes, str(array.dtype), str(array.shape)) )
    lines2.append( '#' + cdes)
    lines2.append(text)
    lines2.append('')
    return lines2, ii+size


def search_line(lines, refline, start=0):
    for i, line in enumerate(lines[start:]):
        if line.startswith(refline):
            return i + start
    return -1



def getLutNames(prefix):
    aa = []
    for a in dir(luts):
        if a.startswith(prefix): aa.append(a)

    def sortkey(x):
        fullnr = x.split(prefix)[1]
        nr, us, subnr = fullnr.partition('_')
        if len(nr) == 1:
            nr = '0'+nr
        return nr + us + subnr

    return [a for a in sorted(aa, key=sortkey)]



if __name__ == '__main__':
    import os
    fname = os.path.join(os.getcwd(), 'LookUpTable.h')

    with open(os.path.join(os.getcwd(), 'mcluts.py'), 'w') as f:
        f.write('# -*- coding: utf-8 -*-\n')
        f.write(
            '# This file was auto-generated from `mc_meta/LookUpTable.h` by\n'
            '# `mc_meta/createluts.py`.\n\n'
        )
        f.write(create_luts(fname))
