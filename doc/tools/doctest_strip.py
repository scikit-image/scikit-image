"""
Copy documentation files into new temp directory removing doctest annotations.

"""

# Stdlib imports
import os
import re
import shutil

find_doctest = re.compile(r"\s*?# doctest: \++")


def strip_doctest_annotations(path):
    """
    Remove trailing ' # doctest +' annotations if present.
    """
    # Open file, read it line by line
    temp = []
    with open(path, 'r') as docfile:
        for line in docfile:
            temp.append(find_doctest.split(line)[0])

    # Write back result
    with open(path, 'w') as new_docfile:
        new_docfile.writelines(temp)


if __name__ == '__main__':
    import sys

    # Copy doc source to a new temp directory
    shutil.copytree('./source/', './_prebuild/')

    # Parse input flags, see if doctest stripping was requested
    if len(sys.argv) < 2:
        # Parse by default, with no additional flags

        # Find text-like files in _prebuild which need this applied, currently
        #   *.py
        #   *.txt
        files_to_strip = []
        for root, dirnames, filenames in os.walk('./_prebuild/'):
            for filename in filenames:
                if filename.endswith(('.txt', '.py')):
                    files_to_strip.append(os.path.join(root, filename))

        # Strip out all lines matching find_doctest pattern
        for text_file in files_to_strip:
            strip_doctest_annotations(text_file)

    else:
        # If any additional argument(s) passed, doc sources are copied but
        # doctest annotation removal is not done
        pass
