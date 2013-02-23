#!/usr/bin/python

# Imports color names from a CSV file (colors.csv) and outputs a Python file
# with color name definitions (colors.py).

import sys, csv

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: make_colors.py csv-file python-file"
        sys.exit(1)

    with open(sys.argv[1], "r") as input_file:
        with open(sys.argv[2], "w") as output_file:
            for row in csv.reader(input_file):
                name = row[0].upper()
                rgb = (int(row[-3]), int(row[-2]), int(row[-1]))
                output_file.write("{0} = {1}\n".format(name, rgb))
