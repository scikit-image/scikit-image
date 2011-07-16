#!/usr/bin/env python

import sys
import os
import csv

# Parameters
missing_string = ":missing:`Not Implemented`"
page_title = "Coverage Tables"
table_names = ["Image Display and Exploration","Image File I/O"]

def table_seperator(cols,lengths,character="-"):
    output = "+"
    output += '+'.join([character*(length+2) for length in lengths])
    output += "+"
    return output
    
def table_row(data,lengths,num_columns=None):
    if num_columns is None:
        num_columns = len(data)
    output = "|"
    for i in xrange(num_columns):
        if len(data)-1 >= i:
            entry = data[i]
        else:
            entry = missing_string
        output += " " + entry + " "*(lengths[i] - len(entry)) + " |"
    return output
    
def generate_table(reader,column_titles=["Functionality","Matlab","Scipy"]):
    # Find number of columns and column widths, base number of columns is
    # determined by the headers
    num_columns = 3
    data = [column_titles]
    for row in reader:
        if len(row) == 0:
            break
        data.append([entry.expandtabs() for entry in row])
        num_columns = max(num_columns,len(row))

    column_lengths = [len(missing_string)]*num_columns
    for row in data:
        for i in xrange(len(row)):
            column_lengths[i] = max(column_lengths[i],len(row[i]))
    
    output = table + "\n"
    output += "-"*len(table)+"\n\n"
    output += table_seperator(num_columns,column_lengths,character="-") + "\n"
    output += table_row(data[0],column_lengths,num_columns) + "\n"
    output += table_seperator(num_columns,column_lengths,character="=") + "\n"
    for row in data[1:]:
        output += table_row(row,column_lengths,num_columns) + "\n"
        output += table_seperator(num_columns,column_lengths,character='-') + "\n"
    output += "\n\n"
    return output

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = os.path.abspath(sys.argv[1])
    else:
        csv_path = './coverage.csv'

    reader = csv.reader(open(csv_path,'r'),delimiter=',',quotechar='"')

print page_title
print "="*len(page_title)
print
print "..  role:: missing"
print
for table in table_names:
    print generate_table(reader)
