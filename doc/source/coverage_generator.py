#!/usr/bin/env python

import csv

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
            entry = ""
        output += " " + entry + " "*(lengths[i] - len(entry)) + " |"
    return output

csv_path = 'test.csv'
reader = csv.reader(open(csv_path,'r'),delimiter=',',quotechar='"')

# Find number of columns and column widths, base number of columns is
# determined by the headers
page_title = "Coverage Tables"
print page_title
print "="*len(page_title)
print
table_names = ["Image Display and Exploration","Image File I/O"]
for table in table_names:
    num_columns = 3
    data = [["Functionality","Matlab","Scipy"]]
    for row in reader:
        if len(row) == 0:
            break
        data.append([entry.expandtabs() for entry in row])
        num_columns = max(num_columns,len(row))

    column_lengths = [0]*num_columns
    for row in data:
        for i in xrange(len(row)):
            column_lengths[i] = max(column_lengths[i],len(row[i]))
    
    print table
    print "-"*len(table)
    print
    print table_seperator(num_columns,column_lengths,character="-")
    print table_row(data[0],column_lengths,num_columns)
    print table_seperator(num_columns,column_lengths,character="=")
    for row in data[1:]:
        print table_row(row,column_lengths,num_columns)
        print table_seperator(num_columns,column_lengths,character='-')
    print
    print
