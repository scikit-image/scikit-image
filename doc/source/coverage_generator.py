#!/usr/bin/env python

import sys
import os
import csv

# Import StringIO module
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

# Missing item value
MISSING_STRING=":missing:`Not Implemented`"

def read_table_titles(reader):
    r"""Create a dictionary with keys as section names and values as a list of
    table names
    
    return (dict)
    """
    section_titles = []
    table_names = {}
    try:
        for row in reader:
            names = []
            # End of names table
            if len(row[0]) == 0:
                break
            # Extract names of the tables
            for name in row[1:]:
                if len(name) > 0:
                    names.append(name) 
                else:
                    break
            section_titles.append(row[0])
            table_names[row[0]] = names
    except csv.Error, e:
        sys.exit('line %d: %s' % (reader.line_num, e))
    
    return section_titles,table_names

def table_seperator(stream,cols,lengths,character="-"):
    stream.write("+")
    stream.write('+'.join([character*(length+2) for length in lengths]))
    stream.write("+")
    
def table_row(stream,data,lengths,num_columns=None):
    if num_columns is None:
        num_columns = len(data)
    stream.write("|")
    for i in xrange(num_columns):
        if len(data)-1 >= i:
            if len(data[i]) == 0:
                entry = MISSING_STRING
            else:
                entry = data[i]
        else:
            entry = MISSING_STRING
        stream.write(" " + entry + " "*(lengths[i] - len(entry)) + " |")
    
def generate_table(reader,stream,table_name,
                    column_titles=["Functionality","Matlab","Scipy","Scipy"]):
    # Find number of columns and column widths, base number of columns is
    # determined by the headers
    num_columns = len(column_titles)
    data = [column_titles]
    try:
        for row in reader:
            # print row
            if len(row[0]) == 0:
                break
            data.append([entry.expandtabs() for entry in row])
            num_columns = max(num_columns,len(row))
    except csv.Error, e:
        sys.exit('line %d: %s' % (reader.line_num, e))

    column_lengths = [len(MISSING_STRING)]*num_columns
    for row in data:
        for i in xrange(len(row)):
            column_lengths[i] = max(column_lengths[i],len(row[i]))
    
    # Output table header
    stream.write(table_name + "\n")
    stream.write("~"*len(table_name)+"\n\n")
    stream.write(".. tabularcolumns:: |p{40%}|p{20%}|p{20%}|p{20%}|\n\n")
    stream.write(".. table::%s\n\n" % table_name)
    table_seperator(stream,num_columns,column_lengths,character="-")
    stream.write("\n")
    table_row(stream,data[0],column_lengths,num_columns)
    stream.write("\n")
    table_seperator(stream,num_columns,column_lengths,character="=")
    stream.write("\n")
    
    # Output table data
    for row in data[1:]:
        table_row(stream,row,column_lengths,num_columns)
        stream.write("\n")
        table_seperator(stream,num_columns,column_lengths,character='-')
        stream.write("\n")
    stream.write("\n\n")
    
def generate_page(reader,stream,page_title="Coverage Tables"):
    stream.write("%s\n" % page_title)
    stream.write("="*len(page_title) + "\n\n")
    stream.write(".. role:: missing\n")
    stream.write(".. role:: partial\n")
    stream.write(".. role:: done\n")
    stream.write(".. role:: na\n\n")
    stream.write("Color Key\n")
    stream.write("---------\n")
    stream.write(":done:`Complete` ")
    stream.write(":partial:`Partial` ")
    stream.write(":missing:`Missing` ")
    stream.write(":na:`Not applicable`\n\n")
    
    sections,table_names = read_table_titles(reader)
    for section_name in sections:
        stream.write(section_name + "\n")
        stream.write("-"*len(section_name) + "\n\n")
        for table_name in table_names[section_name]:
            generate_table(reader,stream,table_name)

if __name__ == "__main__":
    csv_path = './coverage.csv'
    output_path = './coverage_table.txt'
    if len(sys.argv) == 2:
        csv_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) == 3:
        output_path = os.path.abspath(sys.argv[2])
    
    # Figure out dialect and create csv reader
    csv_file = open(csv_path,'U')
    # dialect = csv.Sniffer().sniff(csv_file.read(1024))
    # csv_file.seek(0)
    # reader = csv.reader(csv_file, dialect)
    reader = csv.reader(csv_file,)
    
    output = open(output_path,'w')
    generate_page(reader,output)

    csv_file.close()
    output.close()
    
    print "Generated %s from %s." % (output_path,csv_path)


