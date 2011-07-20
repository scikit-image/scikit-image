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

def calculate_coverage(reader,blocks=100):
    r"""Calculate portions of code that are in one of the coverage categories
    
    Returns a tuple representing the weighted items as integers.  The order is
        
        (done,partial,missing,not applicable)
    """
    # Coverage counters
    total_items = 0
    partial_items = 0
    done_items = 0
    na_items = 0

    # Skip table names
    for row in reader:
        if len(row[0]) == 0:
            break

    # Count items
    for row in reader:
        if len(row[0]) > 0:
            total_items += 1
            if ":done:" in row[2] or ":done:" in row[3]:
                done_items += 1
            elif ":partial:" in row[2] or ":partial:" in row[3]:
                partial_items += 1
            elif ":na:" in row[2] or ":na:" in row[3]:
                na_items += 1
    
    weight = float(blocks) / total_items
    
    return (int(weight*done_items),
            int(weight*partial_items),
            int(weight*(total_items - (partial_items + done_items + na_items))),
            int(weight*na_items))
                

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

def table_seperator(stream,lengths,character="-"):
    r"""Write out table row seperator
    
    :Input:
     - *stream* (io/stream) Stream where output is put
     - *lengths* (list) A list of the lengths of the columns
     - *character* (string) Character to be filled between +, defaults to "-".
     
    """
    stream.write("+")
    stream.write('+'.join([character*(length+2) for length in lengths]))
    stream.write("+")
    
def table_row(stream,data,lengths,num_columns=None):
    r"""Write out table row data
    
    :Input:
     - *stream* (io/stream) Stream where output is put
     - *data* (list) List of strings containing data
     - *lengths* (list) A list of the lengths of the columns
     - *num_columns* (string) Number of columns, defaults to the length of the 
       data array
     
    """
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
    
def generate_table(reader,stream,table_name=None,
                    column_titles=["Functionality","Matlab","Scipy","Scipy"]):
    r"""Generate a reST grid table based on the CSV data in reader
    
    Reads CSV data from *reader* until an empty line is found and generates a
    reST table based on the data into *stream*.  A table name can be given for
    a section and table label.  All rows are read in and checked for maximum
    number of columns (defaults to the size of column_titles) and column
    widths so that the table can be constructed.  If a row contains less than
    the maximum number of columns a string is inserted that defaults to the
    string *MISSING_STRING* which is a global parameter.
    
    :Input:
     - reader (csv.reader) The CSV reader to read in from
     - stream (iostream) Output target
     - table_name (string) Optional name of table, defaults to *None*
     - column_titles (list) List of column titles
     
    """
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
    if table_name is not None:
        stream.write("~"*len(table_name)+"\n\n")
    stream.write(".. tabularcolumns:: |p{40%}|p{20%}|p{20%}|p{20%}|\n\n")
    if table_name is not None:
        stream.write(".. table::%s\n\n" % table_name)
    table_seperator(stream,column_lengths,character="-")
    stream.write("\n")
    table_row(stream,data[0],column_lengths,num_columns)
    stream.write("\n")
    table_seperator(stream,column_lengths,character="=")
    stream.write("\n")
    
    # Output table data
    for row in data[1:]:
        table_row(stream,row,column_lengths,num_columns)
        stream.write("\n")
        table_seperator(stream,column_lengths,character='-')
        stream.write("\n")
    stream.write("\n\n")
    
def generate_page(csv_path,stream,page_title="Coverage Tables"):
    r"""Generate coverage table page
    
    Generates all reST for all tables contained in the CSV file at *csv_path*
    and output it to *stream*.
    
    :Input:
     - *csv_path* (path) Path to CSV file
     - *stream* (iostream) Output stream
     - *page_title* (string) Optional page title, defaults to 
       ``Coverage Tables``.
    """
    # Open reader
    csv_file = open(csv_path,'U')
    
    # Sniffer does not seem to work all the time even when an Excel
    # spread sheet is being used
    # dialect = csv.Sniffer().sniff(csv_file.read(1024))
    # csv_file.seek(0)
    # reader = csv.reader(csv_file, dialect)
    
    reader = csv.reader(csv_file)
    item_counts = calculate_coverage(reader,blocks=90)
    csv_file.seek(0)
    
    # Write out header
    stream.write("%s\n" % page_title)
    stream.write("="*len(page_title) + "\n\n")
    stream.write(".. role:: missing\n")
    stream.write(".. role:: partial\n")
    stream.write(".. role:: done\n")
    stream.write(".. role:: na\n\n")
    stream.write(".. role:: missing-bar\n")
    stream.write(".. role:: partial-bar\n")
    stream.write(".. role:: done-bar\n")
    stream.write(".. role:: na-bar\n\n")
    stream.write("Color Key\n")
    stream.write("---------\n")
    stream.write(":done:`Complete` ")
    stream.write(":partial:`Partial` ")
    stream.write(":missing:`Missing` ")
    stream.write(":na:`Not applicable`\n\n")
    stream.write("Coverage Bar\n")
    stream.write("------------\n\n")
    if item_counts[0] > 0:
        stream.write(":done-bar:`%s` " % ("="*item_counts[0]))
    if item_counts[1] > 0:
        stream.write(":partial-bar:`%s` " % ("="*item_counts[1]))
    if item_counts[2] > 0:
        stream.write(":missing-bar:`%s` " % ("="*item_counts[2]))
    if item_counts[3] > 0:
        stream.write(":na-bar:`%s`" % ("="*item_counts[3]))
    stream.write("\n\n")
    
    sections,table_names = read_table_titles(reader)
    for section_name in sections:
        stream.write(section_name + "\n")
        stream.write("-"*len(section_name) + "\n\n")
        for table_name in table_names[section_name]:
            generate_table(reader,stream,table_name)
    
    csv_file.close()

if __name__ == "__main__":
    csv_path = './coverage.csv'
    output_path = './coverage_table.txt'
    if len(sys.argv) > 1:
        if sys.argv[1][:5].lower() == "help":
            print "Coverage Table Generator: coverage_generator.py"
            print "  Usage: coverage_generator.py [csv] [output]"
            print "    csv - Path to csv file, defaults to ./coverage.csv"
            print "    output - Ouput path, defaults to ./coverage_table.txt"
            print ""
            sys.exit(0)
        if len(sys.argv) == 2:
            csv_path = os.path.abspath(sys.argv[1])
        if len(sys.argv) == 3:
            output_path = os.path.abspath(sys.argv[2])
    
    output = open(output_path,'w')
    generate_page(csv_path,output)
    output.close()
    
    print "Generated %s from %s." % (output_path,csv_path)


