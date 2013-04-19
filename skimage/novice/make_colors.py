#!/usr/bin/python
import argparse, csv

# Imports color names from a CSV file (colors.csv) and outputs a Python file
# with color name definitions (colors.py).

def download_colors(path):
    import urllib2, re
    from bs4 import BeautifulSoup

    # All colors seem to be on this page
    url = "http://en.wikipedia.org/wiki/List_of_colors:_A-M"
    request = urllib2.Request(url)
    opener = urllib2.build_opener()
    request.add_header("User-Agent", "Mozilla/5.0")
    html = opener.open(request).read()

    # Find first table with "Color names" caption
    soup = BeautifulSoup(html, "lxml")
    color_table = [t for t in soup.findAll("table") if t.find("caption") is not None
                   and t.find("caption").string == "Color names"][0]

    with open(path, "w") as out_file:
        writer = csv.writer(out_file)

        # Each table row has the color name in a th and the hex
        # color in the second td cell.
        for row in color_table.findAll("tr"):
            th = row.find("th")
            cells = row.findAll("td")
            if th is None or len(cells) < 2:
                continue

            name = th.string.encode("ascii", "ignore")

            # Strip parens and replace non-alphanumeric characters with underscores
            id_name = re.sub("[^a-zA-Z0-9]", "_", name.lower().replace("(", "").replace(")", ""))
            hex_str = cells[1].string.encode("ascii", "ignore")
            r, g, b = (int(hex_str[1:3], 16), int(hex_str[3:5], 16), int(hex_str[5:7], 16))

            writer.writerow((id_name, name, hex_str, str(r), str(g), str(b)))

# -------------------------------------------------- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="colors.csv", help="Input file (CSV)")
    parser.add_argument("--output", "-o", type=str, default="colors.py", help="Output file (Python)")
    parser.add_argument("--download", "-d", action="store_true", help="Download from Wikipedia")
    args = parser.parse_args()

    if args.download:
        download_colors(args.input)

    with open(args.input, "r") as input_file:
        with open(args.output, "w") as output_file:
            for row in csv.reader(input_file):
                name = row[0].upper()
                rgb = (int(row[-3]), int(row[-2]), int(row[-1]))
                output_file.write("{0} = {1}\n".format(name, rgb))
