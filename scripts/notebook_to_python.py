#!/usr/bin/env python
import json
import sys

file_data = json.load(sys.stdin)
file_cells = file_data['cells']
for cell in file_cells:
    source = cell['source']
    if cell['cell_type'] == "code":
        sys.stdout.write("# Begin Code\n")
        for line in source:
            if line == "%matplotlib inline\n":
                sys.stdout.write("# %matplotlib inline\n")
            else:
                sys.stdout.write(line)
    else:
        sys.stdout.write("# Begin Markdown\n")
        for line in source:
            sys.stdout.write("# " + line)
    sys.stdout.write("\n \n")
