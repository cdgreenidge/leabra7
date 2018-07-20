import json
import os
from os.path import join
import re

notebook_path = os.path.abspath("notebooks")
notebook_dir = os.listdir("notebooks")

filenames = [name for name in notebook_dir if re.match("^.*pynb$", name)]

for name in filenames:
    file_path = join(notebook_path, name)

    out_filepath = join(notebook_path, name[:-5] + "py")
    out_file = open(out_filepath, "w")

    file_data = json.loads(open(file_path).read())
    file_cells = file_data['cells']

    for cell in file_cells:
        source = cell['source']
        if cell['cell_type'] == "code":
            out_file.write("# Begin Code\n")
            for line in source:
                if line == "%matplotlib inline\n":
                    out_file.write("# %matplotlib inline\n")
                else:
                    out_file.write(line)
        else:
            out_file.write("# Begin Markdown\n")
            for line in source:
                out_file.write("# " + line)
        out_file.write("\n \n")
    source = file_cells[0]['source']
