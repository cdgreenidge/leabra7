import json
import os
from os.path import isfile
from os.path import join
import re



notebook_path = os.path.abspath("notebooks")
notebook_dir = os.listdir("notebooks")

filenames = [name for name in notebook_dir if re.match("^.*pynb$", name)]

test_file_path = join(notebook_path, "test.py")
test_file = open(test_file_path, "w")

for name in filenames:
    file_path = join(notebook_path, name)
    file_data = json.loads(open(file_path).read())
    file_cells = file_data['cells']
    for cell in file_cells:
        source = cell['source']
        if cell['cell_type'] == "code":
            for line in source:
                test_file.write(line)
        else:
            test_file.write("# Begin Markdown \n")
            for line in source:
                test_file.write("# " + line)
            test_file.write("\n# End Markdown")
        test_file.write("\n \n")
    source = file_cells[0]['source']


print()
