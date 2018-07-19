import os
from os.path import isfile
from os.path import join
import re

notebook_path = os.path.abspath("notebooks")
notebook_dir = os.listdir("notebooks")

files = [name for name in notebook_dir if re.match("^.*py$", name)]

for f in files:
    print(f)
