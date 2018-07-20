#!/usr/bin/env python
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import sys

def break_cells(data: List[str]) -> Tuple[List[List[str]], List[bool]]:
    """Breaks string into markdown and code cells."""
    last_cell = 0

    cells: List[List[str]] = []
    cell_code: List[bool] = []
    curr_code = False

    for n, line in enumerate(file_data):
        if line == "# Begin Markdown\n":
            if n:
                cells += [file_data[last_cell + 1:n - 1]]
                cell_code += [curr_code]
            last_cell = n
            curr_code = False

        elif line == "# Begin Code\n":
            if n:
                cells += [file_data[last_cell + 1:n - 1]]
                cell_code += [curr_code]
            last_cell = n
            curr_code = True

    cells += [file_data[last_cell + 1:len(file_data) - 1]]
    cell_code += [curr_code]

    return cells, cell_code

def build_cell_list(cells: List[List[str]], cell_code: List[bool]) -> List[Dict[str, Any]]:
    """Creates list of cells with notebook dictionary entries."""
    out_cells: List[Dict[str, Any]] = []

    for i, raw_source in enumerate(cells):
        new_cell: Dict[str, Any] = dict()
        source: List[str] = []
        if cell_code[i]:
            new_cell["cell_type"] = "code"
            new_cell["execution_count"] = None
            new_cell["outputs"] = []
        else:
            new_cell["cell_type"] = "markdown"

        new_cell["metadata"] = dict()

        for line in raw_source:
            if cell_code[i] and line != "# %matplotlib inline":
                source += [line + "\n"]
            else:
                source += [line[2:] + "\n"]
                
        if source == []:
            continue

        source[-1] = source[-1][:-1]
        new_cell["source"] = source
        out_cells += [new_cell]

    return out_cells

def build_notebook_dict(out_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds final dictionary to be dumped as json."""
    kernelspec: Dict[str, str] = dict()
    kernelspec["display_name"] = "Python Leabra7"
    kernelspec["language"] = "python"
    kernelspec["name"] = "leabra7"

    codemirror_mode: Dict[str, Any] = dict()
    codemirror_mode["name"] = "ipython"
    codemirror_mode["version"] = 3

    language_info: Dict[str, Any] = dict()
    language_info["codemirror_mode"] = codemirror_mode
    language_info["file_extension"] = ".py"
    language_info["mimetype"] = "text/x-python"
    language_info["name"] = "python"
    language_info["nbconvert_exporter"] = "python"
    language_info["pygments_lexer"] = "ipython3"
    language_info["version"] = "3.6.6"

    metadata: Dict[str, Dict[str, Any]] = dict()
    metadata["kernelspec"] = kernelspec
    metadata["language_info"] = language_info

    out_data: Dict[str, Any] = dict()
    out_data["cells"] = out_cells
    out_data["metadata"] = metadata
    out_data["nbformat"] = 4
    out_data["nbformat_minor"] = 2

    return out_data

file_data = sys.stdin.readlines()
file_cells, file_code = break_cells(file_data)
list_cells = build_cell_list(file_cells, file_code)
notebook_dict = build_notebook_dict(list_cells)
json.dump(notebook_dict, sys.stdout, sort_keys=True, indent=2)
