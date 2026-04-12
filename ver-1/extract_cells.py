import json
import glob
import os

def extract(ipynb_path):
    out_path = ipynb_path + ".py"
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            code_cells.append("".join(lines))
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# ---\n\n".join(code_cells))
    print(f"Extracted {out_path}")

try:
    extract("Milestone 2/week4.ipynb")
    extract("Milestone 3/Week5(ErosionDT).ipynb")
except Exception as e:
    print(e)
