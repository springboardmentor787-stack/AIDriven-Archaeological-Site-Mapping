import json
import traceback

try:
    with open('Milestone 3/Week6.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = [("".join(cell.get('source', []))) for cell in nb.get('cells', []) if cell.get('cell_type') == 'code']
    with open('Milestone 3/Week6.ipynb.py', 'w', encoding='utf-8') as f:
        f.write("\n\n# ---\n\n".join(cells))
except Exception as e:
    traceback.print_exc()
