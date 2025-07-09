#!/usr/bin/env python3
"""
Convert v1_tutorial_notebook.py to a proper Jupyter notebook format
"""

import json

def convert_to_notebook():
    # Read the notebook-style Python file
    with open('v1_tutorial_notebook.py', 'r') as f:
        content = f.read()

    # Simple conversion to notebook format
    cells = []
    current_cell = {'cell_type': 'code', 'source': [], 'metadata': {}}

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('# %% [markdown]'):
            # Save current cell if it has content
            if current_cell['source']:
                cells.append(current_cell)
            
            # Start new markdown cell
            current_cell = {'cell_type': 'markdown', 'source': [], 'metadata': {}}
            i += 1
            
            # Collect markdown content
            while i < len(lines) and not lines[i].startswith('# %%'):
                if lines[i].startswith('# '):
                    current_cell['source'].append(lines[i][2:])
                else:
                    current_cell['source'].append(lines[i])
                i += 1
                
        elif line.startswith('# %%'):
            # Save current cell if it has content
            if current_cell['source']:
                cells.append(current_cell)
            
            # Start new code cell
            current_cell = {
                'cell_type': 'code', 
                'source': [], 
                'metadata': {}, 
                'execution_count': None, 
                'outputs': []
            }
            i += 1
            
        else:
            current_cell['source'].append(line)
            i += 1

    # Add the last cell
    if current_cell['source']:
        cells.append(current_cell)

    # Create notebook structure
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.5'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    # Write the notebook
    with open('V1_Training_Tutorial.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)

    print('Notebook created: V1_Training_Tutorial.ipynb')

if __name__ == "__main__":
    convert_to_notebook() 