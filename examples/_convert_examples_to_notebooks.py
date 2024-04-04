"""
Utility script to convert all Python scripts in the current directory to
 Jupyter notebooks and then to HTML files.

"""

import os

import nbformat as nbf


def script_to_notebook(script_path, notebook_path):
    # Read Python script
    with open(script_path, "r") as f:
        python_code = f.read()

    # Append to the bottom of the code suppression of warnings
    python_code += """
import warnings
warnings.filterwarnings('ignore')
"""

    # Create a new Jupyter notebook
    nb = nbf.v4.new_notebook()

    # The first line of code it the title, copy it, remove and
    # leading quotes or comments and make it a markdown cell with one hash
    title = python_code.split("\n")[0].strip().strip("#").strip().strip('"').strip().strip("'")
    nb["cells"].append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Add Python code to the notebook
    nb["cells"].append(nbf.v4.new_code_cell(python_code))

    # Write the notebook to a file
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)


# Traverse the current directory and subdirectories to find
# all python scripts that start with a number
# and end with .py and make a list of all such scripts including relative path
scripts = sorted(
    [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(".")
        for f in filenames
        if f.endswith(".py") and f[0].isdigit()
    ]
)


# For each Python script, convert it to a Jupyter notebook, run it, and then convert it to HTML
notebook_directories = []
notebook_filenames = []
for script_path in scripts:
    print(f"Converting {script_path} to HTML...")

    notebook_path = script_path.replace(".py", ".ipynb")
    notebook_directories.append(os.path.dirname(notebook_path))
    notebook_filenames.append(os.path.basename(notebook_path))

    script_to_notebook(script_path, notebook_path)


# Make a dictionary of all the notebooks, whose keys are
# unique entries in the notebook_directories list
# and values are lists of notebook filenames in that directory
notebooks = {k: [] for k in notebook_directories}
for i, directory in enumerate(notebook_directories):
    notebooks[directory].append(notebook_filenames[i])

print(notebooks)

# Now read in the _toc.yaml file one level up and add each of the note books to a new chapter
# called examples and re-write the _toc.yaml file
toc_path = "../_toc.yml"

# Load the toc file as a file
with open(toc_path, "r") as f:
    toc = f.read()

# Append a blank line and then "  - caption: Developer Reference" to the toc
toc += "\n  - caption: Examples\n    chapters:\n"

# For each entry in the '.' directory, add it to the toc as a file
for nb in notebooks["."]:
    toc += f"      - file: examples/{nb}\n"

# For the remaining keys in the notebooks dictionary, first add a section for the directory
# and then add the notebooks in that directory as a file
for directory in notebooks:
    if directory == ".":
        continue
    toc += "      sections:"
    for nb in notebooks[directory]:
        toc += f"      - file: examples/{directory}/{nb}\n"

# Save the toc
with open(toc_path, "w") as f:
    f.write(toc)
