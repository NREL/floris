"""
Utility script to convert all Python scripts in the current directory to
 Jupyter notebooks.

"""

import os

import nbformat as nbf


def script_to_notebook(script_path, notebook_path):
    # Read Python script
    with open(script_path, "r") as f:
        python_code = f.read()

    # Clear out leading whitespace
    python_code = python_code.strip()

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

    # # Every code block starts with a comment block surrounded by """ and ends with """
    # # Find that block and place it in markdown cell
    # code_comments = python_code.split('"""')[1]

    # # Remove the top line
    # code_comments = code_comments.split("\n")[1:]

    # # Add the code comments
    # nb["cells"].append(nbf.v4.new_markdown_cell(code_comments))

    # # Add Python code to the notebook

    # # Remove the top commented block ("""...""") but keep everything after it
    # python_code = python_code.split('"""')[2]

    # Strip any leading white space
    python_code = python_code.strip()

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


# For each Python script, convert it to a Jupyter notebook
notebook_directories = []
notebook_filenames = []
for script_path in scripts:
    print(f"Converting {script_path} to Notebook...")

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
    toc += f"    - file: examples/{nb}\n"

# For the remaining keys in the notebooks dictionary, first add a section for the directory
# and then add the notebooks in that directory as a file
for directory in notebooks:
    if directory == ".":
        continue
    dir_without_dot_slash = directory[2:]
    dir_without_examples_ = dir_without_dot_slash.replace("examples_", "")
    dir_without_examples_ = dir_without_examples_.replace("_", " ").capitalize()
    toc += f"\n  - caption: Examples - {dir_without_examples_}\n    chapters:\n"
    for nb in notebooks[directory]:
        toc += f"    - file: examples/{dir_without_dot_slash}/{nb}\n"

# Print the toc
print("\n\nTOC: FILE:\n")
print(toc)

# Save the toc
with open(toc_path, "w") as f:
    f.write(toc)
