# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""
converts simple example files into rst files, to then be converted to html for documentation.

commandline args:
    arg[0]: Name of this Python script
    arg[1]: Name of Python script to convert
    arg[2]: Name of output rst file
    arg[3]: Number of headerlines in input python file
"""

import sys


# get command line arguments
inputs = sys.argv

# default values
infile = "dummy.py"
outfile = "dummy.rst"
headerlines = 13
# assign input values
if len(inputs) > 1:
    infile = inputs[1]
    outfile = infile.split(".")[0] + ".rst"
if len(inputs) > 2:
    outfile = inputs[2]
if len(inputs) > 3:
    headerlines = int(inputs[3])

# read file contents as a list
f = open(infile, "r")
fileContents = f.readlines()

# remove copyright/headerlines
for ii, x in enumerate(fileContents[: headerlines + 1]):
    fileContents.remove(x)

# block of text to add to signify codeblocks for rst
codeblock = "\n.. code-block:: python3 \n\n"
# flag to determine indentation
codesec = False
commentsec = False

# make title and section header
title = infile.split("/")[-1]
fileContents.insert(0, title + " \n")
fileContents.insert(1, "".join(["="] * len(title) + [" \n\n"]))

# append 'EOF' to filecontents to signify end of list
fileContents += ["EOF"]

flag = None
ii = 2
while flag is not "EOF":
    # detect beginning/end of docstring
    if '"""' in fileContents[ii]:
        fileContents[ii] = "\n"

        if commentsec == False:
            commentsec = True
        else:
            commentsec = False

    # prepend lines in docstring with hash
    if commentsec:
        fileContents[ii] = "# " + fileContents[ii]

    # lines of code do NOT start with a hash or newline
    if fileContents[ii][0] not in [
        "#",
        "\n",
    ]:
        commentsec = False
        if not codesec:
            fileContents.insert(ii, codeblock)
            codesec = True
        else:
            fileContents[ii] = "\t" + fileContents[ii]
            codesec = True

    # add new line to end of code section
    elif codesec and fileContents[ii][0] in ["#", "\n"]:
        # fileContents.insert(ii, '\n')
        codesec = False

    # comment lines DO start with a hash
    else:
        fileContents[ii] = fileContents[ii][2:]
        codesec = False

    ii += 1
    flag = fileContents[ii]

# trim off 'EOF'
dump = fileContents.pop(-1)

# write to rst file
with open(outfile, "w") as filehandle:
    for item in fileContents:
        filehandle.write(item)
