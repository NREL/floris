(installation)=
# Installation

FLORIS can be installed by downloading the source code or via the PyPI package manager with `pip`.
The following sections detail how download and install FLORIS for each use case.

(requirements)=
## Requirements

FLORIS is intended to be used with Python 3.8 and up, and it is highly recommended that users
work within a virtual environment for both working with and working on FLORIS, to maintain a clean
and sandboxed environment. The simplest way to get started with virtual environments is through
[conda](https://docs.conda.io/en/latest/miniconda.html).

Installing into a Python environment that contains a previous version of FLORIS may cause conflicts.
If you intend to use [pyOptSparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/)
with FLORIS, it is recommended to install that package first before installing FLORIS.


```{note}
If upgrading, it is highly recommended to install FLORIS v4 into a new virtual environment.
```

(pip)=
## Pip

The simplest method is with `pip` by using this command:

```bash
pip install floris
```

(source)=
## Source Code Installation

Developers and anyone who intends to inspect the source code or wants to run examples can install FLORIS by downloading the
git repository from GitHub with ``git`` and use ``pip`` to locally install it. The following commands in a terminal or shell will download and install FLORIS.

```bash
# Download the source code from the `main` branch
git clone -b main https://github.com/NREL/floris.git

# If using conda, be sure to activate your environment prior to installing
# conda activate <env name>

# If using pyOptSpare, install it first
conda install -c conda-forge pyoptsparse

# Install FLORIS
pip install -e floris
```

With both methods, the installation can be verified by opening a Python interpreter
and importing FLORIS:

```python
>>> import floris
>>> help(floris)

Help on package floris:

NAME
    floris - # Copyright 2024 NREL

PACKAGE CONTENTS
    convert_floris_input_v3_to_v4
    convert_turbine_v3_to_v4
    core (package)
    cut_plane
    floris_model
    flow_visualization
    layout_visualization
    logging_manager
    optimization (package)
    parallel_floris_model
    turbine_library (package)
    type_dec
    uncertain_floris_model
    utilities
    version
    wind_data

VERSION
    4.0

FILE
    ~/floris/floris/__init__.py
```

(developers)=
## Developer Installation

For users that will also be contributing to the FLORIS code repository, the process is similar to
the source code installation, but with a few extra considerations. The steps are laid out in our
[developer's guide](dev_guide.md).

(updating)=
## Updating FLORIS

It is important to regularly check for new updates and releases as new features, improvements, and
bug fixes will be issued on an ongoing basis, and will require manually updating the software.

(pip-update)=
### Pip

```bash
pip install --upgrade floris

# Alternatively, users can specify a particular version, for example:
# pip install --upgrade floris==3.2.1
```

(source-update)=
### From Source
```bash

# If you're not already on the main branch, save your changes and move there
git checkout main

# Pull down the changes from GitHub
git pull main
```
