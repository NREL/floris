(installation)=
# Installation

**If upgrading from v2, it is highly recommended to install FLORIS V3 into a new virtual environment**.
Installing into a Python environment that contains FLORIS v2 may cause conflicts.
If you intend to use [pyOptSparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/) with FLORIS,
it is recommended to install that package first before installing FLORIS.

FLORIS can be installed by downloading the source code or via the PyPI
package manager with `pip`.

The simplest method is with `pip` by using this command:

```bash
pip install floris
```

Developers and anyone who intends to inspect the source code
can install FLORIS by downloading the git repository
from GitHub with ``git`` and use ``pip`` to locally install it.
It is highly recommended to use a Python virtual environment manager
such as [conda](https://docs.conda.io/en/latest/miniconda.html)
in order to maintain a clean and sandboxed environment. The following
commands in a terminal or shell will download and install FLORIS.

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
    floris - # Copyright 2021 NREL

PACKAGE CONTENTS
    logging_manager
    simulation (package)
    tools (package)
    type_dec
    utilities

DATA
    ROOT = PosixPath('/Users/rmudafor/Development/floris')
    VERSION = '3.2'
    version_file = <_io.TextIOWrapper name='/Users/rmudafor/Development/fl...

VERSION
    3.2

FILE
    ~/floris/floris/__init__.py
```

It is important to regularly check for new updates and releases as new
features, improvements, and bug fixes will be issued on an ongoing basis.
