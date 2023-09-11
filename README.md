# FLORIS Wake Modeling and Wind Farm Controls Software

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance-focused Python
framework. It has been in active development at NREL since 2013 and the latest
release is [FLORIS v3.4.1](https://github.com/NREL/floris/releases/latest).
Online documentation is available at https://nrel.github.io/floris.

The software is in active development and engagement with the development team
is highly encouraged. If you are interested in using FLORIS to conduct studies
of a wind farm or extending FLORIS to include your own wake model, please join
the conversation in [GitHub Discussions](https://github.com/NREL/floris/discussions/)!

## Installation

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
        turbine_library (package)
        type_dec
        utilities
        version

    VERSION
        3.4

    FILE
        ~/floris/floris/__init__.py
```

It is important to regularly check for new updates and releases as new
features, improvements, and bug fixes will be issued on an ongoing basis.

## Quick Start

FLORIS is a Python package run on the command line typically by providing
an input file with an initial configuration. It can be installed with
```pip install floris``` (see [installation](https://github.nrel.io/floris/installation)).
The typical entry point is
[FlorisInterface](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface)
which accepts the path to the input file as an argument. From there,
changes can be made to the initial configuration through the
[FlorisInterface.reinitialize](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface.reinitialize)
routine, and the simulation is executed with
[FlorisInterface.calculate_wake](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface.calculate_wake).

```python
from floris.tools import FlorisInterface
fi = FlorisInterface("path/to/input.yaml")
fi.reinitialize(wind_directions=[i for i in range(10)])
fi.calculate_wake()
```

Finally, results can be analyzed via post-processing functions available within
[FlorisInterface](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface)
such as
- [FlorisInterface.get_turbine_layout](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface.get_turbine_layout)
- [FlorisInterface.get_turbine_powers](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface.get_turbine_powers)
- [FlorisInterface.get_farm_AEP](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.floris_interface.FlorisInterface.get_farm_AEP)

and in a visualization package at [floris.tools.visualization](https://nrel.github.io/floris/_autosummary/floris.tools.floris_interface.FlorisInterface.html#floris.tools.visualization).
A collection of examples describing the creation of simulations as well as
analysis and post processing are included in the
[repository](https://github.com/NREL/floris/tree/main/examples)
and described in detail in [Examples Index](https://github.nrel.io/floris/examples).

## Engaging on GitHub

FLORIS leverages the following GitHub features to coordinate support and development efforts:

- [Discussions](https://github.com/NREL/floris/discussions): Collaborate to develop ideas for new use cases, features, and software designs, and get support for usage questions
- [Issues](https://github.com/NREL/floris/issues): Report potential bugs and well-developed feature requests
- [Projects](https://github.com/orgs/NREL/projects/18/): Include current and future work on a timeline and assign a person to "own" it

Generally, the first entry point for the community will be within one of the
categories in Discussions.
[Ideas](https://github.com/NREL/floris/discussions/categories/ideas) is a great spot to develop the
details for a feature request. [Q&A](https://github.com/NREL/floris/discussions/categories/q-a)
is where to get usage support.
[Show and tell](https://github.com/NREL/floris/discussions/categories/show-and-tell) is a free-form
space to show off the things you are doing with FLORIS.


# License

Copyright 2022 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
