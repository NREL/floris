FLORIS Wake Modeling and Wind Farm Controls Software
----------------------------------------------------

FLORIS is a controls-focused wind farm simulation software incorporating
steady-state engineering wake models into a performance focused Python
framework. It has been in active development at NREL since 2013 and the latest
release is `FLORIS v2.4 <https://github.com/NREL/floris/releases/tag/v2.4>`_
in July 2021. The ``v3`` branch of the repository
contains an architectural redesign of the software to enable improved
performance in AEP calculation and controls optimization.

We are actively seeking beta testers for the new framework. If you are interested
in using FLORIS to conduct studies of a wind farm or extending FLORIS to include
your own wake model, please get in touch! Register for beta testing at https://forms.office.com/g/AmpAkJVvja
and join the conversations at `GitHub Discussions <https://github.com/NREL/floris/discussions/categories/v3-design-discussion>`_.

For more context and background on previous work in FLORIS, see the
documentation at http://floris.readthedocs.io/.


Installation
============
Beta testers should install FLORIS v3 by downloading the source code
from GitHub with ``git`` and using ``pip`` to locally install it.
It is recommended to use a Python virtual environment such as `conda <https://docs.conda.io/en/latest/miniconda.html>`_
in order to maintain a clean and sandboxed environment. The following
commands in a terminal or shell will download and install **FLORIS v3.0rc1**.

.. code-block:: bash

    # Download the source code from the `v3.0rc1` tag
    git clone -b v3.0rc1 https://github.com/NREL/floris.git

    # If using conda, be sure to activate your environment prior to installing
    # conda activate <env name>

    # Install into your Python environment
    pip install -e floris

Upon success, the installation can be verified by opening a Python interpreter
and importing FLORIS:

.. code-block:: python

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
        VERSION = '3.0rc1'
        version_file = <_io.TextIOWrapper name='/Users/rmudafor/Development/fl...

    VERSION
        3.0rc1

    FILE
        ~/floris/src/floris/__init__.py


It is important to regularly check for new updates and releases as new
features, improvements, and bug fixes will be issued on an ongoing basis.


Getting Started
===============

A series of examples is included in the `examples/ <https://github.com/NREL/floris/tree/v3.0rc1/examples>`_
directory. These are ordered from simplest to most complex. They demonstrate various
use cases of FLORIS, and generally provide a good starting point for building a more
complex simulation.


License
=======

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
