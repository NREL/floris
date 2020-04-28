FLORIS Wake Modeling Utility
----------------------------

**Further documentation is available at http://floris.readthedocs.io/.**

For technical questions regarding FLORIS usage please first search for or post
your questions to
`stackoverflow <https://stackoverflow.com/questions/tagged/floris>`_ using
the **floris** tag. Alternatively, email the NREL FLORIS team at
`NREL.Floris@nrel.gov <mailto:floris@nrel.gov>`.

.. image:: https://github.com/nrel/floris/workflows/Automated%20tests%20%26%20code%20coverage/badge.svg
  :target: https://github.com/nrel/floris/actions
.. image:: https://codecov.io/gh/nrel/floris/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/nrel/floris

Background and Objectives
=========================
This FLORIS framework is designed to provide a computationally inexpensive,
controls-oriented modeling tool of the steady-state wake characteristics in
a wind farm. The wake models implemented in this version of FLORIS are:

- Jensen model for velocity deficit
- Jimenez model for wake deflection
- Gauss model for wake deflection and velocity deficit
- Multi zone model for wake deflection and velocity deficit
- Curl  model for wake deflection and velocity deficit

More information on all of these models can be found in the
`theory <https://floris.readthedocs.io/en/develop/source/theory.html>`_
section of the online documentation.

A couple of publications with practical information on using floris as a
modeling and simulation tool for controls research are

1. Annoni, J., Fleming, P., Scholbrock, A., Roadman, J., Dana, S., Adcock, C.,
   Porté-Agel, F, Raach, S., Haizmann, F., and Schlipf, D.: `Analysis of
   control-oriented wake modeling tools using lidar field results <https://www.wind-energ-sci.net/3/819/2018/>`__,
   in: Wind Energy Science, vol. 3, pp. 819-831, Copernicus Publications,
   2018.

Citation
========

.. image:: https://zenodo.org/badge/178914781.svg
  :target: https://zenodo.org/badge/latestdoi/178914781

If FLORIS played a role in your research, please cite it. This software can be
cited as:

   FLORIS. Version 2.0.1 (2020). Available at https://github.com/NREL/floris.

For LaTeX users:

.. code-block:: latex

    @misc{FLORIS_2020,
    author = {NREL},
    title = {{FLORIS. Version 2.0.1},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/floris}
    }

.. _installation:

Installation
============
The FLORIS repository consists of two primary branches:

- `master <https://github.com/NREL/FLORIS/tree/master>`_ - Stable
  release corresponding to a specific version number.
- `develop <https://github.com/NREL/FLORIS/tree/dev>`_ - Latest
  updates including bug fixes and improvements.

These can be cloned (i.e. downloaded) directly from GitHub with one of the
following commands:

.. code-block:: bash

    # master branch
    git clone https://github.com/nrel/floris -b master

    # develop branch
    git clone https://github.com/nrel/floris -b develop

After obtaining the source code, it can be "installed" using ``pip`` or another
Python package manager. With ``pip``, there are two options:

- local editable install
- using a tagged release version from the ``pip`` repo

For consistency between all developers, it is recommended to use Python
virtual environments;
`this link <https://realpython.com/blog/python/python-virtual-environments-a-primer/>`_
provides a great introduction. Using virtual environments in a Jupyter Notebook
is described `here <https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/>`_.

Local Editable Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The local editable installation allows developers to maintain an importable
instance of FLORIS while continuing to extend it. The alternative is to
constantly update python paths within the package to match the local
environment.

Before doing the local install, the source code repository must be cloned
directly from GitHub:

.. code-block:: bash

    git clone https://github.com/nrel/floris

Then, using the local editable installation is as simple as running the
following command from the parent directory of the
cloned repository:

.. code-block:: bash

    pip install -e floris

Finally, test the installation by starting a python terminal and importing
FLORIS:

.. code-block:: bash

    import floris

pip Repo Installation
~~~~~~~~~~~~~~~~~~~~~
The FLORIS version available through the pip repository is typically the latest
tagged and released major version. This version represents the most recent
stable, tested, and validated code.

In this case, there is no need to download the source code directly. FLORIS
and its dependencies can be installed with:

.. code-block:: bash

    pip install floris

Dependencies
============
FLORIS has dependencies on various math, statistics, and plotting libraries in
addition to other general purpose packages. For the simulation and tool
modules, the dependencies are listed in ``floris/requirements.txt``. The
documentation has additional requirements listed in
``floris/docs/requirements.txt``.

The requirements files can be used to install everything with:

.. code-block:: bash

    pip install -r requirements.txt

License
=======

Copyright 2020 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
