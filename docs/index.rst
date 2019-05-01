
.. toctree::
    :hidden:
    :glob:
    :titlesonly:

    source/theory
    source/code
    source/inputs
    source/examples
    source/developers
    source/changelog


FLORIS Wake Modeling Utility
----------------------------
For technical questions regarding FLORIS usage please first search for or post
your questions to
`stackoverflow <https://stackoverflow.com/questions/tagged/floris>`_ using
the **floris** tag. Alternatively, please contact
`Jen King <mailto:jennifer.king@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_,
`Chris Bay <mailto:chris.bay@nrel.gov>`_, and
`Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.

Background and objectives
=========================
This FLORIS framework is designed to provide a computationally inexpensive,
controls-oriented modeling tool of the steady-state wake characteristics in
a wind farm. The wake models implement in this version of FLORIS are:

- Jensen model for velocity deficit
- Jimenez model for wake deflection
- Gauss model for wake deflection and velocity deficit
- Multi zone model for wake deflection and velocity deficit
- Curl  model for wake deflection and velocity deficit

More information on all of these models can be found in the
:ref:`theory <theory>` section of the online documentation.

A couple of publications with practical information on using floris as a
modeling and simulation tool for controls research are

- Jen paper
- Chris paper

Citation
========

If FLORIS played a role in your research, please cite it. This software can be
cited as:

   FLORIS. Version X.Y.Z (2019). Available at https://github.com/nrel/floris.

For LaTeX users:

.. code-block:: latex

    @misc{FLORIS_2019,
    author = {NREL},
    title = {{FLORIS. Version X.Y.Z}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/floris}
    }

.. _installation:

Installation
============
Using ``pip``, FLORIS can be installed in two ways

- local editable install

- using a tagged release version from the ``pip`` repo

For consistency between all developers, it is recommended to use Python
virtual environments;
`this link <https://realpython.com/blog/python/python-virtual-environments-a-primer/>`_
provides a great introduction. Using virtual environments in a Jupyter Notebook
is described `here <https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/>`_.

Local editable installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The local editable installation allows developers maintain an importable
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

pip repo installation
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

Copyright 2019 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
