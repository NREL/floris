.. floris documentation master file, created by
   sphinx-quickstart on Mon Jan  8 11:59:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Floris
------

Welcome to FLORIS
=================

.. image:: https://travis-ci.org/WISDEM/FLORIS.svg?branch=develop
        :target: https://travis-ci.org/WISDEM/FLORIS

For questions regarding FLORIS, please contact `Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_, `Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.


Background and objectives
=========================
Coming soon.

Architecture
============
An architecture diagram as in input to `draw.io <https://www.draw.io>`_ is contained in the repository at ``FLORIS/florisarch.xml``.

Generally, a user will not have to write Python code in order to express a wind farm, turbine, or wake model combination. Currently, 
an example wake model, turbine, and wind farm is expressed in ``examples/floris.json``.

Dependencies
============
The following packages are required for FLORIS

- Python3

- NumPy v1.12.1

- SciPy v0.19.1

- matplotlib v2.1.0

- pytest v3.3.1

After installing Python3, the remaining dependencies can be installed with ``pip`` referencing the requirements list using this command:

``pip install -r requirements.txt``

Installation
============
Using ``pip``, FLORIS can be installed in two ways

- local editable install

- using a tagged release version from the ``pip`` repo

For consistency between all developers, it is recommended to use Python virtual environments;
`this link <https://realpython.com/blog/python/python-virtual-environments-a-primer/>`_  provides a great introduction. Using virtual environments in a Jupyter Notebook is described `here <https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/>`_.

Local editable installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The local editable installation allows developers maintain an importable instance of FLORIS while continuing to extend it.
The alternative is to constantly update python paths within the package to match the local environment.

Before doing the local install, the source code repository must be cloned directly from GitHub:

``git clone https://github.com/wisdem/floris``

Then, using the local editable installation is as simple as running the following command from the parent directory of the
cloned repository:

``pip install -e FLORIS/``

Finally, test the installation by starting a python terminal and importing FLORIS:

``import floris``

pip repo installation
~~~~~~~~~~~~~~~~~~~~~
The Floris version available through the pip repository is always the latest tagged and released version.
This version represents the most recent stable, tested, and validated code.

In this case, there is no need to download the source code directly. FLORIS and its dependencies can be installed with:

``pip install floris``

Executing FLORIS
================
``floris`` is an importable package and should be driven by a custom script. We have
provided an example driver script at ``example/example_script.py`` and a Jupyter notebook
detailing a real world use case at ``example/FLORIS_Run_Notebook.ipynb``.

Generally, a ``Floris`` class should be instantiated with a path to an input file
as the sole argument:

``Floris("path/to/floris.json")``

Then, driver programs can calculate the flow field, produce flow field plots,
and incorporate the wake estimation into an optimization routine or other functionality.

Testing
=======

In order to maintain a level of confidence in the software, FLORIS is expected to
maintain a reasonable level of test coverage. To that end, there are unit, integration,
and regression tests included in the package.

Unit tests are currently included in FLORIS and integrated with the `pytest <https://docs.pytest.org/en/latest/>`_
framework.

See the testing documentation at ``tests/README.rst`` for more information.

Future work
===========
Coming soon

License
=======

Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Documentation Navigation
========================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   index.rst
   testindex.rst