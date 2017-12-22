
FLORIS
------
The ``develop`` branch of the FLORIS repo represents a rework of the software as a general wind farm wake analysis tool.

For questions regarding FLORIS, please contact `Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_, `Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.

**NOTE: This FLORIS repository is under active development. Be aware that the implementation is not yet validated or verified.**

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


After installing Python3, the remaining dependencies can be installed with ``pip`` referencing the requirements list using this command:

``pip install -r requirements.txt``

Installation
============
Using ``pip``, FLORIS can be installed in two ways

- local editable install

- using a tagged release version from the ``pip`` repo

For consistency between all developers, it is recommended to use Python virtual environments;
`this link <https://realpython.com/blog/python/python-virtual-environments-a-primer/>`_ provides a great introduction.

Local editable installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The local editable installation allows developers maintain an importable instance of floris while continuing to extending it.
The alternative is to constantly update python paths within the package to match the local environment.

Before doing the local install, the source code repository must be cloned directly from GitHub:

``git clone https://github.com/wisdem/floris``

Then, using the local editable installation is as simple as running the following command from the parent directory of the
cloned repository:

``pip install -e FLORIS/``

Finally, test the installation by starting a python terminal and importing floris:

``import floris``

pip repo installation
~~~~~~~~~~~~~~~~~~~~~~~
The floris version available through the pip repository is always the latest tagged and released version.
This version represents the most recent stable, tested, and validated code.

In this case, there is no need to download the source code directly. It can be installed with:

``pip install floris``

Executing FLORIS
================
Currently, FLORIS can be executed by simply running ``FLORIS.py``:

``python3 FLORIS.py``

The FLORIS module can also be imported into other projects to extend the current capability.

Future work
===========
- improve the swept area point sampling to consistently include more points

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
