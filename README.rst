
FLORIS Wake Modeling Utility
----------------------------

**Further documentation is available at http://floris.readthedocs.io/.**

For questions regarding FLORIS, please contact `Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_, `Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_,
or join the conversation on our `Slack team <https://join.slack.com/t/floris-nwtc/shared_invite/enQtMzMzODczNzE2NTAwLTYyZTcyZDVmODA5NDFmYzNmZmY0YzNjZTQwNTYxMzkyMGE1YWE0ZTBmNWRmNGI3NTZmZjFjMTljYWMxNzM4MmI>`_.

Citation
========

If FLORIS played a role in your research, please cite it. This software can be cited as::

   FLORIS. Version X.Y.Z (2018). Available at https://github.com/wisdem/floris.

Dependencies
============
The following packages are used in FLORIS

- Python3

- NumPy v1.12.1

- SciPy v0.19.1

- matplotlib v2.1.0

- pytest v3.3.1 (optional)

- Sphinx v1.6.6 (optional)

After installing Python3, the remaining required dependencies can be installed with ``pip`` referencing the requirements list using this command:

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

``Floris("path/to/example_input.json")``

Then, driver programs can calculate the flow field, produce flow field plots,
and incorporate the wake estimation into an optimization routine or other functionality.

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
