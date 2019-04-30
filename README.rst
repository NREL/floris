FLORIS Wake Modeling Utility
----------------------------
**Further documentation is available at http://floris.readthedocs.io/.**

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
This FLORIS model is designed to provide a computationally inexpensive,
controls-oriented model of the steady-state wake characteristics in a wind
farm. This can be used for real-time optimization and control. This version of
FLORIS implements a 3D version of the Jensen, original FLORIS (Gebraad et. al.
2016), Gaussian, and Curl wake model.

Literature on the Gaussian model can be found in the following papers:

1. Niayifar, A. and Porté-Agel, F.: A new 15 analytical model for wind farm
   power prediction, in: Journal of Physics: Conference Series, vol. 625,
   p. 012039, IOP Publishing, 2015.

2. Dilip, D. and Porté-Agel, F.: Wind Turbine Wake Mitigation through Blade
   Pitch Offset, Energies, 10, 757, 2017.

3. Abkar, M. and Porté-Agel, F.: Influence of atmospheric stability on
   wind-turbine wakes: A large-eddy simulation study, Physics of Fluids,
   27, 035 104, 2015.

4. Bastankhah, M. and Porté-Agel, F.: A new analytical model for
   wind-turbine wakes, Renewable Energy, 70, 116–123, 2014.

5. Bastankhah, M. and Porté-Agel, 5 F.: Experimental and theoretical study of
   wind turbine wakes in yawed conditions, Journal of FluidMechanics, 806,
   506–541, 2016.

Citation
========

If FLORIS played a role in your research, please cite it. This software can be cited as::

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
