FLORIS Wake Modeling Utility
----------------------------

**Further documentation is available at http://floris.readthedocs.io/.**

For technical questions regarding FLORIS usage please post your questions to
`GitHub Discussions <https://github.com/NREL/floris/discussions>`_ on the
FLORIS repository. We no longer plan to actively answer questions on
StackOverflow and will use GitHub Discussions as the main forum for FLORIS.
Alternatively, email the NREL FLORIS team at
`christopher.bay@nrel.gov <mailto:christopher.bay@nrel.gov>`_,
`bart.doekemeijer@nrel.gov <mailto:bart.doekemeijer@nrel.gov>`_,
`rafael.mudafort@nrel.gov <mailto:rafael.mudafort@nrel.gov>`_, or
`paul.fleming@nrel.gov <mailto:paul.fleming@nrel.gov>`_.

.. image:: https://github.com/nrel/floris/workflows/Automated%20tests%20%26%20code%20coverage/badge.svg
  :target: https://github.com/nrel/floris/actions
.. image:: https://codecov.io/gh/nrel/floris/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/nrel/floris
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Background and Objectives
=========================
This FLORIS framework is designed to provide a computationally efficient,
controls-oriented modeling tool of the steady-state wake characteristics in
a wind farm. The wake models implemented in this version of FLORIS are:

- Jensen model for velocity deficit
- Jimenez model for wake deflection
- Gauss model for wake deflection and velocity deficit
- Multi zone model for wake deflection and velocity deficit
- Curl  model for wake deflection and velocity deficit
- TurbOPark model for wake velocity deficit

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

   FLORIS. Version 2.4 (2021). Available at https://github.com/NREL/floris.

For LaTeX users:

.. code-block:: latex

    @misc{FLORIS_2021,
    author = {NREL},
    title = {FLORIS. Version 2.4},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/floris}
    }

.. _installation:

Installation
============
For full installation instructions, see
https://floris.readthedocs.io/en/latest/source/installation.html.

Users who want to run FLORIS without downloading the full source code
can install with `pip` or `conda`, as shown below.

.. code-block:: bash

    # Using pip...
    pip install floris         # Latest version
    pip install floris==1.1.0  # Specified version number

    # Using conda...
    conda install floris        # Latest version
    conda install floris=1.1.0  # Specified version number


To download the source code and use the local code, download the project
and add it to your Python path:

.. code-block:: bash

    # Download the source code.
    git clone https://github.com/NREL/floris.git

    # Install into your Python environment
    pip install -e floris


Finally, users who will be contributing code to the project should align
their environment with the linting and formatting tools used by the
FLORIS development team. This is enabled in the `setup.py` script and
can be activated with these commands:

.. code-block:: bash

    git clone https://github.com/NREL/floris.git -b develop
    cd floris
    pip install -e '.[develop]'
    pre-commit install


After any form of installation, the environment should be tested.
Within a Python shell or a Python script, this code should
display information:

.. code-block:: python

    import floris
    print( help( floris ) )
    print( dir( floris ) )
    print( help( floris.simulation ) )

License
=======

Copyright 2021 NREL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
