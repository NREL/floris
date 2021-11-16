
.. toctree::
    :hidden:
    :glob:
    :titlesonly:

    source/installation
    source/theory
    source/code
    source/inputs
    source/examples
    source/developers
    source/references


FLORIS Wake Modeling Utility
----------------------------
For technical questions regarding FLORIS usage please post your questions to
`GitHub Discussions <https://github.com/NREL/floris/discussions>`_ on the
FLORIS repository. We no longer plan to actively answer questions on
StackOverflow and will use GitHub Discussions as the main forum for FLORIS.
Alternatively, email the NREL FLORIS team at
`christopher.bay@nrel.gov <mailto:christopher.bay@nrel.gov>`_,
`bart.doekemeijer@nrel.gov <mailto:bart.doekemeijer@nrel.gov>`_,
`rafael.mudafort@nrel.gov <mailto:rafael.mudafort@nrel.gov>`_, or
`paul.fleming@nrel.gov <mailto:paul.fleming@nrel.gov>`_.

Background and Objectives
=========================
This FLORIS framework is designed to provide a computationally inexpensive,
controls-oriented modeling tool of the steady-state wake characteristics in
a wind farm. The wake models implemented in this version of FLORIS are:

- Jensen model for velocity deficit
- Jimenez model for wake deflection
- Multi zone model for velocity deficit
- Gaussian models for wake deflection and velocity deficit
- Gauss-Curl-Hybrid (GCH) model for second-order wake steering effects
- Curl  model for wake deflection and velocity deficit
- TurbOPark model for wake velocity deficit

Further, all wake models can now be overlayed onto spatially heterogenous
inflows. More information on all models can be found in :ref:`theory`.

FLORIS further includes a suite of design and analysis tools useful in wind farm
control and co-designed layout optimization.  Examples include:

- Methods for optimization and design of wind farm control and layout
- Visualization methods for flow analysis
- Methods for wind rose and annual energy production analysis
- Methods for analysis of field campaigns of wind farm control
- Coupling methods to other tools, including SOWFA and CC-Blade
- Methods to model heterogenous atmospheric conditions

Example applications of these tools are provided in the `examples/` folder, and
it is highly recommended that new users begin with those in
`examples/_getting_started`.

See :cite:`ind-annoni2018analysis` for practical information on using floris as
a modeling and simulation tool for controls research.

References:
    .. bibliography:: /source/zrefs.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: ind-

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
