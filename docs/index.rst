
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
For technical questions regarding FLORIS usage please first search for or post
your questions to
`stackoverflow <https://stackoverflow.com/questions/tagged/floris>`_ using
the **floris** tag. Alternatively, email the NREL FLORIS team at
`NREL.Floris@nrel.gov <mailto:floris@nrel.gov>`_.

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

   FLORIS. Version 2.2.2 (2020). Available at https://github.com/NREL/floris.

For LaTeX users:

.. code-block:: latex

    @misc{FLORIS_2020,
    author = {NREL},
    title = {{FLORIS. Version 2.2.2}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/NREL/floris}
    }


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
