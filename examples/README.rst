
FLORIS Examples
---------------

As FLORIS is a versatile, standalone wake analysis tool, it can be used effectively
in a variety of ways. Some example cases are provided in ``examples/``.

For questions regarding FLORIS, please contact `Jen Annoni <mailto:jennifer.annoni@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.

Input
=====
A sample input file to the Floris model is provided at ``examples/example_inputs.json``.
This example case uses the NREL 5MW turbine and the Gaussian wake model as a reference.
All model parameters provided have been published in previous work, but the inputs to
in the example input file can be changed as needed. However, be aware that changing these parameters
may result in an unphysical solution.

Annotated Usage
===============
``FLORIS_Run_Notebook.ipynb`` is an interactive python notebook with useful notes that details the
execution of FLORIS in normal operating conditions. It also demonstrates how to perform an example
optimization for wake steering with the optimization tools at ``examples/OptModules.py``.
The results of the optimization are based on pre-tuned parameters and the NREL 5MW turbine. 

There is no warranty on the optimization results.

example_script.py
=================
This script provides an example of how to execute Floris.  In particular, this script:

1. Loads the input file ``example_input.json`` and initializes Floris:

::

    floris = Floris("example_input.json")

2. Computes the local power coefficient, thrust coefficients, power, axial induction,
   and wind speeds at each turbine. Here is an example of how to get the local wind speeds at a turbine:

::

    turbine.get_average_velocity())

3. Plot the flow field at a horizontal slice. In this example, the flow field
   is plotted at 50% (0.5) of the total z domain, which is 2x the hub height:

::

    floris.farm.flow_field.plot_z_planes([0.2, 0.5, 0.8])

example_optimization.py
=======================
The Annotated Usage goes into detail on how to run Floris as well as how to set up
an optimization. The ``example_optimization.py`` script allows a user to run an 
optimization in Python rather than through an interactive console.  

The optimial yaw angles are computed using:

::

	opt_yaw_angles = OptModules.wake_steering(floris, minimum_yaw_angle, maximum_yaw_angle)

This script will write out the optimal yaw angles as well as plot the resulting
flow field for the given direction.

Future work
===========
Coming soon.

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