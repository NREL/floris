
Examples
---------

The FLORIS code includes wake models, and a number of related analysis and visualization tools to be used in
connection with wind farm controls research.  A number of examples are provided in the directory ``examples/``
to provide instruction on the use of most of the underlying codes.

For questions not covered in the examples, or to request additional examples, please first search for or 
submit your questions to stackoverflow.com using the tag FLORIS.  Additionally you can contact 
 `Jen King <mailto:jennifer.king@nrel.gov>`_,
`Paul Fleming <mailto:paul.fleming@nrel.gov>`_, or `Rafael Mudafort <mailto:rafael.mudafort@nrel.gov>`_.



FLORIS Input
=====
A sample input file to the Floris model is provided in ``examples/example_input.json``.
This example case uses the NREL 5MW turbine and the Gaussian wake model as a reference.
All model parameters provided have been published in previous work, but the inputs to
in the example input file can be changed as needed. However, be aware that changing these parameters
may result in an unphysical solution.  Many of the example files will make use of this example input.



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