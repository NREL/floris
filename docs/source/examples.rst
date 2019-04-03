
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



example_0000_open_and_vis_floris.py
=================
This first example provides an essential introduction to using FLORIS.  A floris model is instantiated,
and a floris interface setup using the example_input.json file.  The model is run using only the 
wind speed and direction specified in the input file and a hub-height visualization is produced.

The first block of code reads in the input file and runs the model without modification

::

    fi = wfct.floris_utilities.FlorisInterface("example_input.json")
    fi.run_floris()

TODO ADD LINKS
Note that run_floris is a wrapper to the calculate_wake function, and so only computes the wakes assuming that changes
since instantation are limited to changes in turbine yaw angle or other control function.  Changes to wind speed, wind direction,
or turbine location require an additional call to reinitialize_flow_field

The second block of code extracts a slice of flow at hub_height using the cut_plane tools

::

    hor_plane = wfct.cut_plane.HorPlane(
        fi.get_flow_field(),
        fi.floris.farm.turbines[0].hub_height
    )


The final block of code visualizes the hub-height plane

::

    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane,ax=ax)
    plt.show()


The results is shown below

.. image:: ../doxygen/images/hh_plane.png

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