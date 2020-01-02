example_0000_open_and_vis_floris.py
===================================

The code for this example can be found here: 
`example_0000_open_and_vis_floris.py
<https://github.com/NREL/floris/blob/develop/examples/example_0000_open_and_vis_floris.py>`_

This first example provides an essential introduction to using FLORIS. A FLORIS 
model is instantiated, and a FLORIS interface is set up using the 
:ref:`example_input.json <sample_input_file_ref>` file.  The model is run using 
only the wind speed and direction specified in the input file and a hub-height 
visualization is produced.

The first block of code reads in the input file and runs the model without 
modification:

.. code-block:: python3

    # Initialize and run FLORIS model
    fi = wfct.floris_interface.FlorisInterface("example_input.json")
    fi.calculate_wake()

Note that :py:meth:`fi.calculate_wake()
<floris.tools.floris_interface.FlorisInterface.calculate_wake>`
is a wrapper to the 
:py:meth:`floris.simulation.flow_field.FlowField.calculate_wake()
<floris.simulation.flow_field.FlowField.calculate_wake>`
function, and so only computes the wakes assuming that changes since 
instantation are limited to changes in turbine yaw angle or other control
function. Changes to wind speed, wind direction, or turbine location require an
additional call to :py:meth:`reinitialize_flow_field()
<floris.tools.floris_interface.FlorisInterface.reinitialize_flow_field>`.

The second block of code extracts a slice of flow at hub_height using the
:py:mod:`cut_plane<floris.tools.cut_plane>` tools:

.. code-block:: python3

    # Initialize the horizontal cut
    hor_plane = wfct.cut_plane.HorPlane(
        fi.get_flow_data(),
        fi.floris.farm.turbines[0].hub_height
    )

Note that flow data is saved flow information in the form of x,y,z,u,v,w.

The final block of code visualizes the hub-height plane:

.. code-block:: python3

    # Plot and show
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane,ax=ax)
    plt.show()

The result is shown below.

.. image:: ../../_static/images/hh_plane.png