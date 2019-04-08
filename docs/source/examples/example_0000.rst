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

.. image:: ../../doxygen/images/hh_plane.png
