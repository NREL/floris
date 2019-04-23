example_0005_adjust_floris.py
=============================

The code for this example can be found here: `example_0005_adjust_floris.py <https://github.com/NREL/floris/blob/develop/examples/example_0005_adjust_floris.py>`_

In this example, the FLORIS model is adjusted within the code and provides examples of how to make various adjustments.

The floris model and interface are initially instantiated as before but then the number of turbines and their locations are changed 
via the line:

::

    fi.floris.farm.set_turbine_locations(layout_x, layout_y, calculate_wake=True)

Note that by setting the turbine locations using the function :py:meth:`set_turbine_locations()<floris.simulation.farm.Farm.set_turbine_locations>`, 
the flow field is automatically reinitialized because the turbine points need to be re-assigned. 
:py:meth:`Calculate_wake<floris.simulation.flow_field.FlowField.calculate_wake>` is optionally set to *True* (equivalent to running 
:py:meth:`fi.run_floris()<floris.tools.floris_utilities.FlorisInterface.run_floris>` or 
:py:meth:`fi.floris.farm.flow_field.calculate_wake()<floris.simulation.flow_field.FlowField.calculate_wake>` later). 
This run is considered the baseline and the initial farm power is computed in the line:

::

    power_initial = np.sum(fi.get_turbine_power())


The next block cycles through wind speed and wind directions and updates the FLORIS model by first reinitlizing the flow field and 
then recalculating the wakes.

::

    for i,speed in enumerate(ws):
        for j,wdir in enumerate(wd):
            print('Calculating wake: wind direction = ', wdir, 'and wind speed = ', speed)

            fi.floris.farm.flow_field.reinitialize_flow_field(wind_speed=speed,
                                                                            wind_direction=wdir,

                                                                            # keep these the same
                                                                            wind_shear=fi.floris.farm.flow_field.wind_shear,
                                                                            wind_veer=fi.floris.farm.flow_field.wind_veer,
                                                                            turbulence_intensity=fi.floris.farm.flow_field.turbulence_intensity,
                                                                            air_density=fi.floris.farm.flow_field.air_density,
                                                                            wake=fi.floris.farm.flow_field.wake,
                                                                            turbine_map=fi.floris.farm.flow_field.turbine_map)
            # recalculate the wake
            fi.run_floris()


These individual runs are visualized in sub plots.

The final block of code changes the turbine yaw angles.

::

    fi.floris.farm.set_yaw_angles(yaw_angles, calculate_wake=True)
    power_yaw = np.sum(fi.get_turbine_power())


Note that if only changing yaw angles it is not necessary to reinitialize the flow field; however, before collecting the power
it is necessary either to recalulate the wake within the update to the yaw angles (as is done here), or through a call to 
:py:meth:`fi.run_floris()<floris.tools.floris_utilities.FlorisInterface.run_floris>` or 
:py:meth:`fi.floris.farm.flow_field.calculate_wake()<floris.simulation.flow_field.FlowField.calculate_wake>`.
