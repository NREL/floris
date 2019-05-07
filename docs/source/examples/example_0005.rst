example_0005_adjust_floris.py
=============================

The code for this example can be found here: `example_0005_adjust_floris.py
<https://github.com/NREL/floris/blob/develop/examples/example_0005_adjust_floris.py>`_

In this example, the FLORIS model is adjusted within the code and provides
examples of how to make various adjustments.

The FLORIS model and interface are initially instantiated as before, but then
the number of turbines and their locations are changed via the line:

.. code-block:: python3

    # Assign turbine locations
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

    # Calculate wake
    fi.calculate_wake()

Note that since changing the turbine layout requires calling the simulation 
function :py:meth:`calculate_wake() 
<floris.simulation.flow_field.FlowField.reinitialize_flow_field>`,
the interface function to change the layout (as well as wind speed, 
direction, TI...) is made through this function's wrapper.

The next code block cycles through wind speed and wind directions and updates 
the FLORIS model by first reinitializing the flow field and then recalculating the wakes.

.. code-block:: python3

    for i,speed in enumerate(ws):
        for j,wdir in enumerate(wd):
            print('Calculating wake: wind direction = ', wdir, 'and wind
            speed = ', speed)

            # Update the flow field
            fi.reinitialize_flow_field(wind_speed=speed,wind_direction=wdir)

            # Recalculate the wake
            fi.calculate_wake()

            # Record powers
            power[i,j] = np.sum(fi.get_turbine_power())


These individual runs are visualized in sub plots.

The final block of code changes the turbine yaw angles.

.. code-block:: python3

    fi.calculate_wake()
    power_initial = np.sum(fi.get_turbine_power())

    # Set the yaw angles
    yaw_angles = [25.0,0,25.0,0]
    fi.calculate_wake(yaw_angles=yaw_angles)

    # Check the new power
    power_yaw = np.sum(fi.get_turbine_power())
    print('Power aligned: %.1f' % power_initial)
    print('Power yawed: %.1f' % power_yaw)


Note that if only changing yaw angles it is not necessary to reinitialize the
flow field; therefore the update is made through the calculate_wake function.
