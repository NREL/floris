example_0010_optimize_yaw.py
============================

The code for this example can be found here: `example_0010_optimize_yaw.py 
<https://github.com/NREL/floris/blob/develop/examples/example_0010_optimize_yaw.py>`_

This example uses the :py:meth:`optimize_yaw()<floris.tools.optimization.optimize_yaw>` 
method to determine the optimial yaw angles for a given wind farm for a single 
wind speed and direction. The optimization function accepts the floris 
interface instance and bounds on the yaw angles, and returns the optimal angles.

The initial setup and power output are computed here:

::

    # Set turbine locations to 3 turbines in a row
    D = fi.floris.farm.turbines[0].rotor_diameter
    layout_x = [0,7*D,14*D]
    layout_y = [0,0,0]
    fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
    fi.calculate_wake()

    # Initial power output
    power_initial = np.sum(fi.get_turbine_power())

The optimization is then called here:

::

    min_yaw = 0.0
    max_yaw = 25.0
    yaw_angles = optimize_yaw(fi,min_yaw,max_yaw)


To determine the gain, the power is compared using the initial angles and the 
optimal angles.