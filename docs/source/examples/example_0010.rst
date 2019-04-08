example_0010_optimize_yaw.py
============================

This function uses the optimize_yaw function to determine the optimial yaw angles for a given wind farm for a single wind speed and
direction.  The optimization function accepts the floris interface instance, and bounds and yaw angles, and returns the optimal angles

::

    min_yaw = 0.0
    max_yaw = 25.0
    yaw_angles = optimize_yaw(fi,min_yaw,max_yaw)


To determine the gain, the power is read using the initial angles and the optimal angles