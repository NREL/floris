"""Example: Optimize yaw and compare AEP in parallel

This example demonstrates how to perform a yaw optimization and evaluate the performance
over a full wind rose.  The example repeats the steps in 04 except using parallel
optimization and evaluation.

Note that constraints on parallelized operations mean that some syntax is different and
not all operations are possible.  Also, rather passing the ParallelFlorisModel
object to a YawOptimizationSR object, the optimization is performed
directly by member functions

"""

from time import perf_counter as timerpc

import numpy as np

from floris import (
    FlorisModel,
    ParallelFlorisModel,
    TimeSeries,
    WindRose,
)


# When using parallel optimization it is importat the "root" script include this
# if __name__ == "__main__": block to avoid problems
if __name__ == "__main__":

    # Load the wind rose from csv
    wind_rose = WindRose.read_csv_long(
        "../inputs/wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val",
        ti_col_or_value=0.06
    )

    # Load FLORIS
    fmodel = FlorisModel("../inputs/gch.yaml")

    # Specify wind farm layout and update in the floris object
    N = 2  # number of turbines per row and per column
    X, Y = np.meshgrid(
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
    )
    fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten())

    # Get the number of turbines
    n_turbines = len(fmodel.layout_x)

    # Optimize the yaw angles.  This could be done for every wind direction and wind speed
    # but in practice it is much faster to optimize only for one speed and infer the rest
    # using a rule of thumb
    time_series = TimeSeries(
        wind_directions=wind_rose.wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
    )
    fmodel.set(wind_data=time_series)

    # Set up the parallel model
    parallel_interface = "concurrent"
    max_workers = 16
    pfmodel = ParallelFlorisModel(
        fmodel=fmodel,
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True,
    )

    # Get the optimal angles using the parallel interface
    start_time = timerpc()
    # Now optimize the yaw angles using the Serial Refine method
    df_opt = pfmodel.optimize_yaw_angles(
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=20.0,
        Ny_passes=[5, 4],
        exclude_downstream_turbines=False,
    )
    end_time = timerpc()
    t_tot = end_time - start_time
    print("Optimization finished in {:.2f} seconds.".format(t_tot))


    # Calculate the AEP in the baseline case, using the parallel interface
    fmodel.set(wind_data=wind_rose)
    pfmodel = ParallelFlorisModel(
        fmodel=fmodel,
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True,
    )

    # Note the pfmodel does not use run() but instead uses the get_farm_power() and get_farm_AEP()
    # directly, this is necessary for the parallel interface
    aep_baseline = pfmodel.get_farm_AEP(freq=wind_rose.unpack_freq())

    # Now need to apply the optimal yaw angles to the wind rose to get the optimized AEP
    # do this by applying a rule of thumb where the optimal yaw is applied between 6 and 12 m/s
    # and ramped down to 0 above and below this range

    # Grab wind speeds and wind directions from the fmodel.  Note that we do this because the
    # yaw angles will need to be n_findex long, and accounting for the fact that some wind
    # directions and wind speeds may not be present in the wind rose (0 frequency) and aren't
    # included in the fmodel
    wind_directions = fmodel.wind_directions
    wind_speeds = fmodel.wind_speeds
    n_findex = fmodel.n_findex


    # Now define how the optimal yaw angles for 8 m/s are applied over the other wind speeds
    yaw_angles_opt = np.vstack(df_opt["yaw_angles_opt"])
    yaw_angles_wind_rose = np.zeros((n_findex, n_turbines))
    for i in range(n_findex):
        wind_speed = wind_speeds[i]
        wind_direction = wind_directions[i]

        # Interpolate the optimal yaw angles for this wind direction from df_opt
        id_opt = df_opt["wind_direction"] == wind_direction
        yaw_opt_full = np.array(df_opt.loc[id_opt, "yaw_angles_opt"])[0]

        # Now decide what to do for different wind speeds
        if (wind_speed < 4.0) | (wind_speed > 14.0):
            yaw_opt = np.zeros(n_turbines)  # do nothing for very low/high speeds
        elif wind_speed < 6.0:
            yaw_opt = yaw_opt_full * (6.0 - wind_speed) / 2.0  # Linear ramp up
        elif wind_speed > 12.0:
            yaw_opt = yaw_opt_full * (14.0 - wind_speed) / 2.0  # Linear ramp down
        else:
            yaw_opt = yaw_opt_full  # Apply full offsets between 6.0 and 12.0 m/s

        # Save to collective array
        yaw_angles_wind_rose[i, :] = yaw_opt


    # Now apply the optimal yaw angles and get the AEP
    fmodel.set(yaw_angles=yaw_angles_wind_rose)
    pfmodel = ParallelFlorisModel(
        fmodel=fmodel,
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True,
    )
    aep_opt = pfmodel.get_farm_AEP(freq=wind_rose.unpack_freq(), yaw_angles=yaw_angles_wind_rose)
    aep_uplift = 100.0 * (aep_opt / aep_baseline - 1)

    print("Baseline AEP: {:.2f} GWh.".format(aep_baseline/1E9))
    print("Optimal AEP: {:.2f} GWh.".format(aep_opt/1E9))
    print("Relative AEP uplift by wake steering: {:.3f} %.".format(aep_uplift))
