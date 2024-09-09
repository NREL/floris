"""Example: Compare parallel interfaces
"""

from time import perf_counter as timerpc

import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.parallel_floris_model import ParallelFlorisModel as ParallelFlorisModel_orig
from floris.parallel_floris_model_2 import ParallelFlorisModel as ParallelFlorisModel_new


if __name__ == "__main__":
    # Parallelization parameters
    parallel_interface = "multiprocessing"
    max_workers = 16

    # Load the wind rose from csv
    wind_rose = WindRose.read_csv_long(
        "inputs/wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val",
        ti_col_or_value=0.06
    )
    fmodel = FlorisModel("inputs/gch.yaml")

    # Specify wind farm layout and update in the floris object
    N = 12  # number of turbines per row and per column
    X, Y = np.meshgrid(
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
    )
    fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten())

    # Set up original parallel Floris model
    pfmodel_orig = ParallelFlorisModel_orig(
        fmodel=fmodel,
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True
    )


    # Set up new parallel Floris model
    pfmodel_new = ParallelFlorisModel_new(
        "inputs/gch.yaml",
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        print_timings=True,
    )

    # Set up new parallel Floris model using only powers
    pfmodel_new_p = ParallelFlorisModel_new(
        "inputs/gch.yaml",
        max_workers=max_workers,
        n_wind_condition_splits=max_workers,
        interface=parallel_interface,
        return_turbine_powers_only=True,
        print_timings=True,
    )

    # Set layout, wind data on all models
    fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)
    pfmodel_orig.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)
    pfmodel_new.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)
    pfmodel_new_p.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=wind_rose)

    # Run and evaluate farm over the wind rose
    t0 = timerpc()
    fmodel.run()
    aep_fmodel = fmodel.get_farm_AEP()
    t_fmodel = timerpc() - t0

    t0 = timerpc()
    #pfmodel_orig.run()
    aep_pfmodel_orig = pfmodel_orig.get_farm_AEP(freq=wind_rose.unpack_freq())
    t_pfmodel_orig = timerpc() - t0

    t0 = timerpc()
    pfmodel_new.run()
    aep_pfmodel_new = pfmodel_new.get_farm_AEP()
    t_pfmodel_new = timerpc() - t0

    t0 = timerpc()
    pfmodel_new_p.run()
    aep_pfmodel_new_p = pfmodel_new_p.get_farm_AEP()
    t_pfmodel_new_p = timerpc() - t0

    print("FlorisModel AEP calculation took {:.2f} seconds.".format(t_fmodel))
    print("Original ParallelFlorisModel AEP calculation took {:.2f} seconds.".format(
            t_pfmodel_orig
        )
    )
    print("New ParallelFlorisModel AEP calculation took {:.2f} seconds.".format(t_pfmodel_new))
    print("New ParallelFlorisModel (powers only) AEP calculation took {:.2f} seconds.".format(
            t_pfmodel_new_p
        )
    )

    print("\n")
    print("FlorisModel AEP: {:.2f} GWh.".format(aep_fmodel/1E9))
    print("Original ParallelFlorisModel AEP: {:.2f} GWh.".format(aep_pfmodel_orig/1E9))
    print("New ParallelFlorisModel AEP: {:.2f} GWh.".format(aep_pfmodel_new/1E9))
    print("New ParallelFlorisModel (powers only) AEP: {:.2f} GWh.".format(aep_pfmodel_new/1E9))
