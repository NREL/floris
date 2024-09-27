"""Example: Timing tests for parallel computation interfaces.

Tests:
- max_workers specified, small.
- max_workers specified, large.
- max_workers unspecified.

- various n_findex
- various n_turbines

- return_turbine_powers_only=True
- return_turbine_powers_only=False
"""

from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris import (
    FlorisModel,
    TimeSeries,
)
from floris.parallel_floris_model import ParallelFlorisModel as ParallelFlorisModel_orig
from floris.parallel_floris_model_2 import ParallelFlorisModel as ParallelFlorisModel_new


DEBUG = True

if __name__ == "__main__":
    max_workers_options = [2, 16, -1]
    n_findex_options = [100, 1000, 10000]
    n_turbines_options = [5, 10, 15] # Will be squared!
    # Parallelization parameters

    def set_up_and_run_models(n_turbs, n_findex, max_workers):
        # Create random wind data
        np.random.seed(0)
        wind_speeds = np.random.normal(loc=8.0, scale=2.0, size=n_findex)
        wind_directions = np.random.normal(loc=270.0, scale=15.0, size=n_findex)
        turbulence_intensities = 0.06*np.ones_like(wind_speeds)

        time_series = TimeSeries(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensities=turbulence_intensities,
        )

        # Clip wind_rose to specified n_findex

        fmodel = FlorisModel("../inputs/gch.yaml")

        # Specify wind farm layout and update in the floris object
        N = n_turbs

        X, Y = np.meshgrid(
            5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
            5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
        )
        fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten())

        # Set up original parallel Floris model
        parallel_interface = "multiprocessing"
        pfmodel_orig = ParallelFlorisModel_orig(
            fmodel=fmodel,
            max_workers=100 if max_workers < 0 else max_workers,
            n_wind_condition_splits=100 if max_workers < 0 else max_workers,
            interface=parallel_interface,
            print_timings=True
        )

        # Set up new parallel Floris model
        pfmodel_new = ParallelFlorisModel_new(
            "../inputs/gch.yaml",
            max_workers=max_workers,
            n_wind_condition_splits=max_workers,
            interface="pathos",
            print_timings=True,
        )

        # Set up new parallel Floris model using only powers
        pfmodel_new_p = ParallelFlorisModel_new(
            "../inputs/gch.yaml",
            max_workers=max_workers,
            n_wind_condition_splits=max_workers,
            interface=parallel_interface,
            return_turbine_powers_only=True,
            print_timings=True,
        )

        # Set layout, wind data on all models
        fmodel.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)
        pfmodel_orig.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)
        pfmodel_new.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)
        pfmodel_new_p.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)

        # Limit to a subset of the wind rose, maybe.


        # Run and evaluate farm over the wind rose
        t0 = timerpc()
        fmodel.run()
        aep_fmodel = fmodel.get_farm_AEP()
        t_fmodel = timerpc() - t0

        t0 = timerpc()
        #pfmodel_orig.run()
        aep_pfmodel_orig = pfmodel_orig.get_farm_AEP(freq=time_series.unpack_freq())
        t_pfmodel_orig = timerpc() - t0

        t0 = timerpc()
        pfmodel_new.run()
        aep_pfmodel_new = pfmodel_new.get_farm_AEP()
        t_pfmodel_new = timerpc() - t0

        t0 = timerpc()
        pfmodel_new_p.run()
        aep_pfmodel_new_p = pfmodel_new_p.get_farm_AEP()
        t_pfmodel_new_p = timerpc() - t0

        # Save the data
        df = pd.DataFrame({
            "model": ["FlorisModel", "ParallelFlorisModel_orig", "ParallelFlorisModel_new",
                      "ParallelFlorisModel_new_poweronly"],
            "AEP": [aep_fmodel, aep_pfmodel_orig, aep_pfmodel_new, aep_pfmodel_new_p],
            "time": [t_fmodel, t_pfmodel_orig, t_pfmodel_new, t_pfmodel_new_p],
        })

        df.to_csv(f"comptime_maxworkers{mw}_nturbs{n_turbs}_nfindex{n_findex}.csv")

        return None

    # First run max_workers tests
    for mw in max_workers_options:
        # Set up models
        n_turbs = 2 if DEBUG else 10 # Will be squared
        n_findex = 1000
        set_up_and_run_models(
            n_turbs=n_turbs, n_findex=n_findex, max_workers=mw
        )

    # Then run n_turbines tests
    for nt in n_turbines_options:
        # Set up models
        n_findex = 10 if DEBUG else 1000
        max_workers = 16

        set_up_and_run_models(
            n_turbs=nt, n_findex=n_findex, max_workers=max_workers
        )

    # Then run n_findex tests
    for nf in n_findex_options:
        # Set up models
        n_turbs = 2 if DEBUG else 10 # Will be squared
        max_workers = 16

        set_up_and_run_models(
            n_turbs=n_turbs, n_findex=nf, max_workers=max_workers
        )

