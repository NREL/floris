"""Example: Timing tests for parallel computation interfaces.

Tests:
- "multiprocessing" vs "pathos" interfaces.
"""

from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris import (
    FlorisModel,
    TimeSeries,
)
from floris.par_floris_model import ParFlorisModel


DEBUG = True

if __name__ == "__main__":
    # Create random wind data
    np.random.seed(0)
    n_findex = 10 if DEBUG else 1000
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
    N = 20 if DEBUG else 100

    X, Y = np.meshgrid(
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
        5.0 * fmodel.core.farm.rotor_diameters_sorted[0][0] * np.arange(0, N, 1),
    )

    # Set up new parallel Floris model
    print("Beginning multiprocessing test")
    t0 = timerpc()
    pfmodel_mp = ParFlorisModel(
        "../inputs/gch.yaml",
        max_workers=-1,
        n_wind_condition_splits=-1,
        interface="multiprocessing",
        print_timings=True,
    )
    pfmodel_mp.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)
    t1 = timerpc()
    pfmodel_mp.run()
    aep1 = pfmodel_mp.get_farm_AEP()
    t2 = timerpc()
    pfmodel_mp.set(layout_x=X.flatten()+10, layout_y=Y.flatten())
    pfmodel_mp.run()
    aep2 = pfmodel_mp.get_farm_AEP()
    t3 = timerpc()

    print(f"Multiprocessing (max_workers={pfmodel_mp.max_workers})")
    print(f"Time to set up: {t1-t0}")
    print(f"Time to run first: {t2-t1}")
    print(f"Time to run second: {t3-t2}")

    # When is the worker pool released, though??
    print("Beginning pathos test")
    t0 = timerpc()
    pfmodel_pathos = ParFlorisModel(
        "../inputs/gch.yaml",
        max_workers=-1,
        n_wind_condition_splits=-1,
        interface="pathos",
        print_timings=True,
    )
    pfmodel_pathos.set(layout_x=X.flatten(), layout_y=Y.flatten(), wind_data=time_series)
    t1 = timerpc()
    pfmodel_pathos.run()
    aep3 = pfmodel_pathos.get_farm_AEP()
    t2 = timerpc()
    pfmodel_pathos.set(layout_x=X.flatten()+10, layout_y=Y.flatten())
    pfmodel_pathos.run()
    aep4 = pfmodel_pathos.get_farm_AEP()
    t3 = timerpc()

    print(f"Pathos (max_workers={pfmodel_pathos.max_workers})")
    print(f"Time to set up: {t1-t0}")
    print(f"Time to run first: {t2-t1}")
    print(f"Time to run second: {t3-t2}")

    if np.isclose(aep1 + aep2 + aep3 + aep4, 4*aep4):
        print("AEPs are equal!")
