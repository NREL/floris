"""Example: Layout optimization with genetic random search
This example shows a layout optimization using the genetic random search
algorithm. It provides options for the users to try different distance
probability mass functions for the random search perturbations.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)


if __name__ == '__main__':
    # Set up FLORIS
    fmodel = FlorisModel('../inputs/gch.yaml')


    # Setup 72 wind directions with a random wind speed and frequency distribution
    wind_directions = np.arange(0, 360.0, 5.0)
    np.random.seed(1)
    wind_speeds = 8.0 + np.random.randn(1) * 0.0
    # Shape frequency distribution to match number of wind directions and wind speeds
    freq = (
        np.abs(
            np.sort(
                np.random.randn(len(wind_directions))
            )
        )
        .reshape( ( len(wind_directions), len(wind_speeds) ) )
    )
    freq = freq / freq.sum()
    fmodel.set(
        wind_data=WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            freq_table=freq,
            ti_table=0.06
        )
    )

    # Set the boundaries
    # The boundaries for the turbines, specified as vertices
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    # Set turbine locations to 4 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    layout_x = [0, 0, 6 * D, 6 * D]
    layout_y = [0, 4 * D, 0, 4 * D]
    fmodel.set(layout_x=layout_x, layout_y=layout_y)

    # Perform the optimization
    distance_pmf = None

    # Other options that users can try
    # 1.
    # distance_pmf = {"d": [100, 1000], "p": [0.8, 0.2]}
    # 2.
    # p = gamma.pdf(np.linspace(0, 900, 91), 15, scale=20); p = p/p.sum()
    # distance_pmf = {"d": np.linspace(100, 1000, 91), "p": p}

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.,
        seconds_per_iteration=10,
        total_optimization_seconds=60.,
        distance_pmf=distance_pmf
    )
    layout_opt.describe()
    layout_opt.plot_distance_pmf()

    layout_opt.optimize()

    layout_opt.plot_layout_opt_results()

    layout_opt.plot_progress()

    plt.show()
