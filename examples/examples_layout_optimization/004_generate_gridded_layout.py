"""Example: Gridded layout design
This example shows a layout optimization that places as many turbines as
possible into a given boundary using a gridded layout pattern.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
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

    layout_opt = LayoutOptimizationGridded(
        fmodel,
        boundaries,
        min_dist_D=5., # results in spacing of 5*125.88 = 629.4 m
        min_dist=None, # Alternatively, can specify spacing directly in meters
    )

    layout_opt.optimize()

    # Note that the "initial" layout that is provided with the fmodel is
    # not used by the layout optimization.
    layout_opt.plot_layout_opt_results()

    plt.show()
