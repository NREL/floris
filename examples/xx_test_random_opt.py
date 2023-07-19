import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface
from floris.tools.optimization.layout_optimization.layout_optimization_random import (
    LayoutOptimizationRandom,
)


if __name__ == '__main__':
    # Set up FLORIS
    fi = FlorisInterface('inputs/gch.yaml')


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
    fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

    # Set the boundaries
    # The boundaries for the turbines, specified as vertices
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    # Set turbine locations to 4 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    layout_x = [0, 0, 6 * D, 6 * D]
    layout_y = [0, 4 * D, 0, 4 * D]
    fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

    # Perform the optimization
    layout_opt = LayoutOptimizationRandom(fi,
                                        boundaries,
                                            freq=freq,
                                            min_dist_D=5.,
                                            seconds_per_iteration=10,
                                            total_optimization_seconds=60.)
    layout_opt.describe()

    layout_opt.optimize()

    layout_opt.plot_layout_opt_results()

    plt.show()
